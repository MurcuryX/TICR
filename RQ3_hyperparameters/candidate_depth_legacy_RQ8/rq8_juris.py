import os

# 1. \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf

import pandas as pd
import numpy as np
import ast
import torch
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'pile_of_law_shuffle.csv'
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
TOP_K_PER_PASS = 20
KS = [3, 5, 10, 20]
# C_VALUES: \u7814\u7a76\u68c0\u7d22\u6b21\u6570\u7684\u53d8\u5316。5000 \u4f1a\u81ea\u52a8\u89e6\u53d1\u8be5 Query \u7684\u6240\u6709\u63a2\u9488 (Full-3M)
C_VALUES = [1, 2, 3, 5, 8, 10, 20, 5000]
g = 3  # NLI Top-g \u805a\u5408

# \u6743\u91cd\u914d\u7f6e (\u540c\u6b65\u81ea\u4f60\u7684 juris_logits.py)
W_VEC = 0.88
W_LOGIC = 0.12
W_LOGIC_FULL = 0.5
W_LOGIC_ATOM = 0.5
W_DIRECT = 0.0
W_INDIRECT = 1.0


# ================= \u8f85\u52a9\u51fd\u6570 =================
def safe_eval(val):
    try:
        if pd.isna(val): return None
        s_val = str(val).strip()
        parsed = ast.literal_eval(s_val)
        return parsed if isinstance(parsed, list) else None
    except:
        return None


def force_list_str(val):
    parsed = safe_eval(val)
    if parsed is not None: return parsed
    return [] if pd.isna(val) else [str(val).strip()]


def get_text_hash(text):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


def flatten_list(lst):
    flat = []
    if lst is None: return []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        elif isinstance(item, str) and item.strip():
            flat.append(item)
    return flat


# ================= \u5b9e\u9a8c\u7cfb\u7edf =================
class Juris_RQ8_Evaluator:
    def __init__(self, df):
        self.df = df
        self.doc_id_map = {}
        self.corpus_hashes = {}
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)
        self.query_cache = []

    def build_corpus(self):
        print("🏗️ Building Corpus for Juris-Logic...")
        unique_docs, seen = [], {}
        doc_cols = [('entail', 'entail_atoms'), ('conflict', 'conflict_atoms')]
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            for col_txt, col_atom in doc_cols:
                texts = force_list_str(row.get(col_txt))
                atoms_pool = safe_eval(row.get(col_atom))
                is_nested = isinstance(atoms_pool, list) and len(atoms_pool) > 0 and isinstance(atoms_pool[0], list)
                for i, text in enumerate(texts):
                    h = get_text_hash(text)
                    if h not in seen:
                        doc_id = len(unique_docs)
                        unique_docs.append(text)
                        seen[h] = doc_id
                        cur_atoms = atoms_pool[i] if is_nested and i < len(atoms_pool) else atoms_pool
                        self.doc_id_map[doc_id] = {'text': text, 'atoms': flatten_list(cur_atoms)}
                        self.corpus_hashes[h] = doc_id
        self.corpus_texts = unique_docs
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True,
                                                      batch_size=64)

    def preprocess_queries(self):
        print("🛠️ Preprocessing Juris Queries & Probes...")
        for idx, row in self.df.iterrows():
            q_text = str(row.get('query', '')).strip()
            if not q_text: continue
            gt_ids = set()
            for t in force_list_str(row.get('conflict')):
                h = get_text_hash(t)
                if h in self.corpus_hashes: gt_ids.add(self.corpus_hashes[h])
            if not gt_ids: continue

            # \u83b7\u53d6\u751f\u6210\u7684\u53cd\u5411\u539f\u5b50\u4f5c\u4e3a\u63a2\u9488\u7d20\u6750
            raw_negs = safe_eval(row.get('negs_sourse_atoms')) or []
            if raw_negs and isinstance(raw_negs[0], str): raw_negs = [raw_negs]
            neg_atoms = flatten_list(raw_negs)

            # \u6784\u9020 3M \u4e2a\u6f5c\u5728\u63a2\u9488\u6587\u672c
            cand_probes = [f"{q_text} {str(a).strip()}" for a in neg_atoms if str(a).strip()]
            if not cand_probes: cand_probes = [q_text]

            instruction = "Represent this sentence for searching relevant passages: "
            q_emb_origin = self.embedder.encode(instruction + q_text, convert_to_tensor=True)
            probe_embs = self.embedder.encode([instruction + p for p in cand_probes], convert_to_tensor=True)

            self.query_cache.append({
                'sent0': q_text, 'q_emb_origin': q_emb_origin, 'probe_embs': probe_embs,
                'q_atoms': flatten_list(safe_eval(row.get('query_atoms'))),
                'all_neg_atoms': neg_atoms, 'gt_ids': gt_ids
            })

    def _get_fps_indices(self, q_emb, probe_embs, C):
        num_probes = probe_embs.shape[0]
        if C >= num_probes: return list(range(num_probes))
        selected_indices = []
        # \u7b2c\u4e00\u4e2a\u70b9：\u79bb\u539f\u53e5\u6700\u8fdc\u7684\u70b9
        cos_to_q = util.cos_sim(q_emb, probe_embs).squeeze(0)
        p1_idx = torch.argmin(cos_to_q).item()
        selected_indices.append(p1_idx)
        # \u8fed\u4ee3 FPS
        distances = 1 - util.cos_sim(probe_embs[p1_idx], probe_embs).squeeze(0)
        for _ in range(1, C):
            next_idx = torch.argmax(distances).item()
            selected_indices.append(next_idx)
            new_dist = 1 - util.cos_sim(probe_embs[next_idx], probe_embs).squeeze(0)
            distances = torch.min(distances, new_dist)
        return selected_indices

    def _batch_calculate_logic(self, candidate_ids, q_data):
        full_pairs, atom_pairs, atom_meta = [], [], []
        sent0 = q_data['sent0']
        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            full_pairs.append([sent0, doc_data['text']])
            d_atoms, q_atoms, q_negs = doc_data['atoms'], q_data['q_atoms'], q_data['all_neg_atoms']
            curr_ap, curr_ty = [], []
            if q_atoms:
                for da in d_atoms:
                    for qa in q_atoms: curr_ap.append([da, qa]); curr_ty.append(0)
            if q_negs:
                for da in d_atoms:
                    for qn in q_negs: curr_ap.append([da, qn]); curr_ty.append(1)
            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap), 'types': curr_ty})
            atom_pairs.extend(curr_ap)

        f_logits = self.nli_model.predict(full_pairs, show_progress_bar=False) if full_pairs else []
        a_logits = self.nli_model.predict(atom_pairs, show_progress_bar=False) if atom_pairs else []

        raw_scores = []
        for i in range(len(candidate_ids)):
            s_full = f_logits[i][0] if len(f_logits) > 0 else 0.0
            s_ind = 0.0
            m = atom_meta[i]
            if m['count'] > 0:
                sl = a_logits[m['start']: m['start'] + m['count']]
                inds = [sl[k][1] for k, t in enumerate(m['types']) if t == 1]
                if inds: s_ind = np.mean(sorted(inds)[-g:])
            s_atom = (W_DIRECT * 0.0) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))

        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-9:
            return ((raw_scores - np.mean(raw_scores)) / np.std(raw_scores)).tolist()
        return (raw_scores - np.mean(raw_scores)).tolist()

    def run_rq8(self):
        print(f"🚀 Starting RQ8 (Juris) Experiment for C: {C_VALUES}")
        final_history = []
        for C in C_VALUES:
            label = "Full" if C >= 1000 else str(C)
            print(f"\n--- Testing C = {label} ---")
            metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

            for q_data in tqdm(self.query_cache, desc=f"C={label}"):
                # 1. FPS \u91c7\u6837\u63a2\u9488
                fps_idx = self._get_fps_indices(q_data['q_emb_origin'], q_data['probe_embs'], C)
                target_embs = q_data['probe_embs'][fps_idx]

                # 2. \u7b2c\u4e00\u9636\u6bb5\u68c0\u7d22 (Max-Pooling)
                sim_matrix = util.cos_sim(target_embs, self.corpus_embeddings)
                max_sims, _ = torch.max(sim_matrix, dim=0)
                cand_ids = torch.topk(max_sims, k=min(TOP_K_PER_PASS, len(self.corpus_texts))).indices.tolist()

                # 3. \u7b2c\u4e8c\u9636\u6bb5\u91cd\u6392
                s_vecs = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings[cand_ids]).squeeze(0).cpu().numpy()
                s_logics = self._batch_calculate_logic(cand_ids, q_data)

                final_rank = []
                for idx, cid in enumerate(cand_ids):
                    score = (W_VEC * s_vecs[idx]) + (W_LOGIC * s_logics[idx])
                    final_rank.append({'is_gt': cid in q_data['gt_ids'], 'score': score})
                final_rank.sort(key=lambda x: x['score'], reverse=True)

                for k in KS:
                    top_k = final_rank[:k]
                    r = 1.0 if any(it['is_gt'] for it in top_k) else 0.0
                    dcg = sum(1.0 / np.log2(rank + 2) for rank, it in enumerate(top_k) if it['is_gt'])
                    idcg = sum(1.0 / np.log2(n + 2) for n in range(min(len(q_data['gt_ids']), k)))
                    metrics[k]['recall'].append(r)
                    metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            # \u6253\u5370\u672c\u8f6e\u7ed3\u679c
            res_row = {'C': label}
            print(f"Result for C={label}:")
            for k in KS:
                r_val, n_val = np.mean(metrics[k]['recall']) * 100, np.mean(metrics[k]['ndcg']) * 100
                res_row[f'R@{k}'], res_row[f'N@{k}'] = r_val, n_val
                print(f"  K={k:2d} | Recall: {r_val:.2f} | NDCG: {n_val:.2f}")
            final_history.append(res_row)

        df_final = pd.DataFrame(final_history)
        df_final.to_csv('rq8_juris_results.csv', index=False)
        print("\n" + "=" * 50 + "\nFinal RQ8 Summary Table (Juris):\n" + df_final.to_string(index=False))


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        evaluator = Juris_RQ8_Evaluator(df)
        evaluator.build_corpus()
        evaluator.preprocess_queries()
        evaluator.run_rq8()
    else:
        print(f"❌ File not found: {CSV_PATH}")