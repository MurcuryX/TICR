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
CSV_PATH = 'Legal_Contra.csv'
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
TOP_K_PER_PASS = 20
KS = [3, 5, 10, 20]
C_VALUES = [1, 2, 3, 5, 8, 10, 20, 5000] # \u7814\u7a76 C \u7684\u53d8\u5316
g = 3

# \u6838\u5fc3\u6743\u91cd
W_VEC = 0.9
W_LOGIC = 0.1
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
        if isinstance(item, list): flat.extend(flatten_list(item))
        elif isinstance(item, str) and item.strip(): flat.append(item)
    return flat

# ================= \u5b9e\u9a8c\u7cfb\u7edf =================
class RQ8_Evaluator:
    def __init__(self, df):
        self.df = df
        self.doc_id_map = {}
        self.corpus_hashes = {}
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)
        self.query_cache = []

    def build_corpus(self):
        print("🏗️ Building Corpus...")
        unique_docs, seen = [], {}
        doc_cols = [
            ('Conflict1 (Contradiction)', 'Conflict1 (Contradiction)_atoms'),
            ('Conflict2 (Contradiction)', 'Conflict2 (Contradiction)_atoms'),
            ('Paraphrase_Structure_1 (Entailment)', 'Paraphrase_Structure_1 (Entailment)_atoms'),
            ('Paraphrase_Structure_2 (Entailment)', 'Paraphrase_Structure_2 (Entailment)_atoms')
        ]
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
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True)

    def preprocess_queries(self):
        print("🛠️ Preprocessing Queries...")
        for idx, row in self.df.iterrows():
            q_raw = force_list_str(row.get('Original_Text'))
            if not q_raw: continue
            sent0 = q_raw[0]
            gt_ids = set()
            for col in ['Conflict1 (Contradiction)', 'Conflict2 (Contradiction)']:
                for t in force_list_str(row.get(col)):
                    h = get_text_hash(t)
                    if h in self.corpus_hashes: gt_ids.add(self.corpus_hashes[h])
            if not gt_ids: continue

            raw_negs = safe_eval(row.get('Neg_Original_Text_atoms')) or []
            if raw_negs and isinstance(raw_negs[0], str): raw_negs = [raw_negs]
            
            cand_queries = []
            for sub_list in raw_negs:
                for atom in sub_list:
                    atom_str = str(atom).strip()
                    if atom_str: cand_queries.append(f"{sent0} {atom_str}")
            
            if not cand_queries: continue

            instruction = "Represent this sentence for searching relevant passages: "
            q_emb_origin = self.embedder.encode(instruction + sent0, convert_to_tensor=True)
            probe_embs = self.embedder.encode([instruction + pq for pq in cand_queries], convert_to_tensor=True)

            self.query_cache.append({
                'sent0': sent0, 'q_emb_origin': q_emb_origin, 'probe_embs': probe_embs,
                'q_atoms': flatten_list(safe_eval(row.get('Original_Text_atoms'))),
                'raw_negs': raw_negs, 'gt_ids': gt_ids
            })

    def _get_fps_indices(self, q_emb, probe_embs, C):
        num_probes = probe_embs.shape[0]
        if C >= num_probes: return list(range(num_probes))
        selected_indices = []
        cos_to_q = util.cos_sim(q_emb, probe_embs).squeeze(0)
        p1_idx = torch.argmin(cos_to_q).item()
        selected_indices.append(p1_idx)
        distances = 1 - util.cos_sim(probe_embs[p1_idx], probe_embs).squeeze(0)
        for _ in range(1, C):
            next_idx = torch.argmax(distances).item()
            selected_indices.append(next_idx)
            new_dist = 1 - util.cos_sim(probe_embs[next_idx], probe_embs).squeeze(0)
            distances = torch.min(distances, new_dist)
        return selected_indices

    def _batch_calculate_logic(self, candidate_ids, q_data):
        full_pairs, atom_pairs, atom_meta = [], [], []
        sent0, q_atoms, q_negs = q_data['sent0'], q_data['q_atoms'], flatten_list(q_data['raw_negs'])
        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            full_pairs.append([sent0, doc_data['text']])
            d_atoms = doc_data['atoms']
            curr_ap, curr_ty = [] , []
            if q_atoms:
                for da in d_atoms:
                    for qa in q_atoms: curr_ap.append([da, qa]); curr_ty.append(0)
            if q_negs:
                for da in d_atoms:
                    for qn in q_negs: curr_ap.append([da, qn]); curr_ty.append(1)
            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap), 'types': curr_ty})
            atom_pairs.extend(curr_ap)
        f_logits = self.nli_model.predict(full_pairs, show_progress_bar=False)
        a_logits = self.nli_model.predict(atom_pairs, show_progress_bar=False) if atom_pairs else []
        raw_scores = []
        for i in range(len(candidate_ids)):
            s_full = f_logits[i][0]
            s_ind = 0.0
            m = atom_meta[i]
            if m['count'] > 0:
                sl = a_logits[m['start']: m['start'] + m['count']]
                ty = m['types']
                inds = [sl[k][1] for k, t in enumerate(ty) if t == 1]
                if inds: s_ind = np.mean(sorted(inds)[-g:])
            s_atom = (W_DIRECT * 0.0) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))
        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-9:
            return ((raw_scores - np.mean(raw_scores)) / np.std(raw_scores)).tolist()
        return (raw_scores - np.mean(raw_scores)).tolist()

    def run_rq8_evaluation(self):
        print(f"🚀 Running RQ8 Evaluation for C values: {C_VALUES}")
        results_summary = []

        # \u5916\u5c42\u8fdb\u5ea6\u6761：\u76d1\u63a7\u4e0d\u540c C \u503c\u7684\u8fdb\u5ea6
        c_pbar = tqdm(C_VALUES, desc="Overall RQ8 Progress", position=0)

        for C in c_pbar:
            label = "Full" if C >= 1000 else str(C)
            c_pbar.set_description(f"Processing C={label}")

            metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

            # \u5185\u5c42\u8fdb\u5ea6\u6761：\u76d1\u63a7\u5f53\u524d C \u503c\u4e0b Query \u7684\u5904\u7406\u8fdb\u5ea6
            for q_data in tqdm(self.query_cache, desc=f"  Evaluating C={label}", leave=False, position=1):
                # 1. FPS \u91c7\u6837
                fps_indices = self._get_fps_indices(q_data['q_emb_origin'], q_data['probe_embs'], C)
                target_embs = q_data['probe_embs'][fps_indices]

                # 2. \u68c0\u7d22 (Max-Pooling)
                sim_matrix = util.cos_sim(target_embs, self.corpus_embeddings)
                max_sims, _ = torch.max(sim_matrix, dim=0)
                cand_ids = torch.topk(max_sims, k=min(TOP_K_PER_PASS, len(self.corpus_texts))).indices.tolist()

                # 3. \u91cd\u6392
                s_vecs = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings[cand_ids]).squeeze(0).cpu().numpy()
                s_logics = self._batch_calculate_logic(cand_ids, q_data)

                final_rank = []
                for idx, cid in enumerate(cand_ids):
                    score = (W_VEC * s_vecs[idx]) + (W_LOGIC * s_logics[idx])
                    final_rank.append({'is_gt': cid in q_data['gt_ids'], 'score': score})
                final_rank.sort(key=lambda x: x['score'], reverse=True)

                for k in KS:
                    top_k = final_rank[:k]
                    recall_val = 1.0 if any(r['is_gt'] for r in top_k) else 0.0
                    dcg = sum(1.0 / np.log2(rank + 2) for rank, r in enumerate(top_k) if r['is_gt'])
                    idcg = sum(1.0 / np.log2(n + 2) for n in range(min(len(q_data['gt_ids']), k)))
                    metrics[k]['recall'].append(recall_val)
                    metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            # --- \u6bcf\u4e00\u8f6e\u7ed3\u675f\u6253\u5370\u8be5 C \u503c\u7684\u7ed3\u679c ---
            c_result = {'C': label}
            tqdm.write(f"\n✨ Round Finished: C={label}")
            for k in KS:
                r_avg = np.mean(metrics[k]['recall']) * 100
                n_avg = np.mean(metrics[k]['ndcg']) * 100
                c_result[f'R@{k}'] = r_avg
                c_result[f'N@{k}'] = n_avg
                tqdm.write(f"  K={k:2d} | Recall: {r_avg:5.2f}% | NDCG: {n_avg:5.2f}%")

            results_summary.append(c_result)

        # \u6700\u7ec8\u6c47\u603b
        final_df = pd.DataFrame(results_summary)
        print("\n" + "=" * 60 + "\n📊 Final RQ8 Results Summary Table:\n" + final_df.to_string(index=False))
        final_df.to_csv('rq8_legal_results.csv', index=False)
        return final_df

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"❌ {CSV_PATH} not found.")
    else:
        df = pd.read_csv(CSV_PATH)
        evaluator = RQ8_Evaluator(df)
        evaluator.build_corpus()
        evaluator.preprocess_queries()
        evaluator.run_rq8_evaluation()
