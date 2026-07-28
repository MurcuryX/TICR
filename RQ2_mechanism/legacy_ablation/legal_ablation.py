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

# \u6a21\u578b\u914d\u7f6e
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
TOP_K_PER_PASS = 20
KS = [3, 5, 10, 20]
NUM_CHUNKS = 5  # ✅ \u4fdd\u7559\u5206\u5757\u903b\u8f91
g = 3

# 🔥 \u6838\u5fc3\u53c2\u6570 (\u586b\u5165\u4f60\u641c\u7d22\u51fa\u6765\u7684\u6700\u4f18\u53c2\u6570)
W_VEC = 0.9
W_LOGIC = 0.1
W_LOGIC_FULL = 0.5
W_LOGIC_ATOM = 0.5
W_DIRECT = 0.0
W_INDIRECT = 1.0


print(f"🔥 \u914d\u7f6e\u53c2\u6570: W_VEC={W_VEC}, W_LOGIC={W_LOGIC}, W_LOGIC_FULL={W_LOGIC_FULL}, W_LOGIC_ATOM={W_LOGIC_ATOM}")


# ================= \u8f85\u52a9\u51fd\u6570 =================
def safe_eval(val):
    try:
        if pd.isna(val): return None
        s_val = str(val).strip()
        if s_val.startswith('{'): return None
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
class TICR_Final_Evaluator:
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

            instruction = "Represent this sentence for searching relevant passages: "
            q_emb_origin = self.embedder.encode(instruction + sent0, convert_to_tensor=True)

            self.query_cache.append({
                'sent0': sent0, 'q_emb_origin': q_emb_origin,
                'q_atoms': flatten_list(safe_eval(row.get('Original_Text_atoms'))),
                'raw_negs': raw_negs, 'gt_ids': gt_ids
            })

    def _batch_calculate_logic(self, candidate_ids, q_data):
        full_pairs, atom_pairs, atom_meta = [], [], []
        sent0, q_atoms, q_negs = q_data['sent0'], q_data['q_atoms'], flatten_list(q_data['raw_negs'])

        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            full_pairs.append([sent0, doc_data['text']])
            d_atoms = doc_data['atoms']
            curr_ap, curr_ty = [], []
            if q_atoms:
                for da in d_atoms:
                    for qa in q_atoms: curr_ap.append([da, qa]); curr_ty.append(0)
            if q_negs:
                for da in d_atoms:
                    for qn in q_negs: curr_ap.append([da, qn]); curr_ty.append(1)
            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap), 'types': curr_ty})
            atom_pairs.extend(curr_ap)

        # 🔥 \u4fee\u6539\u70b9：\u76f4\u63a5\u4f7f\u7528 predict \u8fd4\u56de\u7684\u539f\u59cb Logits，\u79fb\u9664 torch.softmax
        f_logits = self.nli_model.predict(full_pairs, show_progress_bar=False)
        a_logits = self.nli_model.predict(atom_pairs, show_progress_bar=False) if atom_pairs else []

        raw_scores = []
        for i in range(len(candidate_ids)):
            # [0] \u5bf9\u5e94 Contradiction \u6807\u7b7e\u7684 Logit
            s_full = f_logits[i][0]
            s_ind = 0.0
            m = atom_meta[i]
            if m['count'] > 0:
                sl = a_logits[m['start']: m['start'] + m['count']]
                ty = m['types']
                # [1] \u5bf9\u5e94 Entailment \u6807\u7b7e\u7684 Logit
                inds = [sl[k][1] for k, t in enumerate(ty) if t == 1]
                if inds: s_ind = np.mean(sorted(inds)[-g:])

            s_atom = (W_DIRECT * 0.0) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))

        # Z-score \u5f52\u4e00\u5316\u5728\u5904\u7406 Logits \u65f6\u81f3\u5173\u91cd\u8981
        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-9:
            return ((raw_scores - np.mean(raw_scores)) / np.std(raw_scores)).tolist()
        return (raw_scores - np.mean(raw_scores)).tolist()

    def run_split_evaluation(self):
        total_q = len(self.query_cache)
        chunk_size = total_q // NUM_CHUNKS

        # 1. \u7528\u4e8e\u8ba1\u7b97\u65b9\u5dee (Std Dev) \u7684\u5206\u5757\u6570\u636e
        chunk_results_for_std = []

        # 2. \u7528\u4e8e\u8ba1\u7b97\u5747\u503c (Mean) \u7684\u5168\u5c40\u6570\u636e (\u89e3\u51b3\u6570\u503c\u53d8\u5c0f\u95ee\u9898)
        global_all_metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

        print(f"🚀 Running {NUM_CHUNKS} Experiments using 3M-Pass Max-Pooling Retrieval...")

        for i in range(NUM_CHUNKS):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q
            chunk_queries = self.query_cache[start:end]

            # \u5f53\u524d\u5206\u5757\u7684\u7edf\u8ba1\u7f13\u5b58
            current_chunk_metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

            for q_data in tqdm(chunk_queries, desc=f"Exp {i + 1}/5", leave=False):
                # --- \u68c0\u7d22\u903b\u8f91 (\u4fdd\u6301\u4e0d\u53d8) ---
                cand_queries = []
                for sub_list in q_data['raw_negs']:
                    for atom in sub_list:
                        atom_str = str(atom).strip()
                        if atom_str: cand_queries.append(f"{q_data['sent0']} {atom_str}")
                if not cand_queries: cand_queries = [q_data['sent0']]

                instruction = "Represent this sentence for searching relevant passages: "
                q_embs_3m = self.embedder.encode([instruction + q for q in cand_queries], convert_to_tensor=True)
                sim_matrix = util.cos_sim(q_embs_3m, self.corpus_embeddings)
                max_sims, _ = torch.max(sim_matrix, dim=0)
                cand_ids = torch.topk(max_sims, k=min(TOP_K_PER_PASS, len(self.corpus_texts))).indices.tolist()

                # --- \u91cd\u6392\u903b\u8f91 (\u4fdd\u6301\u4e0d\u53d8) ---
                s_vecs = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings[cand_ids]).squeeze(0).cpu().numpy()
                s_logics = self._batch_calculate_logic(cand_ids, q_data)

                final_rank = []
                for idx, cid in enumerate(cand_ids):
                    score = (W_VEC * s_vecs[idx]) + (W_LOGIC * s_logics[idx])
                    final_rank.append({'is_gt': cid in q_data['gt_ids'], 'score': score})
                final_rank.sort(key=lambda x: x['score'], reverse=True)

                # --- \u5173\u952e\u6539\u52a8：\u540c\u65f6\u5199\u5165\u5206\u5757\u7f13\u5b58\u548c\u5168\u5c40\u7f13\u5b58 ---
                for k in KS:
                    top_k = final_rank[:k]
                    recall_val = 1.0 if any(r['is_gt'] for r in top_k) else 0.0
                    dcg = sum(1.0 / np.log2(rank + 2) for rank, r in enumerate(top_k) if r['is_gt'])
                    idcg = sum(1.0 / np.log2(n + 2) for n in range(min(len(q_data['gt_ids']), k)))
                    ndcg_val = dcg / idcg if idcg > 0 else 0.0

                    # \u5199\u5165\u5206\u5757 (\u7528\u4e8e Std)
                    current_chunk_metrics[k]['recall'].append(recall_val)
                    current_chunk_metrics[k]['ndcg'].append(ndcg_val)

                    # \u5199\u5165\u5168\u5c40 (\u7528\u4e8e Mean)
                    global_all_metrics[k]['recall'].append(recall_val)
                    global_all_metrics[k]['ndcg'].append(ndcg_val)

            # \u4fdd\u5b58\u5f53\u524d\u5206\u5757\u7684\u5e73\u5747\u503c，\u7528\u4e8e\u540e\u7eed\u8ba1\u7b97 Std
            chunk_means = {k: {m: np.mean(current_chunk_metrics[k][m]) for m in ['recall', 'ndcg']} for k in KS}
            chunk_results_for_std.append(chunk_means)

            print(f"✅ Chunk {i + 1} Finished")

        self._print_hybrid_summary(chunk_results_for_std, global_all_metrics)

    def _print_hybrid_summary(self, chunk_results, global_metrics):
        print("\n" + "=" * 110)
        print(f"📊 Final Results (Global Mean ± Chunk Std) | Scale: x100")
        print("-" * 110)
        header = f"{'Metric':<15}"
        for k in KS: header += f" | k={k:<20}"
        print(header + "\n" + "-" * 110)

        for m_name in ['recall', 'ndcg']:
            line = f"{m_name.upper():<15}"
            for k in KS:
                # 1. \u5747\u503c：\u4f7f\u7528\u5168\u5c40\u6240\u6709\u6570\u636e\u7684\u5747\u503c (\u4e0e Search \u4ee3\u7801\u7ed3\u679c\u5bf9\u9f50)
                global_mean = np.mean(global_metrics[k][m_name]) * 100

                # 2. \u6807\u51c6\u5dee：\u4f7f\u7528 5 \u4e2a Chunk \u7684\u5747\u503c\u7684\u6807\u51c6\u5dee (\u4fdd\u7559\u9c81\u68d2\u6027\u8bc1\u660e)
                chunk_values = [res[k][m_name] * 100 for res in chunk_results]
                std_dev = np.std(chunk_values)

                line += f" | {global_mean:05.2f} (±{std_dev:05.2f})"
            print(line)
        print("=" * 110)


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    evaluator = TICR_Final_Evaluator(df)
    evaluator.build_corpus()
    evaluator.preprocess_queries()
    evaluator.run_split_evaluation()