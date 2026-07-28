import os

# 1. \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf

import pandas as pd
import numpy as np
import ast
import torch
import re
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
import hashlib

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'RQ5_real.csv'  # \u66ff\u6362\u4e3a\u60a8\u7684\u771f\u5b9e\u6587\u4ef6\u540d

# \u6a21\u578b\u914d\u7f6e
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570 (\u4fdd\u6301 inv_3m_std.py \u539f\u53c2\u6570)
TOP_K_PER_PASS = 20
KS = [3, 5, 10, 20]
NUM_CHUNKS = 5
g = 1

# 🔥 \u6838\u5fc3\u53c2\u6570 (\u4e25\u683c\u4fdd\u7559\u60a8\u7684\u903b\u8f91)
W_VEC = 0.9
W_LOGIC = 0.1
W_LOGIC_FULL = 0.05
W_LOGIC_ATOM = 0.95
W_DIRECT = 0.0
W_INDIRECT = 1.0

# ================= RQ5 \u5217\u540d\u6620\u5c04 =================
COL_MAPPING = {
    'query_text': 'col1_query_en',
    'gt_doc_text': 'col2_conflict_en',
    'distractor_cols': [
        'col3_entail1_en',
        'col4_entail2_en',
        'col5_entail3_en'
    ],
    'query_neg_atoms': 'col11_query_negations'
}


# ================= \u8f85\u52a9\u51fd\u6570 =================
def parse_nested_negations(val):
    """
    \u89e3\u6790 RQ5 \u6570\u636e\u96c6\u4e2d\u7684 col11，\u786e\u4fdd\u8fd4\u56de\u5d4c\u5957\u5217\u8868\u7ed3\u6784 [[atom1_neg...], [atom2_neg...]]
    \u517c\u5bb9 Python List \u5b57\u7b26\u4e32\u548c\u7eaf\u6587\u672c\u683c\u5f0f
    """
    if pd.isna(val): return []
    s_val = str(val).strip()

    # 1. \u5c1d\u8bd5 Python List \u89e3\u6790
    if s_val.startswith('[') and s_val.endswith(']'):
        try:
            parsed = ast.literal_eval(s_val)
            if isinstance(parsed, list):
                # \u517c\u5bb9\u5904\u7406：\u786e\u4fdd\u662f\u5d4c\u5957\u5217\u8868
                if len(parsed) > 0 and isinstance(parsed[0], str):
                    return [parsed]  # \u89c6\u4e3a 1 \u4e2a\u539f\u5b50\u7684\u53cd\u9a73\u5217\u8868
                return parsed  # \u89c6\u4e3a N \u4e2a\u539f\u5b50\u7684\u53cd\u9a73\u5217\u8868
        except:
            pass

            # 2. \u6587\u672c\u89e3\u6790 (Fallback)
    groups = []
    current_group = []
    lines = s_val.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue

        # \u5206\u7ec4\u6807\u8bb0\u68c0\u6d4b
        if "Atoms" in line and "Negations:" in line:
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        clean_line = ""
        if line.startswith("- "):
            clean_line = line[2:].strip()
        elif re.match(r'^\d+\.\s', line):
            clean_line = re.sub(r'^\d+\.\s', '', line).strip()
        elif len(line) > 5:
            clean_line = line

        if clean_line:
            current_group.append(clean_line)

    if current_group:
        groups.append(current_group)

    return groups


def flatten_list(lst):
    flat = []
    if lst is None: return []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        elif isinstance(item, str) and item.strip():
            flat.append(item)
    return flat


def get_text_hash(text):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


# ================= \u5b9e\u9a8c\u7cfb\u7edf =================
class TICR_Final_Evaluator:
    def __init__(self, df):
        self.df = df
        self.doc_id_map = {}
        self.corpus_hashes = {}
        self.text_to_id = {}  # RQ5 \u65b0\u589e：\u7528\u4e8e\u5feb\u901f\u67e5\u627e GT ID
        print(f"🔄 Loading Models on {DEVICE}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)
        self.query_cache = []

    def build_corpus(self):
        print("🏗️ Building Corpus (Conflict + Entailment)...")
        unique_docs = []

        # \u6536\u96c6 RQ5 \u4e2d\u7684\u6240\u6709\u6587\u6863\u5217 (GT + \u5e72\u6270\u9879)
        all_doc_cols = [COL_MAPPING['gt_doc_text']] + COL_MAPPING['distractor_cols']

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            for col in all_doc_cols:
                text = str(row.get(col, '')).strip()
                if not text or text.lower() == 'nan': continue

                h = get_text_hash(text)
                if h not in self.corpus_hashes:
                    doc_id = len(unique_docs)
                    unique_docs.append(text)
                    self.corpus_hashes[h] = doc_id
                    # RQ5 \u6570\u636e\u6ca1\u6709 doc_atoms，atoms \u8bbe\u4e3a\u7a7a\u5217\u8868
                    self.doc_id_map[doc_id] = {'text': text, 'atoms': []}

        self.corpus_texts = unique_docs
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True)
        print(f"✅ Corpus built with {len(unique_docs)} unique documents.")

    def preprocess_queries(self):
        print("🛠️ Preprocessing Queries...")
        q_col = COL_MAPPING['query_text']
        gt_col = COL_MAPPING['gt_doc_text']
        neg_col = COL_MAPPING['query_neg_atoms']

        valid_count = 0
        for idx, row in self.df.iterrows():
            q_text = str(row.get(q_col, '')).strip()
            if not q_text: continue

            # \u67e5\u627e GT ID
            gt_text = str(row.get(gt_col, '')).strip()
            gt_hash = get_text_hash(gt_text)

            gt_ids = set()
            if gt_hash in self.corpus_hashes:
                gt_ids.add(self.corpus_hashes[gt_hash])

            # \u89e3\u6790\u8d1f\u5411\u539f\u5b50 (\u4f7f\u7528 RQ5 \u4e13\u7528\u89e3\u6790\u5668)
            nested_negs = parse_nested_negations(row.get(neg_col))

            instruction = "Represent this sentence for searching relevant passages: "
            q_emb_origin = self.embedder.encode(instruction + q_text, convert_to_tensor=True)

            self.query_cache.append({
                'sent0': q_text,
                'q_emb_origin': q_emb_origin,
                'q_atoms': [],  # RQ5 \u672a\u4f7f\u7528 query atoms
                'raw_negs': nested_negs,  # \u4fdd\u7559\u5d4c\u5957\u7ed3\u6784
                'gt_ids': gt_ids
            })
            if gt_ids: valid_count += 1
        print(f"✅ Processed {len(self.query_cache)} queries (Valid GT found for {valid_count}).")

    def _batch_calculate_logic(self, candidate_ids, q_data):
        full_pairs, atom_pairs, atom_meta = [], [], []
        sent0 = q_data['sent0']
        # \u5c55\u5e73\u8d1f\u5411\u539f\u5b50\u7528\u4e8e\u8bc4\u5206
        q_negs = flatten_list(q_data['raw_negs'])

        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            doc_text = doc_data['text']
            # Full Text Pair
            full_pairs.append([sent0, doc_text])

            # Atom Pair (Doc Text vs Neg Atom)
            curr_ap, curr_ty = [], []
            if q_negs:
                # RQ5: Doc Text \u4f5c\u4e3a\u524d\u63d0，Neg Atom \u4f5c\u4e3a\u5047\u8bbe
                # \u9650\u5236\u524d10\u4e2a\u6700\u5f3a\u539f\u5b50，\u9632\u6b62 OOM
                for qn in q_negs[:10]:
                    curr_ap.append([doc_text, qn])
                    curr_ty.append(1)  # \u6807\u8bb0\u4e3a\u8d1f\u5411\u68c0\u67e5

            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap), 'types': curr_ty})
            atom_pairs.extend(curr_ap)

        # \u6279\u91cf\u63a8\u7406
        f_probs = torch.softmax(torch.tensor(self.nli_model.predict(full_pairs, show_progress_bar=False)),
                                dim=1).numpy()
        a_probs = torch.softmax(torch.tensor(self.nli_model.predict(atom_pairs, show_progress_bar=False)),
                                dim=1).numpy() if atom_pairs else []

        raw_scores = []
        for i in range(len(candidate_ids)):
            # 🔥 \u4fdd\u6301 inv_3m_std.py \u7684\u7d22\u5f15\u903b\u8f91
            # [0] = Contradiction (cross-encoder/nli-deberta-v3-base)
            s_full = f_probs[i][0]
            s_ind = 0.0

            m = atom_meta[i]
            if m['count'] > 0:
                sl = a_probs[m['start']: m['start'] + m['count']]
                ty = m['types']
                # 🔥 \u4fdd\u6301 inv_3m_std.py \u7684\u7d22\u5f15\u903b\u8f91
                # [1] = Entailment (\u6211\u4eec\u671f\u671b Doc Entails Neg_Atom)
                inds = [sl[k][1] for k, t in enumerate(ty) if t == 1]
                if inds:
                    raw = np.mean(sorted(inds)[-g:])
                    s_ind = raw

            s_atom = (W_DIRECT * 0.0) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))

        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-9:
            return ((raw_scores - np.mean(raw_scores)) / np.std(raw_scores)).tolist()
        return (raw_scores - np.mean(raw_scores)).tolist()

    def run_split_evaluation(self):
        total_q = len(self.query_cache)
        chunk_size = total_q // NUM_CHUNKS
        experiment_results = []

        print(f"🚀 Running {NUM_CHUNKS} Experiments using 3M-Pass Max-Pooling Retrieval...")

        for i in range(NUM_CHUNKS):
            # \u5904\u7406\u5c0f\u6570\u636e\u96c6\u5207\u5206\u53ef\u80fd\u4e3a\u7a7a\u7684\u60c5\u51b5
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q
            if chunk_size == 0:  # \u5982\u679c\u6570\u636e\u96c6\u6781\u5c0f，\u5168\u91cf\u8dd1\u4e00\u6b21
                if i == NUM_CHUNKS - 1:
                    start, end = 0, total_q
                else:
                    continue

            chunk_queries = self.query_cache[start:end]
            if not chunk_queries: continue

            chunk_metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

            for q_data in tqdm(chunk_queries, desc=f"Exp {i + 1}/5", leave=False):
                # ==========================================================
                # 🔥 1. 3M-Pass Max-Pooling Retrieval (\u5b8c\u5168\u7167\u642c)
                # ==========================================================
                cand_queries = []
                # \u904d\u5386\u6240\u6709\u5d4c\u5957\u5217\u8868\u4e2d\u7684\u6240\u6709\u539f\u5b50
                if q_data['raw_negs']:
                    for sub_list in q_data['raw_negs']:
                        for atom in sub_list:
                            atom_str = str(atom).strip()
                            if atom_str:
                                cand_queries.append(f"{q_data['sent0']} {atom_str}")

                if not cand_queries:
                    cand_queries = [q_data['sent0']]

                # \u6279\u91cf\u7f16\u7801
                instruction = "Represent this sentence for searching relevant passages: "
                q_embs_3m = self.embedder.encode([instruction + q for q in cand_queries],
                                                 convert_to_tensor=True)

                sim_matrix = util.cos_sim(q_embs_3m, self.corpus_embeddings)
                max_sims, _ = torch.max(sim_matrix, dim=0)

                top_k_indices = torch.topk(max_sims, k=min(TOP_K_PER_PASS, len(self.corpus_texts))).indices.tolist()
                cand_ids = top_k_indices

                # ==========================================================
                # 2. \u7b2c\u4e8c\u9636\u6bb5\u91cd\u6392 (\u5b8c\u5168\u7167\u642c)
                # ==========================================================
                s_vecs = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings[cand_ids]).squeeze(0).cpu().numpy()
                s_logics = self._batch_calculate_logic(cand_ids, q_data)

                final_rank = []
                for idx, cid in enumerate(cand_ids):
                    score = (W_VEC * s_vecs[idx]) + (W_LOGIC * s_logics[idx])
                    final_rank.append({'is_gt': cid in q_data['gt_ids'], 'score': score})

                final_rank.sort(key=lambda x: x['score'], reverse=True)

                for k in KS:
                    top_k = final_rank[:k]
                    chunk_metrics[k]['recall'].append(1.0 if any(r['is_gt'] for r in top_k) else 0.0)
                    dcg = sum(1.0 / np.log2(rank + 2) for rank, r in enumerate(top_k) if r['is_gt'])
                    idcg = sum(1.0 / np.log2(n + 2) for n in range(min(len(q_data['gt_ids']), k)))
                    chunk_metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            # \u8f93\u51fa Chunk \u7ed3\u679c
            if chunk_metrics[3]['recall']:
                c_r3 = np.mean(chunk_metrics[3]['recall']) * 100
                print(f"✅ Chunk {i + 1} Finished | Max-Pooling R@3: {c_r3:.2f}")
                experiment_results.append(
                    {k: {m: np.mean(chunk_metrics[k][m]) for m in ['recall', 'ndcg']} for k in KS})

        self._print_final_summary(experiment_results)

    def _print_final_summary(self, results):
        if not results:
            print("No results generated.")
            return
        print("\n" + "=" * 110)
        print(f"📊 Final Results (Averaged across Experiments) | Scale: x100")
        print("-" * 110)
        header = f"{'Metric':<15}"
        for k in KS: header += f" | k={k:<20}"
        print(header + "\n" + "-" * 110)

        for m_name in ['recall', 'ndcg']:
            line = f"{m_name.upper():<15}"
            for k in KS:
                data = [res[k][m_name] * 100 for res in results]
                line += f" | {np.mean(data):05.2f} (±{np.std(data):05.2f})"
            print(line)
        print("=" * 110)


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # \u7b80\u5355\u586b\u5145\u7a7a\u503c
        df.fillna('', inplace=True)
        evaluator = TICR_Final_Evaluator(df)
        evaluator.build_corpus()
        evaluator.preprocess_queries()
        evaluator.run_split_evaluation()
    else:
        print(f"❌ File not found: {CSV_PATH}")