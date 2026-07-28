import os

# 1. \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf
import pandas as pd
import numpy as np
import ast
import torch
import hashlib
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'pile_of_law_shuffle.csv'  # ✅ \u4fee\u6539\u4e3a Pile-of-Law \u6570\u636e\u96c6

# \u6a21\u578b\u914d\u7f6e
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'  # \u4fdd\u6301\u4e0erq7\u4e00\u81f4，\u6216\u8005\u6839\u636e\u663e\u5b58\u6539\u4e3alarge
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
TOP_K_PER_PASS = 20
KS = [3, 5, 10, 20]
NUM_CHUNKS = 1  # \u5206\u62105\u4efd\u8ba1\u7b97\u6807\u51c6\u5dee


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


# ================= \u7eaf\u53ec\u56de\u9636\u6bb5\u6d88\u878d\u5b9e\u9a8c\u7cfb\u7edf (Pile of Law) =================
class PileOfLaw_Recall_Ablation:
    def __init__(self, df):
        self.df = df
        self.doc_id_map = {}
        self.corpus_hashes = {}
        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL_NAME} on {DEVICE}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.query_cache = []
        self.corpus_embeddings = None
        self.corpus_texts = []

    def build_corpus(self):
        print("🏗️ Building Corpus from 'entail' and 'conflict' columns...")
        unique_docs, seen = [], {}

        # ✅ \u9002\u914d Pile of Law \u7684\u5217\u540d\u7ed3\u6784
        # entail: \u5e72\u6270\u9879 (Distractors)
        # conflict: \u76ee\u6807\u9879 (Ground Truth)
        doc_cols = [('entail', 'entail_atoms'), ('conflict', 'conflict_atoms')]

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            for col_txt, col_atom in doc_cols:
                texts = force_list_str(row.get(col_txt))
                atoms_pool = safe_eval(row.get(col_atom))

                # \u5904\u7406\u5d4c\u5957\u5217\u8868\u7684\u60c5\u51b5 (Pile of Law \u6570\u636e\u96c6\u4e2d\u5e38\u89c1)
                is_nested = isinstance(atoms_pool, list) and len(atoms_pool) > 0 and isinstance(atoms_pool[0], list)

                for i, text in enumerate(texts):
                    h = get_text_hash(text)
                    if h not in seen:
                        doc_id = len(unique_docs)
                        unique_docs.append(text)
                        seen[h] = doc_id

                        # \u83b7\u53d6\u5bf9\u5e94\u7684 atom
                        cur_atoms = atoms_pool[i] if is_nested and i < len(atoms_pool) else atoms_pool
                        self.doc_id_map[doc_id] = {'text': text, 'atoms': flatten_list(cur_atoms)}
                        self.corpus_hashes[h] = doc_id

        self.corpus_texts = unique_docs
        print(f"📦 Encoding Corpus ({len(unique_docs)} docs)...")
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True)

    def preprocess_queries(self):
        print("🛠️ Preprocessing Queries (Pile of Law Mapping)...")
        for idx, row in self.df.iterrows():
            # ✅ \u83b7\u53d6 Query
            q_text = str(row.get('query', '')).strip()
            if not q_text: continue

            # ✅ \u83b7\u53d6 GT IDs (\u76ee\u6807\u662f Conflict)
            gt_ids = set()
            conflict_texts = force_list_str(row.get('conflict'))
            for t in conflict_texts:
                h = get_text_hash(t)
                if h in self.corpus_hashes:
                    gt_ids.add(self.corpus_hashes[h])

            # \u5982\u679c\u6ca1\u6709\u627e\u5230 GT，\u8df3\u8fc7\u6b64 Query
            if not gt_ids: continue

            # ✅ \u83b7\u53d6\u53cd\u5411\u539f\u5b50 (\u7528\u4e8e 3M-Recall)
            raw_negs = safe_eval(row.get('negs_sourse_atoms')) or []
            if raw_negs and isinstance(raw_negs[0], str): raw_negs = [raw_negs]

            instruction = "Represent this sentence for searching relevant passages: "
            q_emb_origin = self.embedder.encode(instruction + q_text, convert_to_tensor=True)

            self.query_cache.append({
                'sent0': q_text,
                'q_emb_origin': q_emb_origin,
                'raw_negs': raw_negs,
                'gt_ids': gt_ids
            })
        print(f"✅ Cached {len(self.query_cache)} valid queries.")

    def run_ablation_comparison(self):
        """
        \u7eaf\u53ec\u56de\u9636\u6bb5\u6d88\u878d\u5b9e\u9a8c：\u5bf9\u6bd4 Traditional vs 3M-Recall
        """
        total_q = len(self.query_cache)
        chunk_size = total_q // NUM_CHUNKS

        modes = ["Traditional", "3M_Recall"]
        final_summary = {}

        # \u5b9a\u4e49\u9700\u8981\u8ba1\u7b97\u7684\u6307\u6807\u5217\u8868
        metrics_list = ['recall', 'mrr', 'zero_hit', 'avg_rank']

        for mode in modes:
            print(f"\n🚀 Running Retrieval Mode: [{mode}]")

            chunk_results_for_std = []
            global_all_metrics = {k: {m: [] for m in metrics_list} for k in KS}

            for i in range(NUM_CHUNKS):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q
                chunk_queries = self.query_cache[start:end]

                if not chunk_queries: continue

                # \u521d\u59cb\u5316\u5f53\u524d chunk \u7684\u7f13\u5b58
                current_chunk_metrics = {k: {m: [] for m in metrics_list} for k in KS}

                for q_data in tqdm(chunk_queries, desc=f"Exp {i + 1}/{NUM_CHUNKS} ({mode})", leave=False):

                    scores = None

                    if mode == "Traditional":
                        # \u4f20\u7edf\u7684\u53cc\u7f16\u7801\u5668\u68c0\u7d22
                        scores = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings).squeeze(0)

                    elif mode == "3M_Recall":
                        # \u4f7f\u7528\u539f\u5b50\u589e\u5f3a\u7684\u68c0\u7d22
                        cand_queries = []
                        # \u5c55\u5e73\u5d4c\u5957\u5217\u8868
                        all_atoms = flatten_list(q_data['raw_negs'])

                        for atom in all_atoms:
                            atom_str = str(atom).strip()
                            if atom_str:
                                cand_queries.append(f"{q_data['sent0']} {atom_str}")

                        # \u5982\u679c\u6ca1\u6709\u6709\u6548\u7684\u539f\u5b50，\u56de\u9000\u5230\u539f\u53e5
                        if not cand_queries:
                            cand_queries = [q_data['sent0']]

                        instruction = "Represent this sentence for searching relevant passages: "
                        q_embs_3m = self.embedder.encode([instruction + q for q in cand_queries],
                                                         convert_to_tensor=True)
                        sim_matrix = util.cos_sim(q_embs_3m, self.corpus_embeddings)
                        # Max-Pooling
                        scores, _ = torch.max(sim_matrix, dim=0)

                    # Top-K \u68c0\u7d22
                    top_k_res = torch.topk(scores, k=min(TOP_K_PER_PASS, len(self.corpus_texts)))
                    cand_ids = top_k_res.indices.tolist()
                    gt_ids = q_data['gt_ids']

                    for k in KS:
                        current_cands = cand_ids[:k]

                        # \u627e\u5230\u6240\u6709\u547d\u4e2d\u7684\u4f4d\u7f6e (\u4e0b\u6807\u4ece0\u5f00\u59cb)
                        hits_indices = [idx for idx, cid in enumerate(current_cands) if cid in gt_ids]

                        # 1. RECALL (Standard Recall: \u547d\u4e2d\u6570 / GT\u603b\u6570)
                        recall_val = 1.0 if hits_indices else 0.0  # Pile of Law \u901a\u5e38\u53ea\u6709\u4e00\u4e2a GT，\u8fd9\u91cc\u7b80\u5316\u4e3a Hit@K

                        # 2. MRR (1 / (rank+1))，\u53ea\u53d6\u7b2c\u4e00\u4e2a\u547d\u4e2d\u7684
                        if hits_indices:
                            mrr_val = 1.0 / (hits_indices[0] + 1)
                        else:
                            mrr_val = 0.0

                        # 3. Zero Hit Rate (\u672a\u547d\u4e2d\u7387: \u672a\u547d\u4e2d\u4e3a1，\u547d\u4e2d\u4e3a0)
                        zero_hit_val = 1.0 if not hits_indices else 0.0

                        # 4. Avg First Rank (\u9996\u4e2a\u547d\u4e2d\u7684\u6392\u540d, 1-based)
                        first_rank_val = (hits_indices[0] + 1) if hits_indices else None

                        # \u4fdd\u5b58\u7ed3\u679c
                        for m_val, m_name in zip([recall_val, mrr_val, zero_hit_val, first_rank_val], metrics_list):
                            current_chunk_metrics[k][m_name].append(m_val)
                            global_all_metrics[k][m_name].append(m_val)

                # \u8ba1\u7b97 Chunk \u5747\u503c (\u7528\u4e8e std)
                chunk_means = {}
                for k in KS:
                    chunk_means[k] = {}
                    for m in metrics_list:
                        vals = current_chunk_metrics[k][m]
                        if m == 'avg_rank':
                            # \u8fc7\u6ee4 None
                            valid_vals = [v for v in vals if v is not None]
                            chunk_means[k][m] = np.mean(valid_vals) if valid_vals else 0.0
                        else:
                            chunk_means[k][m] = np.mean(vals)

                chunk_results_for_std.append(chunk_means)

            final_summary[mode] = (chunk_results_for_std, global_all_metrics)

        self._print_ablation_summary(final_summary, metrics_list)

    def _print_ablation_summary(self, summary_data, metrics_list):
        print("\n" + "=" * 120)
        print(f"📊 RETRIEVAL STAGE ABLATION (Pile of Law): Traditional vs. 3M Retrieval")
        print("=" * 120)

        # \u663e\u793a\u540d\u79f0\u6620\u5c04
        display_names = {
            'recall': 'Recall/Hit (%)',
            'mrr': 'MRR (%)',
            'zero_hit': 'Zero Hit Rate (%)',
            'avg_rank': 'Avg First Rank (Index)'
        }

        for m_key in metrics_list:
            d_name = display_names.get(m_key, m_key)
            print(f"\n🔹 Metric: {d_name}")
            print("-" * 120)

            header = f"{'Method':<20}"
            for k in KS: header += f" | k={k:<18}"
            print(header)
            print("-" * 120)

            for mode in ["Traditional", "3M_Recall"]:
                if mode not in summary_data: continue
                chunk_res, global_res = summary_data[mode]
                line = f"{mode:<20}"

                for k in KS:
                    vals = global_res[k][m_key]

                    # \u8ba1\u7b97\u5168\u5c40\u5747\u503c
                    if m_key == 'avg_rank':
                        valid_vals = [v for v in vals if v is not None]
                        mean_val = np.mean(valid_vals) if valid_vals else 0.0
                    else:
                        mean_val = np.mean(vals)

                    # \u8ba1\u7b97\u5206\u5757\u6807\u51c6\u5dee
                    chunk_vals = [r[k][m_key] for r in chunk_res]
                    std_val = np.std(chunk_vals)

                    # \u683c\u5f0f\u5316\u8f93\u51fa
                    if m_key == 'avg_rank':
                        line += f" | {mean_val:05.2f} (±{std_val:05.2f})  "
                    else:
                        line += f" | {mean_val * 100:05.2f} (±{std_val * 100:05.2f})  "

                print(line)
            print("-" * 120)


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        evaluator = PileOfLaw_Recall_Ablation(df)
        evaluator.build_corpus()
        evaluator.preprocess_queries()
        evaluator.run_ablation_comparison()
    else:
        print(f"❌ File not found: {CSV_PATH}")