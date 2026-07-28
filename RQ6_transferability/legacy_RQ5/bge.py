import os

# 1. \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf

import pandas as pd
import numpy as np
import ast
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import hashlib

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'RQ5_real.csv'  # \u4fee\u6539\u4e3a RQ5 \u771f\u5b9e\u6570\u636e

# \u6a21\u578b\u9009\u62e9: BGE-Large (Strong Baseline)
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
TOP_K_RECALL = 20
METRIC_KS = [3, 5, 10, 20]  # \u6839\u636e RQ5 \u7684\u5e38\u7528 K \u503c\u8c03\u6574，\u6216\u4fdd\u6301 [3, 5, 10, 20]
BATCH_SIZE = 32
NUM_CHUNKS = 5  # \u4fdd\u6301\u539f\u6709\u903b\u8f91：\u5207\u6210\u4e94\u4efd\u8fdb\u884c\u4e94\u6b21\u5b9e\u9a8c

# BGE \u4e13\u7528 Query \u6307\u4ee4 (\u4fdd\u6301\u4e0d\u53d8)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ================= \u8f85\u52a9\u51fd\u6570 =================
def force_str(val):
    if pd.isna(val) or str(val).lower() == 'nan': return ""
    return str(val).strip()


def get_text_hash(text):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


# ================= BGE \u57fa\u7ebf\u7cfb\u7edf (5-Part \u5b9e\u9a8c\u7248) =================
class BGE_Legal_Baseline:
    def __init__(self, df):
        self.df = df
        print(f"🔌 Loading model: {MODEL_NAME} on {DEVICE}...")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)

        self.corpus_texts = []
        self.corpus_hashes = {}  # hash -> doc_id
        self.corpus_embeddings = None
        self.query_to_gt = {}  # {query_text: gt_ids_set}

    def build_corpus(self):
        """\u9002\u914d RQ5_real.csv \u7684\u5217\u540d"""
        print("🏗️ Building Deduplicated Corpus (RQ5)...")
        unique_docs = []
        seen_hashes = {}

        # RQ5 \u7684\u6587\u6863\u5217：GT + \u5e72\u6270\u9879
        doc_cols = [
            'col2_conflict_en',  # Conflict (GT)
            'col3_entail1_en',  # Entailment 1
            'col4_entail2_en',  # Entailment 2
            'col5_entail3_en'  # Entailment 3
        ]

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            for col in doc_cols:
                val = force_str(row.get(col))
                if not val: continue
                h = get_text_hash(val)
                if h not in seen_hashes:
                    doc_id = len(unique_docs)
                    unique_docs.append(val)
                    seen_hashes[h] = doc_id
                    self.corpus_hashes[h] = doc_id

        self.corpus_texts = unique_docs
        print(f"🧮 Encoding Corpus ({len(unique_docs)} docs)...")
        self.corpus_embeddings = self.model.encode(
            unique_docs,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    def run_split_evaluation(self):
        """\u6267\u884c 5 \u6b21\u5207\u5206\u5b9e\u9a8c\u5e76\u7edf\u8ba1\u7ed3\u679c"""
        print("🔍 Mapping Queries to GT (RQ5)...")
        # RQ5 \u5217\u540d: Query -> col1_query_en, GT -> col2_conflict_en
        q_col = 'col1_query_en'
        gt_cols = ['col2_conflict_en']

        # 1. \u9884\u5904\u7406\u6240\u6709 Query
        valid_count = 0
        for _, row in self.df.iterrows():
            query_text = force_str(row.get(q_col))
            if not query_text: continue

            gt_ids = set()
            for col in gt_cols:
                t = force_str(row.get(col))
                if t:
                    h = get_text_hash(t)
                    if h in self.corpus_hashes:
                        gt_ids.add(self.corpus_hashes[h])

            if gt_ids:
                if query_text not in self.query_to_gt:
                    self.query_to_gt[query_text] = set()
                self.query_to_gt[query_text].update(gt_ids)
                valid_count += 1

        print(f"✅ Processed Queries with Valid GT: {valid_count}")

        query_texts = list(self.query_to_gt.keys())
        total_q = len(query_texts)
        chunk_size = total_q // NUM_CHUNKS

        all_chunk_metrics = []

        print(f"🚀 Splitting {total_q} queries into {NUM_CHUNKS} chunks for experiments...")

        for i in range(NUM_CHUNKS):
            # \u5904\u7406\u53ef\u80fd\u7684\u8fb9\u754c\u60c5\u51b5
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q

            if start >= total_q: break  # \u9632\u6b62\u6570\u636e\u8fc7\u5c11\u5bfc\u81f4\u7684\u7d22\u5f15\u8d8a\u754c

            sub_queries = query_texts[start:end]
            if not sub_queries: continue

            sub_gt_list = [self.query_to_gt[q] for q in sub_queries]

            print(f"   [Experiment {i + 1}/5] Processing Chunk Size: {len(sub_queries)}")

            # \u7f16\u7801\u5e26\u6307\u4ee4\u7684 Query
            sub_queries_with_inst = [QUERY_INSTRUCTION + q for q in sub_queries]
            sub_q_embs = self.model.encode(
                sub_queries_with_inst,
                batch_size=BATCH_SIZE,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            # \u8bed\u4e49\u641c\u7d22
            hits = util.semantic_search(
                sub_q_embs,
                self.corpus_embeddings,
                top_k=max(TOP_K_RECALL, max(METRIC_KS)),
                score_function=util.dot_score
            )

            # \u7edf\u8ba1\u672c\u8f6e\u6307\u6807
            round_results = {k: {"recall": [], "ndcg": []} for k in METRIC_KS}
            for j, hit_list in enumerate(hits):
                gt_ids = sub_gt_list[j]
                for k in METRIC_KS:
                    top_k_ids = [hit['corpus_id'] for hit in hit_list[:k]]
                    # Recall
                    recall_val = 1.0 if any(did in gt_ids for did in top_k_ids) else 0.0
                    round_results[k]["recall"].append(recall_val)

                    # NDCG
                    dcg = sum([1.0 / np.log2(rank + 2) for rank, did in enumerate(top_k_ids) if did in gt_ids])
                    idcg = sum([1.0 / np.log2(n + 2) for n in range(min(len(gt_ids), k))])
                    ndcg_val = dcg / idcg if idcg > 0 else 0.0
                    round_results[k]["ndcg"].append(ndcg_val)

            # \u8bb0\u5f55\u672c\u6b21\u5b9e\u9a8c\u5747\u503c (\u767e\u5206\u5236)
            all_chunk_metrics.append({
                k: {
                    "recall": np.mean(round_results[k]["recall"]) * 100,
                    "ndcg": np.mean(round_results[k]["ndcg"]) * 100
                } for k in METRIC_KS
            })

        self._print_final_summary(all_chunk_metrics)

    def _print_final_summary(self, results):
        """\u7edf\u8ba1\u5747\u503c\u4e0e\u6807\u51c6\u5dee，\u4fdd\u7559\u4e24\u4f4d\u5c0f\u6570"""
        if not results:
            print("No results to display.")
            return

        print("\n" + "=" * 110)
        print(f"📊 BGE Final Summary (Averaged across {len(results)} Chunks) | Model: {MODEL_NAME}")
        print("-" * 110)
        header = f"{'Metric (%)':<15}"
        for k in sorted(METRIC_KS): header += f" | k={k:<20}"
        print(header + "\n" + "-" * 110)

        r_line, n_line = f"{'Recall@k':<15}", f"{'NDCG@k':<15}"
        for k in sorted(METRIC_KS):
            rs = [res[k]["recall"] for res in results]
            ns = [res[k]["ndcg"] for res in results]
            r_line += f" | {np.mean(rs):05.2f} (±{np.std(rs):05.2f})"
            n_line += f" | {np.mean(ns):05.2f} (±{np.std(ns):05.2f})"

        print(r_line + "\n" + n_line + "\n" + "=" * 110)


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # \u7b80\u5355\u586b\u5145\u9632\u6b62 NaN \u62a5\u9519
            df.fillna('', inplace=True)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            exit()

        system = BGE_Legal_Baseline(df)
        system.build_corpus()
        system.run_split_evaluation()
    else:
        print(f"❌ File not found: {CSV_PATH}")