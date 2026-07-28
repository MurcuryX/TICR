import os

# 1. \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf

import pandas as pd
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import hashlib

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'RQ5_real.csv'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-large'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
METRIC_KS = [3, 5, 10, 20]
NUM_CHUNKS = 5
BATCH_SIZE = 32  # \u663e\u5b58\u5360\u7528\u63a7\u5236

# NLI \u77db\u76fe\u6807\u7b7e\u7d22\u5f15 (nli-deberta-v3-base \u6807\u51c6\u4e3a 0)
CON_IDX = 0


# ================= \u8f85\u52a9\u51fd\u6570 =================
def get_text_hash(text):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


# ================= Global NLI Baseline \u7cfb\u7edf =================
class GlobalNLI_Baseline:
    def __init__(self, df):
        self.df = df
        print(f"🔄 Loading NLI model on {DEVICE}...")
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)
        self.corpus_texts = []
        self.doc_hash_list = []
        self.query_to_gt = {}  # {query_text: set(gt_hashes)}

    def build_corpus(self):
        """\u6784\u5efa\u5168\u91cf\u53bb\u91cd\u8bed\u6599\u5e93 (\u5305\u542b\u6240\u6709 conflict \u548c entail \u5217)"""
        print("🏗️ Building Deduplicated Corpus...")
        unique_docs = []
        seen_hashes = set()
        # \u9002\u914d RQ5_real.csv \u7684\u6587\u6863\u5217
        doc_cols = ['col2_conflict_en', 'col3_entail1_en', 'col4_entail2_en', 'col5_entail3_en']

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            for col in doc_cols:
                text = str(row.get(col, '')).strip()
                if not text or text.lower() == 'nan': continue
                h = get_text_hash(text)
                if h not in seen_hashes:
                    unique_docs.append(text)
                    self.doc_hash_list.append(h)
                    seen_hashes.add(h)

        self.corpus_texts = unique_docs
        print(f"✅ Total Corpus Size: {len(unique_docs)} unique docs")

    def prepare_queries(self):
        print("🔍 Mapping Queries to GT...")
        for _, row in self.df.iterrows():
            q_text = str(row.get('col1_query_en', '')).strip()
            gt_text = str(row.get('col2_conflict_en', '')).strip()
            if q_text and gt_text:
                h = get_text_hash(gt_text)
                if h in self.doc_hash_list:
                    if q_text not in self.query_to_gt:
                        self.query_to_gt[q_text] = set()
                    self.query_to_gt[q_text].add(h)
        print(f"✅ Valid Queries: {len(self.query_to_gt)}")

    def run_eval(self):
        query_texts = list(self.query_to_gt.keys())
        total_q = len(query_texts)
        chunk_size = total_q // NUM_CHUNKS
        all_results = []

        print(f"🚀 Running Global NLI Ranking (Contradiction Score Only)...")

        for i in range(NUM_CHUNKS):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q
            sub_queries = query_texts[start:end]

            metrics = {k: {'recall': [], 'ndcg': []} for k in METRIC_KS}

            for q_text in tqdm(sub_queries, desc=f"Exp {i + 1}/5", leave=False):
                gt_hashes = self.query_to_gt[q_text]

                # --- 1. \u6784\u9020\u5168\u5c40 Pair ---
                pairs = [[q_text, doc] for doc in self.corpus_texts]

                # --- 2. \u6279\u91cf\u63a8\u7406 ---
                logits = self.nli_model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
                probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

                # --- 3. \u63d0\u53d6\u77db\u76fe\u5f97\u5206 (Index 0) ---
                scores = probs[:, CON_IDX]

                # --- 4. \u6392\u5e8f ---
                ranked = []
                for idx, s in enumerate(scores):
                    ranked.append({'hash': self.doc_hash_list[idx], 'score': s})
                ranked.sort(key=lambda x: x['score'], reverse=True)

                # --- 5. \u8ba1\u7b97\u6307\u6807 ---
                for k in METRIC_KS:
                    top_k = ranked[:k]
                    hit = 1.0 if any(r['hash'] in gt_hashes for r in top_k) else 0.0
                    metrics[k]['recall'].append(hit)

                    dcg = sum([1.0 / np.log2(rank + 2) for rank, r in enumerate(top_k) if r['hash'] in gt_hashes])
                    idcg = sum([1.0 / np.log2(pos + 2) for pos in range(min(len(gt_hashes), k))])
                    metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            all_results.append({k: {m: np.mean(metrics[k][m]) for m in ['recall', 'ndcg']} for k in METRIC_KS})

        self._print_summary(all_results)

    def _print_summary(self, results):
        print("\n" + "=" * 100)
        print(f"📊 Global NLI Baseline Results (Scale: x100)")
        print("-" * 100)
        for m in ['recall', 'ndcg']:
            line = f"{m.upper():<10}"
            for k in METRIC_KS:
                data = [res[k][m] * 100 for res in results]
                line += f" | k={k}: {np.mean(data):.2f} (±{np.std(data):.2f})"
            print(line)
        print("=" * 100)


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    system = GlobalNLI_Baseline(df)
    system.build_corpus()
    system.prepare_queries()
    system.run_eval()