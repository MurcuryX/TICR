import os
import json
import pandas as pd
import numpy as np
import hashlib
import string
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# ================= 1. \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'RQ5_real.csv'
JSON_PATH = 'final_dataset_dedup.json'

# \u8bc4\u6d4b\u53c2\u6570
KS = [3, 5, 10, 20]


# ================= 2. \u8f85\u52a9\u5de5\u5177 =================
def get_text_hash(text):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


def simple_tokenize(text):
    """\u6807\u51c6\u7684\u7b80\u5355\u5206\u8bcd\u5904\u7406：\u5c0f\u5199 + \u53bb\u6807\u70b9 + \u7a7a\u683c\u5206\u8bcd"""
    if not isinstance(text, str): return []
    text = text.lower()
    # \u53bb\u9664\u6807\u70b9\u7b26\u53f7
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


# ================= 3. BM25 \u8bc4\u4f30\u5668 =================
class BM25_Clean_Evaluator:
    def __init__(self, df, json_data):
        self.df = df
        self.json_data = json_data

        self.corpus_texts = []
        self.tokenized_corpus = []
        self.text_hash_to_id = {}
        self.query_cache = []
        self.bm25 = None

    def build_corpus(self):
        print("🏗️ Building BM25 Hybrid Corpus (JSON Clean + CSV Conflict)...")
        unique_docs = []

        # 1. \u4f18\u5148\u5165\u5e93 CSV \u4e2d\u7684 Conflict \u5217 (\u786e\u4fdd GT \u5b58\u5728)
        # \u8fd9\u4e0e\u4e4b\u524d\u7684 TICR \u903b\u8f91\u5b8c\u5168\u4e00\u81f4
        count_csv = 0
        for idx, row in self.df.iterrows():
            text = str(row.get('col2_conflict_en', '')).strip()
            if not text or text.lower() == 'nan': continue

            h = get_text_hash(text)
            if h not in self.text_hash_to_id:
                doc_id = len(unique_docs)
                unique_docs.append(text)
                self.text_hash_to_id[h] = doc_id
                count_csv += 1

        # 2. \u5165\u5e93 \u6e05\u6d17\u540e\u7684 JSON \u6570\u636e
        count_json = 0
        for item in self.json_data:
            text = str(item.get('content', '')).strip()
            if not text: continue

            h = get_text_hash(text)
            if h not in self.text_hash_to_id:
                doc_id = len(unique_docs)
                unique_docs.append(text)
                self.text_hash_to_id[h] = doc_id
                count_json += 1

        self.corpus_texts = unique_docs
        print(f"   -> Added {count_csv} GT docs from CSV.")
        print(f"   -> Added {count_json} docs from Clean JSON.")
        print(f"🧮 Tokenizing {len(unique_docs)} documents...")

        # BM25 \u7279\u6709\u7684\u9884\u5904\u7406：\u5bf9\u8bed\u6599\u5e93\u8fdb\u884c\u5206\u8bcd
        self.tokenized_corpus = [simple_tokenize(doc) for doc in tqdm(unique_docs, desc="Tokenizing Corpus")]

        print("🚀 Initializing BM25 Index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("✅ BM25 Ready.")

    def preprocess_queries(self):
        print("🛠️ Preprocessing Queries...")

        valid_count = 0
        for idx, row in self.df.iterrows():
            # Query \u6765\u81ea col1
            q_text = str(row.get('col1_query_en', '')).strip()
            if not q_text: continue

            # GT ID \u67e5\u627e (\u901a\u8fc7 col2 \u7684 hash)
            gt_text = str(row.get('col2_conflict_en', '')).strip()
            gt_hash = get_text_hash(gt_text)

            gt_ids = set()
            if gt_hash in self.text_hash_to_id:
                gt_ids.add(self.text_hash_to_id[gt_hash])

            # BM25 \u4e0d\u9700\u8981 Neg Atoms，\u53ea\u9700\u8981\u5206\u8bcd\u540e\u7684 Query
            q_tokens = simple_tokenize(q_text)

            self.query_cache.append({
                'q_tokens': q_tokens,
                'gt_ids': gt_ids
            })
            if gt_ids: valid_count += 1
        print(f"✅ Processed {len(self.query_cache)} queries (Valid GT found for {valid_count}).")

    def run_eval(self):
        print(f"\n🚀 Running BM25 Evaluation on {len(self.query_cache)} queries...")
        metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

        for q_data in tqdm(self.query_cache, desc="Evaluating"):
            # BM25 \u6253\u5206
            scores = self.bm25.get_scores(q_data['q_tokens'])

            # \u83b7\u53d6 Top-K (\u53d6\u6700\u5927\u7684 K)
            max_k = max(KS)
            # argsort \u8fd4\u56de\u4ece\u5c0f\u5230\u5927\u7684\u7d22\u5f15，[::-1] \u53cd\u8f6c\u4e3a\u4ece\u5927\u5230\u5c0f，\u53d6\u524d max_k
            top_indices = np.argsort(scores)[::-1][:max_k]

            # Metrics Calculation
            for k in KS:
                top_k_indices = top_indices[:k]

                # Recall: \u68c0\u67e5 Top-K \u91cc\u6709\u6ca1\u6709 GT ID
                hit = 1.0 if any(doc_id in q_data['gt_ids'] for doc_id in top_k_indices) else 0.0
                metrics[k]['recall'].append(hit)

                # NDCG
                dcg = 0.0
                for rank, doc_id in enumerate(top_k_indices):
                    if doc_id in q_data['gt_ids']:
                        dcg += 1.0 / np.log2(rank + 2)  # rank \u4ece 0 \u5f00\u59cb，\u6240\u4ee5 +2

                # IDCG (\u7406\u60f3\u60c5\u51b5\u4e0b\u7684 DCG)
                num_gt = len(q_data['gt_ids'])
                idcg = 0.0
                for i in range(min(num_gt, k)):
                    idcg += 1.0 / np.log2(i + 2)

                metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

        # Print Final Table
        print("\n" + "=" * 80)
        print(f"📊 BM25 Results (Clean JSON + CSV GT) | Scale: x100")
        print("-" * 80)
        header = f"{'Metric':<10}"
        for k in KS: header += f" | k={k:<10}"
        print(header + "\n" + "-" * 80)

        for m_name in ['recall', 'ndcg']:
            line = f"{m_name.upper():<10}"
            for k in KS:
                val = np.mean(metrics[k][m_name]) * 100
                line += f" | {val:05.2f}     "
            print(line)
        print("=" * 80)


if __name__ == "__main__":
    if os.path.exists(CSV_PATH) and os.path.exists(JSON_PATH):
        df = pd.read_csv(CSV_PATH)
        df.fillna('', inplace=True)

        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        evaluator = BM25_Clean_Evaluator(df, json_data)
        evaluator.build_corpus()
        evaluator.preprocess_queries()
        evaluator.run_eval()
    else:
        print(f"❌ File missing: Please ensure {CSV_PATH} and {JSON_PATH} exist.")