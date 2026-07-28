import os

# 1. \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import hashlib

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'RQ5_real.csv'  # \u4fee\u6539\u4e3a RQ5 \u771f\u5b9e\u6570\u636e

# \u6a21\u578b
MODEL_NAME = 'naver/splade-cocondenser-ensembledistil'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
KS = [3, 5, 10, 20]
BATCH_SIZE = 16
NUM_CHUNKS = 5  # \u4fdd\u6301\u539f\u6709\u903b\u8f91：\u5207\u6210\u4e94\u4efd\u8fdb\u884c\u4e94\u6b21\u5b9e\u9a8c


# ================= \u8f85\u52a9\u51fd\u6570 =================
def force_str(val):
    if pd.isna(val) or str(val).lower() == 'nan': return ""
    return str(val).strip()


def get_text_hash(text):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


# ================= SPLADE \u6838\u5fc3\u7f16\u7801\u5668 (\u4fdd\u6301\u903b\u8f91\u4e0d\u53d8) =================
class SpladeEncoder:
    def __init__(self, model_name, device):
        print(f"🔌 Loading SPLADE model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()

    def encode(self, texts, batch_size=32):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                # log(1 + relu) max pooling \u903b\u8f91\u4fdd\u6301\u4e0d\u53d8
                values, _ = torch.max(
                    torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1),
                    dim=1
                )
                norms = torch.norm(values, p=2, dim=1, keepdim=True)
                values = values / norms.clamp(min=1e-9)
                all_vecs.append(values.cpu())
        return torch.cat(all_vecs, dim=0)


# ================= SPLADE \u7cfb\u7edf (5-Part \u5b9e\u9a8c\u7248) =================
class Splade_Legal_Baseline:
    def __init__(self, df):
        self.df = df
        self.encoder = SpladeEncoder(MODEL_NAME, DEVICE)
        self.corpus_texts = []
        self.corpus_hashes = {}
        self.corpus_vecs = None
        self.query_to_gt = {}  # {query_text: set(gt_hashes)}

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

        for _, row in self.df.iterrows():
            for col in doc_cols:
                text = force_str(row.get(col))
                if not text: continue
                h = get_text_hash(text)
                if h not in seen_hashes:
                    doc_id = len(unique_docs)
                    unique_docs.append(text)
                    seen_hashes[h] = doc_id
                    self.corpus_hashes[h] = doc_id

        self.corpus_texts = unique_docs
        print(f"✅ Unique Docs in Corpus: {len(unique_docs)}")
        # \u6279\u91cf\u7f16\u7801\u8bed\u6599\u5e93
        self.corpus_vecs = self.encoder.encode(unique_docs, batch_size=BATCH_SIZE)

    def prepare_queries(self):
        """\u9884\u5904\u7406\u67e5\u8be2\u4e0e GT \u5173\u7cfb (\u9002\u914d RQ5)"""
        print("🔍 Mapping Queries to GT (RQ5)...")
        q_col = 'col1_query_en'
        gt_cols = ['col2_conflict_en']  # RQ5 \u53ea\u6709 col2 \u662f GT

        valid_count = 0
        for _, row in self.df.iterrows():
            query_text = force_str(row.get(q_col))
            if not query_text: continue

            gt_hashes = set()
            for col in gt_cols:
                t = force_str(row.get(col))
                if t:
                    h = get_text_hash(t)
                    if h in self.corpus_hashes:
                        gt_hashes.add(self.corpus_hashes[h])

            if gt_hashes:
                if query_text not in self.query_to_gt:
                    self.query_to_gt[query_text] = set()
                self.query_to_gt[query_text].update(gt_hashes)
                valid_count += 1

        print(f"✅ Processed Queries with Valid GT: {valid_count}")

    def run_split_experiments(self):
        """\u6267\u884c 5 \u6b21\u5207\u5206\u5b9e\u9a8c"""
        query_texts = list(self.query_to_gt.keys())
        total_q = len(query_texts)
        chunk_size = total_q // NUM_CHUNKS

        all_exp_metrics = []

        print(f"🚀 Splitting {total_q} queries into {NUM_CHUNKS} chunks for experiments...")

        for i in range(NUM_CHUNKS):
            # \u5904\u7406\u53ef\u80fd\u7684\u8fb9\u754c\u60c5\u51b5
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q

            if start >= total_q: break  # \u9632\u6b62\u6570\u636e\u8fc7\u5c11\u5bfc\u81f4\u7684\u7d22\u5f15\u8d8a\u754c

            sub_queries = query_texts[start:end]
            if not sub_queries: continue

            sub_gt_list = [self.query_to_gt[q] for q in sub_queries]

            print(f"   [Experiment {i + 1}/5] Size: {len(sub_queries)}")

            # \u5b50\u96c6\u7f16\u7801
            query_vecs = self.encoder.encode(sub_queries, batch_size=BATCH_SIZE)

            # \u5f97\u5206\u8ba1\u7b97 (\u4f59\u5f26\u76f8\u4f3c\u5ea6 = \u77e9\u9635\u70b9\u4e58)
            # \u6ce8\u610f：\u8fd9\u91cc\u5047\u8bbe corpus_vecs \u5df2\u7ecf\u5f88\u5927，\u5982\u679c\u663e\u5b58\u4e0d\u591f\u53ef\u80fd\u9700\u8981\u5206\u6279\u8ba1\u7b97，
            # \u4f46 RQ5 \u6570\u636e\u96c6\u8f83\u5c0f，\u76f4\u63a5\u77e9\u9635\u4e58\u6cd5\u901a\u5e38\u6ca1\u95ee\u9898。
            scores = torch.matmul(query_vecs.to(DEVICE), self.corpus_vecs.to(DEVICE).t()).cpu().numpy()

            # \u7edf\u8ba1\u672c\u8f6e\u6307\u6807
            exp_results = {k: {"recall": [], "ndcg": []} for k in KS}
            for j, doc_scores in enumerate(scores):
                gt_ids = sub_gt_list[j]
                # \u6392\u5e8f\u83b7\u53d6 Top-K
                top_indices = np.argsort(-doc_scores)[:max(KS)]

                for k in KS:
                    top_k = top_indices[:k]
                    # Recall@k
                    exp_results[k]["recall"].append(1.0 if any(idx in gt_ids for idx in top_k) else 0.0)
                    # NDCG@k
                    dcg = sum([1.0 / np.log2(rank + 2) for rank, idx in enumerate(top_k) if idx in gt_ids])
                    idcg = sum([1.0 / np.log2(n + 2) for n in range(min(len(gt_ids), k))])
                    exp_results[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)

            # \u5b58\u50a8\u672c\u8f6e\u5747\u503c
            all_exp_metrics.append({
                k: {
                    "recall": np.mean(exp_results[k]["recall"]),
                    "ndcg": np.mean(exp_results[k]["ndcg"])
                } for k in KS
            })

        self._print_final_summary(all_exp_metrics)

    def _print_final_summary(self, results):
        """\u7edf\u8ba1\u5747\u503c\u4e0e\u6807\u51c6\u5dee，\u4fdd\u7559\u4e24\u4f4d\u5c0f\u6570\u5e76\u4e58 100"""
        if not results:
            print("No results to display.")
            return

        print("\n" + "=" * 110)
        print(f"📊 SPLADE Final Summary (Averaged across {len(results)} Chunks) | Scale: x100")
        print("-" * 110)
        header = f"{'Metric (%)':<15}"
        for k in KS: header += f" | k={k:<20}"
        print(header + "\n" + "-" * 110)

        r_line, n_line = f"{'Recall@k':<15}", f"{'NDCG@k':<15}"
        for k in KS:
            # \u653e\u5927 100 \u500d
            rs = [res[k]["recall"] * 100 for res in results]
            ns = [res[k]["ndcg"] * 100 for res in results]

            avg_r, std_r = np.mean(rs), np.std(rs)
            avg_n, std_n = np.mean(ns), np.std(ns)

            r_line += f" | {avg_r:05.2f} (±{std_r:05.2f})"
            n_line += f" | {avg_n:05.2f} (±{std_n:05.2f})"

        print(r_line + "\n" + n_line + "\n" + "=" * 110)


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # \u7b80\u5355\u586b\u5145\u9632\u6b62 NaN \u62a5\u9519
        df.fillna('', inplace=True)
        system = Splade_Legal_Baseline(df)
        system.build_corpus()
        system.prepare_queries()
        system.run_split_experiments()
    else:
        print(f"❌ Dataset not found: {CSV_PATH}")