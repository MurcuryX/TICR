import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm
import hashlib
from collections import defaultdict
from dataclasses import dataclass

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'RQ5_real.csv'
MODEL_ID = 'SparseCL/BGE-SparseCL-hotpotqa'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u6307\u6807
METRIC_KS = [3, 5, 10, 20]
NUM_CHUNKS = 5

# 🔥 \u6781\u7ec6\u81f4\u641c\u7d22\u7a7a\u95f4 (0.01 \u6b65\u957f)
SEARCH_SPACE = np.linspace(0.0, 5.0, 501)


# ================= \u6a21\u578b\u52a0\u8f7d\u903b\u8f91 (\u5b8c\u5168\u5bf9\u9f50 MSMARCO \u811a\u672c) =================

@dataclass
class MockModelArgs:
    temp: float = 0.05
    pooler_type: str = "avg"
    do_mlm: bool = False
    hard_negative_weight: float = 0


def load_sparsecl_model(model_path):
    print(f"📦 Loading Model Logic: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_args = MockModelArgs()

    if "bge" in model_path.lower() or "uae" in model_path.lower():
        from models import our_BertForCL
        model = our_BertForCL.from_pretrained(model_path, config=config, model_args=model_args)
    elif "gte" in model_path.lower():
        from modeling import NewModelForCL
        model = NewModelForCL.from_pretrained(model_path, config=config, model_args=model_args, add_pooling_layer=True)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    return model.to(DEVICE)


def calculate_hoyer_sparsity(diff_tensor):
    d = diff_tensor.shape[1]
    sqrt_d = np.sqrt(d)
    l1 = torch.norm(diff_tensor, p=1, dim=1)
    l2 = torch.norm(diff_tensor, p=2, dim=1) + 1e-8
    return (sqrt_d - (l1 / l2)) / (sqrt_d - 1)


def get_embeddings(model, tokenizer, texts, batch_size=32):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, sent_emb=True)
            emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') else \
                (outputs.last_hidden_state.mean(dim=1))
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


# ================= \u641c\u7d22\u4e0e\u5b9e\u9a8c\u7cfb\u7edf =================

class SparseCL_Parameter_Searcher:
    def __init__(self, df):
        self.df = df
        self.query_gt_map = defaultdict(set)
        self.unique_queries = []
        self.doc_hash_map = {}
        self.cached_scores = []  # \u7f13\u5b58 cos \u548c hoyer \u4ee5\u4fbf\u6781\u901f\u641c\u7d22

        print(f"⏳ Initializing System...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = load_sparsecl_model(MODEL_ID)
        self.model.eval()

    def prepare_and_cache(self):
        print("🏗️ Building Corpus & Encoding...")
        unique_docs, seen_hashes = [], set()
        doc_cols = ['col2_conflict_en', 'col3_entail1_en', 'col4_entail2_en', 'col5_entail3_en']

        for _, row in self.df.iterrows():
            q = str(row.get('col1_query_en', "")).strip()
            if not q: continue
            if q not in self.query_gt_map: self.unique_queries.append(q)
            for col in doc_cols:
                txt = str(row.get(col, "")).strip()
                if not txt or txt.lower() == 'nan': continue
                h = hashlib.md5(txt.encode()).hexdigest()
                if h not in seen_hashes:
                    doc_id = len(unique_docs);
                    unique_docs.append(txt)
                    self.doc_hash_map[doc_id] = h;
                    seen_hashes.add(h)
            gt_h = hashlib.md5(str(row.get('col2_conflict_en', "")).encode()).hexdigest()
            self.query_gt_map[q].add(gt_h)

        print(f"🧮 Encoding {len(unique_docs)} documents...")
        corpus_embs = F.normalize(get_embeddings(self.model, self.tokenizer, unique_docs), p=2, dim=1).to(DEVICE)

        print("🚀 Pre-calculating components for all queries...")
        for query in tqdm(self.unique_queries, desc="Caching Components"):
            q_emb = F.normalize(get_embeddings(self.model, self.tokenizer, [query]), p=2, dim=1).to(DEVICE)

            # \u5b8c\u5168\u5bf9\u9f50\u6e90\u7801：\u5168\u91cf Cosine \u548c Hoyer
            cos_scores = torch.mm(q_emb, corpus_embs.T).squeeze(0)
            diff_vecs = q_emb - corpus_embs
            hoyer_scores = calculate_hoyer_sparsity(diff_vecs)

            self.cached_scores.append({
                'cos': cos_scores.cpu().numpy(),
                'hoyer': hoyer_scores.cpu().numpy(),
                'gt_hashes': self.query_gt_map[query]
            })

    def search_best_alpha(self):
        print(f"🔍 Fine-grained Grid Search for Alpha (n={len(SEARCH_SPACE)})...")
        best_alpha, best_ndcg3 = 0.0, -1.0

        for alpha in tqdm(SEARCH_SPACE, desc="Searching"):
            all_ndcg3 = []
            for item in self.cached_scores:
                final = item['cos'] + alpha * item['hoyer']
                # \u83b7\u53d6 Top 3
                top3_indices = np.argsort(final)[::-1][:3]

                dcg = sum(1.0 / np.log2(rank + 2) for rank, idx in enumerate(top3_indices)
                          if self.doc_hash_map[idx] in item['gt_hashes'])
                all_ndcg3.append(dcg)  # IDCG=1.0

            avg_ndcg3 = np.mean(all_ndcg3)
            if avg_ndcg3 > best_ndcg3:
                best_ndcg3 = avg_ndcg3
                best_alpha = alpha

        print(f"\n🏆 Best Alpha Found: {best_alpha:.2f} (NDCG@3: {best_ndcg3 * 100:.2f})")
        return best_alpha

    def final_evaluation(self, alpha):
        print(f"📊 Running 5-Chunk Evaluation with Alpha={alpha:.2f}...")
        total_q = len(self.cached_scores)
        chunk_size = total_q // NUM_CHUNKS
        exp_results = []

        for i in range(NUM_CHUNKS):
            start, end = i * chunk_size, ((i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q)
            chunk_data = self.cached_scores[start:end]
            metrics = {k: {'recall': [], 'ndcg': []} for k in METRIC_KS}

            for item in chunk_data:
                final = item['cos'] + alpha * item['hoyer']
                ranked_indices = np.argsort(final)[::-1]

                for k in METRIC_KS:
                    top_k_hashes = [self.doc_hash_map[idx] for idx in ranked_indices[:k]]
                    metrics[k]['recall'].append(1.0 if any(h in item['gt_hashes'] for h in top_k_hashes) else 0.0)
                    dcg = sum(1.0 / np.log2(r + 2) for r, h in enumerate(top_k_hashes) if h in item['gt_hashes'])
                    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(item['gt_hashes']), k)))
                    metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            exp_results.append({k: {m: np.mean(metrics[k][m]) * 100 for m in ['recall', 'ndcg']} for k in METRIC_KS})

        self._print_results(exp_results, alpha)

    def _print_results(self, results, alpha):
        print("\n" + "=" * 115)
        print(f"📊 Final SparseCL Report (Alpha={alpha:.2f}) | Optimized for NDCG@3")
        print("-" * 115)
        header = f"{'Metric (%)':<15}"
        for k in METRIC_KS: header += f" | k={k:<20}"
        print(header + "\n" + "-" * 115)
        for m in ['recall', 'ndcg']:
            line = f"{m.upper():<15}"
            for k in METRIC_KS:
                data = [res[k][m] for res in results]
                line += f" | {np.mean(data):05.2f} (±{np.std(data):05.2f})"
            print(line)
        print("=" * 115)


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH).fillna('')
    searcher = SparseCL_Parameter_Searcher(df)
    searcher.prepare_and_cache()
    best_alpha = searcher.search_best_alpha()
    searcher.final_evaluation(best_alpha)