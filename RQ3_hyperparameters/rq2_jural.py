import os
# ================= 1. \u914d\u7f6e\u533a\u57df =================
# \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf
import time
import pandas as pd
import numpy as np
import ast
import torch
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm



CSV_PATH = 'pile_of_law_shuffle.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u6a21\u578b\u5b9a\u4e49
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'

# 🔥 RQ2 \u6838\u5fc3\u7f51\u683c\u53c2\u6570 (\u4e0e rq2_legal.py \u5b8c\u5168\u4e00\u81f4)
G_VALUES = [0, 1, 2, 3]  # \u68c0\u7d22\u8f6e\u6570
K_VALUES = [3, 5, 10, 20, 50]  # \u6bcf\u8f6e\u68c0\u7d22\u6df1\u5ea6
TARGET_KS = [3, 5, 10, 20, 50]  # \u6700\u7ec8\u8bc4\u4f30\u6307\u6807\u6df1\u5ea6
NUM_CHUNKS = 1  # 🚀 \u6307\u4ee4\u8981\u6c42：\u5207\u6210\u4e94\u4efd\u8fdb\u884c\u4e94\u6b21\u5b9e\u9a8c

# 🔥 \u6743\u91cd\u903b\u8f91 (\u5b8c\u5168\u9075\u5faa\u4f60\u5bf9 pile_of_law_shuffle \u7684\u53c2\u6570\u8bbe\u5b9a)
W_VEC = 0.8
W_LOGIC = 0.2
W_LOGIC_FULL = 0.8
W_LOGIC_ATOM = 0.2
W_DIRECT = 0.0
W_INDIRECT = 1.0



# ================= 2. \u8f85\u52a9\u5de5\u5177\u51fd\u6570 =================

def safe_eval(val):
    try:
        if pd.isna(val): return None
        s_val = str(val).strip()
        if not (s_val.startswith('[') or s_val.startswith('(')): return None
        parsed = ast.literal_eval(s_val)
        return parsed if isinstance(parsed, list) else None
    except:
        return None


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


# ================= 3. RQ2 \u5b9e\u9a8c\u7cfb\u7edf (Jural Suffix \u7248) =================

class RQ2_Pile_Jural_Researcher:
    def __init__(self, df):
        self.df = df
        self.doc_id_map = {}
        self.corpus_hashes = {}
        # \u52a0\u8f7d\u6a21\u578b\u903b\u8f91
        print(f"🔄 Loading Models on {DEVICE}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)
        self.query_cache = []

    def build_corpus(self):
        """\u9002\u914d Pile-of-Law \u5217\u540d\u5e76\u6784\u5efa\u8bed\u6599\u5e93"""
        print("🏗️ Building Corpus (Conflict + Entail)...")
        unique_docs, seen = [], {}
        doc_cols = [('conflict', 'conflict_atoms'), ('entail', 'entail_atoms')]

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
            for col_txt, col_atom in doc_cols:
                text = str(row.get(col_txt, '')).strip()
                if not text or text == 'nan': continue
                h = get_text_hash(text)
                if h not in seen:
                    doc_id = len(unique_docs);
                    unique_docs.append(text);
                    seen[h] = doc_id
                    # \u5b58\u50a8\u6587\u6863\u53ca\u5176\u539f\u5b50
                    self.doc_id_map[doc_id] = {
                        'text': text,
                        'atoms': flatten_list(safe_eval(row.get(col_atom)))
                    }
                    self.corpus_hashes[h] = doc_id

        self.corpus_texts = unique_docs
        print(f"🧮 Encoding Corpus ({len(unique_docs)} docs)...")
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True)

    def preprocess_queries(self):
        """\u9884\u5904\u7406\u67e5\u8be2\u53ca\u5176\u5bf9\u5e94\u7684 Ground Truth"""
        print("🛠️ Preprocessing Queries...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Caching"):
            sent0 = str(row.get('query', '')).strip()
            if not sent0: continue

            # Ground Truth \u5904\u7406
            gt_h = get_text_hash(str(row.get('conflict')))
            if gt_h not in self.corpus_hashes: continue

            # \u68c0\u7d22\u7528\u7684\u8d1f\u5411\u539f\u5b50
            raw_negs = safe_eval(row.get('negs_sourse_atoms')) or []
            if raw_negs and isinstance(raw_negs[0], str): raw_negs = [raw_negs]

            # \u9884\u8ba1\u7b97 Query \u5411\u91cf\u4ee5\u8282\u7701\u7f51\u683c\u641c\u7d22\u65f6\u95f4
            q_emb_origin = self.embedder.encode(sent0, convert_to_tensor=True)

            self.query_cache.append({
                'sent0': sent0,
                'q_emb_origin': q_emb_origin,
                'q_atoms': flatten_list(safe_eval(row.get('query_atoms'))),
                'raw_negs': raw_negs,
                'gt_ids': [self.corpus_hashes[gt_h]]
            })

    def _batch_calculate_logic(self, candidate_ids, q_data, g):
        full_pairs, atom_pairs, atom_meta = [], [], []
        sent0, q_negs = q_data['sent0'], flatten_list(q_data['raw_negs'])

        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            full_pairs.append([sent0, doc_data['text']])
            d_atoms = doc_data['atoms']
            curr_ap = []
            if q_negs:
                for da in d_atoms:
                    for qn in q_negs: curr_ap.append([da, qn])
            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap)})
            atom_pairs.extend(curr_ap)

        f_probs = torch.softmax(torch.tensor(self.nli_model.predict(full_pairs, show_progress_bar=False)),
                                dim=1).numpy()
        a_probs = torch.softmax(torch.tensor(self.nli_model.predict(atom_pairs, show_progress_bar=False)),
                                dim=1).numpy() if atom_pairs else []

        raw_scores = []
        for i in range(len(candidate_ids)):
            s_full, s_ind = f_probs[i][0], 0.0

            # 👇 2. \u6838\u5fc3\u4fee\u6539：g > 0 \u624d\u8ba1\u7b97\u95f4\u63a5\u5206，\u4e14\u53d6 Top-g
            if g > 0:
                m = atom_meta[i]
                if m['count'] > 0:
                    sl = a_probs[m['start']: m['start'] + m['count']]
                    inds = [val[1] for val in sl]  # \u53d6 entailment \u6982\u7387
                    if inds:
                        s_ind = np.mean(sorted(inds)[-g:])  # 👈 \u8fd9\u91cc\u6539\u6210 -g

            s_atom = (W_DIRECT * 0.0) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))

        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-9:
            return ((raw_scores - np.mean(raw_scores)) / np.std(raw_scores)).tolist()
        return (raw_scores - np.mean(raw_scores)).tolist()


    def run_rq2_analysis(self):
        """\u6267\u884c\u7f51\u683c\u641c\u7d22，\u6bcf\u7ec4\u8dd1 5 \u6b21\u5b9e\u9a8c\u5e76\u4fdd\u5b58\u7ed3\u679c"""
        total_q = len(self.query_cache)
        chunk_size = total_q // NUM_CHUNKS
        all_results = []

        # \u51c6\u5907\u8868\u5934
        columns_order = ['g', 'K_per_pass', 'Avg_Latency_ms']
        for k in TARGET_KS:
            columns_order.append(f'Recall@{k}')
            columns_order.append(f'NDCG@{k}')

        for g in G_VALUES:
            for k_per_pass in K_VALUES:
                print(f"\n🚀 Testing Configuration: g={g}, K={k_per_pass} (Across 5 Chunks)...")
                chunk_metrics = []
                chunk_latencies = []

                # 5 \u4efd\u5207\u5206\u5b9e\u9a8c\u903b\u8f91
                for i in range(NUM_CHUNKS):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i != NUM_CHUNKS - 1 else total_q
                    sub_queries = self.query_cache[start:end]

                    metrics = {tk: {'recall': [], 'ndcg': []} for tk in TARGET_KS}
                    latencies = []

                    for q_data in sub_queries:
                        start_time = time.time()

                        # 1. 3-Pass \u53ec\u56de\u903b\u8f91
                        all_variations = []

                        # \u904d\u5386\u6bcf\u4e00\u4e2a atom (N)
                        if q_data['raw_negs'] and g>0:
                            for atom_neg_list in q_data['raw_negs']:
                                # \u904d\u5386\u6bcf\u4e00\u4e2a atom \u7684\u524d g \u4e2a\u53cd\u4f8b
                                for d in range(min(g, len(atom_neg_list))):
                                    neg_atom = str(atom_neg_list[d]).strip()
                                    # \u62fc\u63a5: query + \u5355\u4e2a\u53cd\u4f8b
                                    all_variations.append(q_data['sent0'] + " " + neg_atom)

                        # \u9632\u5fa1\u6027\u903b\u8f91: \u5982\u679c\u6ca1\u6709\u53cd\u4f8b(\u6bd4\u5982 g=0 \u6216\u5217\u8868\u4e3a\u7a7a)，\u5219\u56de\u9000\u5230\u53ea\u7528\u539f\u59cb query
                        if not all_variations:
                            all_variations = [q_data['sent0']]

                        # 2. \u6279\u91cf\u7f16\u7801\u8fd9 N*g \u4e2a\u67e5\u8be2\u53d8\u4f53
                        # var_embs shape: (\u53d8\u4f53\u6570\u91cf, Embedding\u7ef4\u5ea6)
                        var_embs = self.embedder.encode(all_variations, convert_to_tensor=True, show_progress_bar=False)

                        # 3. \u8ba1\u7b97\u76f8\u4f3c\u5ea6\u77e9\u9635
                        # sim_matrix shape: (\u53d8\u4f53\u6570\u91cf, \u8bed\u6599\u5e93\u5927\u5c0f)
                        sim_matrix = util.cos_sim(var_embs, self.corpus_embeddings)

                        # 4. \u6267\u884c Max-Pooling：\u53d6\u6bcf\u4e2a\u6587\u6863\u5728\u6240\u6709\u53d8\u4f53\u4e0b\u7684\u6700\u5927\u5f97\u5206
                        # max_scores shape: (\u8bed\u6599\u5e93\u5927\u5c0f,)
                        # \u542b\u4e49: \u5bf9\u4e8e\u6bcf\u4e2a\u6587\u6863，\u5b83\u5339\u914d\u5230\u4e86\u54ea\u4e00\u4e2a\u53cd\u4f8b\u53d8\u4f53\u5f97\u5206\u6700\u9ad8，\u5c31\u7528\u90a3\u4e2a\u5206\u4ee3\u8868\u5b83
                        max_scores, _ = torch.max(sim_matrix, dim=0)

                        # 5. \u6700\u7ec8\u6839\u636e\u6700\u5927\u5f97\u5206，\u9009\u51fa Top-K \u4e2a\u6587\u6863
                        top_val, top_idx = torch.topk(max_scores, k=min(k_per_pass, len(self.corpus_texts)))
                        cand_list = top_idx.tolist()

                        # 2. \u63a8\u7406\u5206\u53d1 (\u53bb\u91cd\u540e\u4e00\u8f6e\u63a8\u7406)
                        s_vecs = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings[cand_list]).squeeze(
                            0).cpu().numpy()
                        s_logics = self._batch_calculate_logic(cand_list, q_data, g)
                        final_scores = (W_VEC * s_vecs) + (W_LOGIC * np.array(s_logics))

                        # \u8bb0\u5f55\u7269\u7406\u5ef6\u8fdf
                        latencies.append(time.time() - start_time)

                        # 3. \u7edf\u8ba1\u6307\u6807
                        ranked_idx = np.argsort(-final_scores)
                        ranked_ids = [cand_list[idx] for idx in ranked_idx]

                        for tk in TARGET_KS:
                            top_k = ranked_ids[:tk]
                            metrics[tk]['recall'].append(1.0 if any(gid in top_k for gid in q_data['gt_ids']) else 0.0)
                            dcg = sum([1.0 / np.log2(r + 2) for r, cid in enumerate(top_k) if cid in q_data['gt_ids']])
                            idcg = sum([1.0 / np.log2(n + 2) for n in range(min(len(q_data['gt_ids']), tk))])
                            metrics[tk]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

                    # \u5b58\u50a8\u672c\u6b21\u5b50\u5b9e\u9a8c\u5747\u503c
                    chunk_metrics.append(
                        {tk: {m: np.mean(metrics[tk][m]) for m in ['recall', 'ndcg']} for tk in TARGET_KS})
                    chunk_latencies.append(np.mean(latencies))

                # --- \u6c47\u603b\u672c\u914d\u7f6e\u7684 5 \u6b21\u5b9e\u9a8c\u5747\u503c\u548c\u6807\u51c6\u5dee ---
                res_row = {
                    'g': g,
                    'K_per_pass': k_per_pass,
                    'Avg_Latency_ms': round(np.mean(chunk_latencies) * 1000, 2)
                }
                for tk in TARGET_KS:
                    rs = [c[tk]['recall'] * 100 for c in chunk_metrics]
                    ns = [c[tk]['ndcg'] * 100 for c in chunk_metrics]
                    res_row[f'Recall@{tk}'] = f"{np.mean(rs):05.2f} (±{np.std(rs):05.2f})"
                    res_row[f'NDCG@{tk}'] = f"{np.mean(ns):05.2f} (±{np.std(ns):05.2f})"

                all_results.append(res_row)

                # \u5b9e\u65f6\u6253\u5370\u5f53\u524d\u8868\u683c
                current_df = pd.DataFrame(all_results)[columns_order]
                print(f"\n📊 Current Results Table (after g={g}, K={k_per_pass}):")
                print(current_df.to_string(index=False))
                print("-" * 120)

        # 🚀 \u6700\u7ec8\u4fdd\u5b58\u903b\u8f91：\u5bf9\u6807\u521a\u521a\u7684\u4ee3\u7801，\u6dfb\u52a0 jural \u540e\u7f00
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('RQ2_Results_Summary_jural.csv', index=False)
        print("\n" + "=" * 50)
        print(f"✅ RQ2 Analysis Completed. Final Data saved to 'RQ2_Results_Summary_jural.csv'.")
        print("=" * 50)
        return results_df


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        researcher = RQ2_Pile_Jural_Researcher(df)
        researcher.build_corpus()
        researcher.preprocess_queries()
        researcher.run_rq2_analysis()
    else:
        print(f"❌ File not found: {CSV_PATH}")