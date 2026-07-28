import os# ================= 1. \u914d\u7f6e\u533a\u57df =================
# \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf（\u5982\u9700\u8981）
import time
import pandas as pd
import numpy as np
import ast
import torch
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm



CSV_PATH = 'Legal_Contra.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u6a21\u578b\u5b9a\u4e49 [cite: 588, 590]
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'

# 🔥 RQ2 \u6838\u5fc3\u7f51\u683c\u53c2\u6570
G_VALUES = [0, 1, 2, 3]  # \u53cd\u4f8b\u751f\u6210/\u68c0\u7d22\u8f6e\u6570
K_VALUES = [3, 5, 10, 20, 50]  # \u6bcf\u8f6e\u68c0\u7d22\u6df1\u5ea6
TARGET_KS = [3, 5, 10, 20, 50]  # \u8bc4\u4f30\u6307\u6807\u6df1\u5ea6

# 🔥 \u6743\u91cd\u903b\u8f91 (\u5b8c\u5168\u9075\u5faa\u4f60\u7684\u8981\u6c42)
W_VEC = 0.8
W_LOGIC = 0.2
W_LOGIC_FULL = 0.5
W_LOGIC_ATOM = 0.5
W_DIRECT = 0.0  # \u4e0d\u8003\u8651\u539f\u53e5\u539f\u5b50\u7684\u77db\u76fe\u5206
W_INDIRECT = 1.0  # \u539f\u5b50\u5f97\u5206\u5168\u770b\u53cd\u4e49\u539f\u5b50\u7684\u8574\u542b\u5206

print("RQ2\u6743\u91cd\u5982\u4e0b：")
print(f"W_VEC: {W_VEC}")
print(f"W_LOGIC: {W_LOGIC}")
print(f"W_LOGIC_FULL: {W_LOGIC_FULL}")
print(f"W_LOGIC_ATOM: {W_LOGIC_ATOM}")


# ================= 2. \u8f85\u52a9\u5de5\u5177\u51fd\u6570 =================
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
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        elif isinstance(item, str) and item.strip():
            flat.append(item)
    return flat


# ================= 3. RQ2 \u5b9e\u9a8c\u7cfb\u7edf =================
class RQ2_TICR_Researcher:
    def __init__(self, df):
        self.df = df
        self.doc_id_map = {}
        self.corpus_hashes = {}
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)
        self.query_cache = []

    def build_corpus(self):
        """\u6784\u5efa\u8bed\u6599\u5e93\u5e76\u8fdb\u884c\u5411\u91cf\u9884\u5904\u7406 [cite: 67, 230]"""
        print("🏗️ Building Corpus...")
        unique_docs, seen = [], {}
        doc_cols = [
            ('Conflict1 (Contradiction)', 'Conflict1 (Contradiction)_atoms'),
            ('Conflict2 (Contradiction)', 'Conflict2 (Contradiction)_atoms'),
            ('Paraphrase_Structure_1 (Entailment)', 'Paraphrase_Structure_1 (Entailment)_atoms')
        ]
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Indexing"):
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
        """\u9884\u5904\u7406\u67e5\u8be2、GT\u53ca\u53cd\u4e49\u539f\u5b50 [cite: 254, 347]"""
        print("🛠️ Preprocessing Queries...")
        for _, row in self.df.iterrows():
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

            # \u7f13\u5b58\u539f\u53e5 Embedding \u4f9b\u540e\u671f $S_{vec}$ \u4f7f\u7528
            q_emb_origin = self.embedder.encode(sent0, convert_to_tensor=True)

            self.query_cache.append({
                'sent0': sent0, 'q_emb_origin': q_emb_origin,
                'q_atoms': flatten_list(safe_eval(row.get('Original_Text_atoms'))),
                'raw_negs': raw_negs, 'gt_ids': gt_ids
            })

    def _batch_calculate_logic(self, candidate_ids, q_data, g):
        """\u7cbe\u6392\u9636\u6bb5\u7684 NLI \u903b\u8f91\u5206\u8ba1\u7b97 [cite: 448, 453]"""
        full_pairs, atom_pairs, atom_meta = [], [], []
        sent0, q_negs = q_data['sent0'], flatten_list(q_data['raw_negs'])

        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            full_pairs.append([sent0, doc_data['text']])  # \u5168\u6587\u77db\u76fe\u68c0\u6d4b
            d_atoms = doc_data['atoms']
            curr_ap = []
            if q_negs:
                for da in d_atoms:
                    for qn in q_negs: curr_ap.append([da, qn])  # \u539f\u5b50\u7ea7\u8574\u542b\u68c0\u6d4b
            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap)})
            atom_pairs.extend(curr_ap)

        # NLI \u63a8\u7406 (0: Contradiction, 1: Entailment, 2: Neutral)
        f_probs = torch.softmax(torch.tensor(self.nli_model.predict(full_pairs, show_progress_bar=False)),
                                dim=1).numpy()
        a_probs = torch.softmax(torch.tensor(self.nli_model.predict(atom_pairs, show_progress_bar=False)),
                                dim=1).numpy() if atom_pairs else []

        raw_scores = []
        for i in range(len(candidate_ids)):
            s_full, s_ind = f_probs[i][0], 0.0

            # 🔥 \u6838\u5fc3\u4fee\u6b63：\u53ea\u6709\u5f53 g > 0 \u65f6\u624d\u8ba1\u7b97\u95f4\u63a5\u903b\u8f91\u5206
            if g > 0:
                m = atom_meta[i]
                if m['count'] > 0:
                    sl = a_probs[m['start']: m['start'] + m['count']]
                    inds = [val[1] for val in sl]
                    if inds:
                        # Top-g \u5747\u503c
                        s_ind = np.mean(sorted(inds)[-g:])

            # \u6839\u636e W_DIRECT=0, W_INDIRECT=1 \u8ba1\u7b97\u539f\u5b50\u5206
            s_atom = (W_DIRECT * 0.0) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))

        # Z-Score \u6807\u51c6\u5316\u4ee5\u5e73\u8861\u91cf\u7ea7
        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-9:
            return ((raw_scores - np.mean(raw_scores)) / np.std(raw_scores)).tolist()
        return (raw_scores - np.mean(raw_scores)).tolist()

    def run_rq2_grid_search(self):
        """\u6838\u5fc3 RQ2 \u626b\u63cf\u4efb\u52a1：\u6bcf\u7ec4\u53c2\u6570\u8dd1\u5b8c\u7acb\u5373\u6253\u5370\u8868\u683c"""
        all_results = []
        print(f"🚀 Starting RQ2 Grid Search: {len(G_VALUES)} Gs x {len(K_VALUES)} Ks")

        # \u4e3a\u4e86\u6253\u5370\u7f8e\u89c2，\u5b9a\u4e49\u4e00\u4e0b\u5217\u7684\u987a\u5e8f
        columns_order = ['g', 'K_per_pass', 'Avg_Latency_ms',
                         'Recall@3', 'NDCG@3', 'Recall@5', 'NDCG@5',
                         'Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20']

        for g in G_VALUES:
            for k_per_pass in K_VALUES:
                # print(f"\n[Test] g={g}, K_per_pass={k_per_pass}") # \u6539\u4e3a\u8868\u683c\u5c55\u793a，\u8fd9\u91cc\u53ef\u4ee5\u7cbe\u7b80
                metrics = {tk: {'recall': [], 'ndcg': []} for tk in TARGET_KS}
                latencies = []

                for q_data in tqdm(self.query_cache, desc=f"Testing g={g},K={k_per_pass}", leave=False):
                    start_time = time.time()

                    # --- Step 1: \u4e09\u8def\u62fc\u63a5\u68c0\u7d22 (\u6839\u636e g \u7684\u8bbe\u5b9a) ---
                    all_variations = []

                    # \u904d\u5386\u6bcf\u4e00\u4e2a atom (N)
                    if g > 0 and q_data['raw_negs']:
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
                    cand_ids = top_idx.tolist()

                    # --- Step 2: RAG \u7cbe\u6392\u6253\u5206 ---
                    s_vecs = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings[cand_ids]).squeeze(
                        0).cpu().numpy()
                    s_logics = self._batch_calculate_logic(cand_ids, q_data, g)

                    final_rank = []
                    for idx, cid in enumerate(cand_ids):
                        score = (W_VEC * s_vecs[idx]) + (W_LOGIC * s_logics[idx])
                        final_rank.append({'is_gt': cid in q_data['gt_ids'], 'score': score})

                    final_rank.sort(key=lambda x: x['score'], reverse=True)
                    latencies.append(time.time() - start_time)

                    # --- Step 3: \u6307\u6807\u8bb0\u5f55 ---
                    for tk in TARGET_KS:
                        top_k_list = final_rank[:tk]
                        metrics[tk]['recall'].append(1.0 if any(r['is_gt'] for r in top_k_list) else 0.0)
                        dcg = sum(1.0 / np.log2(rank + 2) for rank, r in enumerate(top_k_list) if r['is_gt'])
                        idcg = sum(1.0 / np.log2(n + 2) for n in range(min(len(q_data['gt_ids']), tk)))
                        metrics[tk]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

                # --- \u6838\u5fc3\u6539\u52a8：\u8ba1\u7b97\u5e76\u7acb\u5373\u6253\u5370\u8868\u683c ---
                res_row = {
                    'g': g,
                    'K_per_pass': k_per_pass,
                    'Avg_Latency_ms': round(np.mean(latencies) * 1000, 2)
                }
                for tk in TARGET_KS:
                    res_row[f'Recall@{tk}'] = round(np.mean(metrics[tk]['recall']) * 100, 2)
                    res_row[f'NDCG@{tk}'] = round(np.mean(metrics[tk]['ndcg']) * 100, 2)

                all_results.append(res_row)

                # \u8f6c\u6362\u6210 DataFrame \u5e76\u6253\u5370
                current_df = pd.DataFrame(all_results)[columns_order]

                # \u6e05\u9664\u63a7\u5236\u53f0\u4e4b\u524d\u7684\u8f93\u51fa（\u53ef\u9009，\u5982\u679c\u4e0d\u6e05\u9664\u5219\u4f1a\u4e00\u76f4\u5411\u4e0b\u6eda\u52a8\u6253\u5370）
                # os.system('cls' if os.name == 'nt' else 'clear')

                print(f"\n📊 Current Results Table (after g={g}, K={k_per_pass}):")
                print(current_df.to_string(index=False))
                print("-" * 120)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv('RQ2_Results_Summary_legal.csv', index=False)
        print("\n✅ RQ2 Analysis Completed. Final Data saved to 'RQ2_Results_Summary_legal.csv'.")
        return results_df


# ================= 4. \u4e3b\u7a0b\u5e8f\u5165\u53e3 =================
if __name__ == "__main__":
    # \u786e\u4fdd\u6587\u4ef6\u5b58\u5728
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found.")
    else:
        df_raw = pd.read_csv(CSV_PATH)
        researcher = RQ2_TICR_Researcher(df_raw)
        researcher.build_corpus()
        researcher.preprocess_queries()
        researcher.run_rq2_grid_search()