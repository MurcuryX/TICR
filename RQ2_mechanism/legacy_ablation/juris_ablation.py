import os

# 1. \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf

import pandas as pd
import numpy as np
import ast
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
import hashlib

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'pile_of_law_shuffle.csv'  # \u786e\u4fdd\u4f7f\u7528\u5904\u7406\u8fc7\u7684\u6570\u636e\u96c6

# \u6a21\u578b
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570
TOP_K_PER_PASS = 20  # \u6bcf\u6b21\u68c0\u7d22\u53ec\u56de 20 \u6761，\u4e09\u6b21\u540e\u53bb\u91cd
KS = [3, 5, 10, 20]
g = 3   # query\u6bcf\u4e2aatom\u751f\u6210\u7684\u53cd\u4f8b\u6570\u91cf
NUM_FOLDS = 5  # \u5206\u62105\u4efd\u5b9e\u9a8c

# 🔥 \u6743\u91cd\u914d\u7f6e
W_VEC = 0.88
W_LOGIC = 0.12
W_LOGIC_FULL = 1.0
W_LOGIC_ATOM = 0.0
W_DIRECT = 0.0
W_INDIRECT = 1.0



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


# ================= \u5b9e\u9a8c\u7c7b =================
class PileOfLawExperiment:
    def __init__(self, df):
        self.df = df
        self.doc_id_map = {}
        self.corpus_hashes = {}
        print(f"🔄 Loading models on {DEVICE}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)
        self.query_cache = []

    def build_corpus(self):
        print("🏗️ Building Corpus from 'entail' and 'conflict' columns...")
        unique_docs, seen = [], {}
        doc_cols = [('entail', 'entail_atoms'), ('conflict', 'conflict_atoms')]
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
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True,
                                                      batch_size=64)
        print(f"✅ Corpus built: {len(unique_docs)} docs")

    def preprocess_queries(self):
        print("🛠️ Preprocessing Queries (Pile of Law Mapping)...")
        for idx, row in self.df.iterrows():
            q_text = str(row.get('query', '')).strip()
            if not q_text: continue
            gt_ids = set()
            conflict_texts = force_list_str(row.get('conflict'))
            for t in conflict_texts:
                h = get_text_hash(t)
                if h in self.corpus_hashes:
                    gt_ids.add(self.corpus_hashes[h])
            if not gt_ids: continue

            raw_negs = safe_eval(row.get('negs_sourse_atoms')) or []
            if raw_negs and isinstance(raw_negs[0], str): raw_negs = [raw_negs]
            all_neg_atoms = flatten_list(raw_negs)

            instruction = "Represent this sentence for searching relevant passages: "
            q_emb_origin = self.embedder.encode(instruction + q_text, convert_to_tensor=True)

            self.query_cache.append({
                'sent0': q_text,
                'q_emb_origin': q_emb_origin,
                'q_atoms': flatten_list(safe_eval(row.get('query_atoms'))),
                'raw_negs': raw_negs,
                'all_neg_atoms': all_neg_atoms,
                'gt_ids': gt_ids
            })
        print(f"✅ Cached {len(self.query_cache)} queries.")

    def run_5fold_eval(self):
        print(f"\n🚀 Running {NUM_FOLDS}-Fold Evaluation (3M-Pass Max-Pooling Retrieval)")
        total_q = len(self.query_cache)
        fold_size = total_q // NUM_FOLDS

        # \u8bb0\u5f55\u6bcf\u4e00\u4efd\u5b9e\u9a8c\u7684\u5e73\u5747\u5206
        fold_summaries = {k: {'recall': [], 'ndcg': []} for k in KS}

        for f in range(NUM_FOLDS):
            start = f * fold_size
            end = (f + 1) * fold_size if f != NUM_FOLDS - 1 else total_q
            current_queries = self.query_cache[start:end]

            print(f"Experiment Part {f + 1}/{NUM_FOLDS} (n={len(current_queries)})")

            # \u7528\u4e8e\u5b58\u50a8\u672c Fold \u5185\u6240\u6709 Query \u7684\u6307\u6807
            current_fold_metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

            for q_data in tqdm(current_queries, desc=f"Fold {f + 1}", leave=False):
                sent0 = q_data['sent0']

                # ==========================================================
                # 🔥 \u6539\u52a8\u70b9：3M-Pass Max-Pooling Retrieval
                # ==========================================================
                cand_queries = []
                # \u5c55\u5e73 raw_negs \u4e2d\u7684\u6240\u6709\u539f\u5b50\u53d8\u4f53
                # q_data['raw_negs'] \u53ef\u80fd\u662f nested list，\u9700\u5c55\u5e73\u5904\u7406
                # q_data['all_neg_atoms'] \u5df2\u7ecf\u5728 preprocess_queries \u91cc\u5c55\u5e73\u8fc7\u4e86，\u76f4\u63a5\u7528\u66f4\u65b9\u4fbf
                all_atoms = q_data.get('all_neg_atoms', [])

                for atom in all_atoms:
                    atom_str = str(atom).strip()
                    if atom_str:
                        # \u6784\u9020\u63a2\u6d4b\u951a\u70b9：\u539f\u53e5 + \u5355\u4e2a\u9006\u5411\u539f\u5b50
                        cand_queries.append(f"{sent0} {atom_str}")

                # \u9632\u6296\u903b\u8f91：\u5982\u679c\u8be5 Query \u6ca1\u6709\u6709\u6548\u7684\u8d1f\u5411\u539f\u5b50，\u56de\u9000\u5230\u539f\u53e5\u68c0\u7d22
                if not cand_queries:
                    cand_queries = [sent0]

                # 1. \u6279\u91cf\u7f16\u7801\u6240\u6709\u63a2\u6d4b\u951a\u70b9 [N_atoms, Dim]
                # instruction \u9700\u4e0e build_corpus / preprocess \u4fdd\u6301\u4e00\u81f4
                instruction = "Represent this sentence for searching relevant passages: "
                q_embs_3m = self.embedder.encode([instruction + q for q in cand_queries],
                                                 convert_to_tensor=True)

                # 2. \u8ba1\u7b97\u76f8\u4f3c\u5ea6\u77e9\u9635 [N_atoms, N_corpus]
                sim_matrix = util.cos_sim(q_embs_3m, self.corpus_embeddings)

                # 3. Max-Pooling: \u5bf9\u6bcf\u4e2a\u6587\u6863，\u53d6\u5b83\u4e0e\u6240\u6709\u63a2\u6d4b\u70b9\u7684\u6700\u5927\u76f8\u4f3c\u5ea6\u4f5c\u4e3a\u5f97\u5206
                # values: [N_corpus], indices: [N_corpus] (indices \u5728\u8fd9\u91cc\u65e0\u610f\u4e49，\u53ea\u53d6 values)
                max_sims, _ = torch.max(sim_matrix, dim=0)

                # 4. \u5168\u5c40 Top-K \u53ec\u56de (\u76f4\u63a5\u53d6 Max-Pooling \u540e\u5f97\u5206\u6700\u9ad8\u7684 K \u4e2a\u6587\u6863)
                # \u6ce8\u610f：max_sims \u662f tensor，topk \u8fd4\u56de\u7684 indices \u5373\u4e3a doc_id
                top_k_indices = torch.topk(max_sims, k=min(TOP_K_PER_PASS, len(self.corpus_texts))).indices.tolist()
                cand_ids = top_k_indices

                # ==========================================================
                # \u7b2c\u4e8c\u9636\u6bb5：\u903b\u8f91\u91cd\u6392 (\u903b\u8f91\u4fdd\u6301\u4e0d\u53d8)
                # ==========================================================

                # \u8ba1\u7b97\u539f\u53e5 q \u4e0e\u5019\u9009\u6587\u6863\u7684\u8bed\u4e49\u57fa\u51c6\u5206 (Semantic Baseline)
                cand_embs = self.corpus_embeddings[cand_ids]
                sims_origin = util.cos_sim(q_data['q_emb_origin'], cand_embs).squeeze(0)

                # \u6279\u91cf\u8ba1\u7b97\u903b\u8f91\u5206
                # \u6ce8\u610f：_batch_calculate_logic \u9700\u8981\u4f20\u5165\u6b63\u786e\u7684\u53c2\u6570
                # \u8fd9\u91cc q_data \u91cc\u7684 keys \u9700\u4e0e preprocess_queries \u91cc\u7684\u4fdd\u6301\u4e00\u81f4
                # \u6839\u636e\u4f60\u63d0\u4f9b\u7684\u5b8c\u6574\u4ee3\u7801，preprocess_queries \u91cc\u5b58\u7684\u662f 'q_atoms' \u548c 'all_neg_atoms'
                logic_input = {
                    'atoms': q_data['q_atoms'],
                    'negs': q_data['all_neg_atoms']
                }

                # \u6ce8\u610f：\u539f\u4ee3\u7801 _batch_calculate_logic \u5b9a\u4e49\u4e3a (candidate_ids, q_data, sent0)
                # \u4f46\u4f60\u8c03\u7528\u7684\u5730\u65b9\u4f20\u53c2\u6709\u70b9\u6df7\u4e71，\u8fd9\u91cc\u6211\u4eec\u4e25\u683c\u6309\u7167\u4f60\u7684 _batch_calculate_logic \u7b7e\u540d\u6765\u4f20
                # \u539f\u51fd\u6570\u7b7e\u540d：def _batch_calculate_logic(self, candidate_ids, q_data, sent0):
                # \u539f\u51fd\u6570\u5185\u90e8\u4f7f\u7528 q_data['atoms'] \u548c q_data['negs']
                # \u6240\u4ee5\u6211\u4eec\u6784\u9020\u4e00\u4e2a\u4e34\u65f6\u7684 logic_data \u5b57\u5178\u4f20\u8fdb\u53bb
                logic_data_for_func = {'atoms': q_data['q_atoms'], 'negs': q_data['all_neg_atoms']}
                s_logics = self._batch_calculate_logic(cand_ids, logic_data_for_func, sent0)

                ranked_res = []
                for idx, doc_id in enumerate(cand_ids):
                    s_vec = sims_origin[idx].item()
                    s_logic = s_logics[idx]
                    final_score = (W_VEC * s_vec) + (W_LOGIC * s_logic)
                    ranked_res.append({'is_gt': doc_id in q_data['gt_ids'], 'score': final_score})

                ranked_res.sort(key=lambda x: x['score'], reverse=True)

                # \u8ba1\u7b97\u6307\u6807
                for k in KS:
                    top_k = ranked_res[:k]
                    # Recall
                    r = 1.0 if any(r['is_gt'] for r in top_k) else 0.0
                    current_fold_metrics[k]['recall'].append(r)

                    # NDCG
                    dcg = sum(1.0 / np.log2(rk + 2) for rk, r in enumerate(top_k) if r['is_gt'])
                    # IDCG: min(len(gt), k) \u786e\u4fdd\u53cd\u4f8b\u6570\u5c11\u4e8e K \u65f6\u5206\u6bcd\u4e0d\u8fc7\u5927
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(q_data['gt_ids']), k)))
                    current_fold_metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            # \u6c47\u603b\u672c Fold \u7ed3\u679c
            for k in KS:
                fold_summaries[k]['recall'].append(np.mean(current_fold_metrics[k]['recall']))
                fold_summaries[k]['ndcg'].append(np.mean(current_fold_metrics[k]['ndcg']))

            # \u6253\u5370\u672c Fold \u7684 R@3 (\u4f5c\u4e3a\u8fdb\u5ea6\u53c2\u8003)
            print(f"✅ Fold {f + 1} Done | R@3: {fold_summaries[3]['recall'][-1] * 100:.2f}")

        self._print_fold_summary(fold_summaries)

    def _batch_calculate_logic(self, candidate_ids, q_data, sent0):
        full_pairs, atom_pairs, atom_meta = [], [], []
        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            full_pairs.append([sent0, doc_data['text']])
            d_atoms, q_atoms, q_negs = doc_data['atoms'], q_data['atoms'], q_data['negs']
            curr_ap, curr_ty = [], []
            if q_atoms:
                for da in d_atoms:
                    for qa in q_atoms: curr_ap.append([da, qa]); curr_ty.append(0)
            if q_negs:
                for da in d_atoms:
                    for qn in q_negs: curr_ap.append([da, qn]); curr_ty.append(1)
            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap), 'types': curr_ty})
            atom_pairs.extend(curr_ap)

        # 🔥 \u6539\u52a8\u70b9 1：\u79fb\u9664 torch.softmax，\u76f4\u63a5\u83b7\u53d6\u539f\u59cb Logits
        # self.nli_model.predict \u9ed8\u8ba4\u8fd4\u56de numpy \u6570\u7ec4\u683c\u5f0f\u7684 logits
        f_logits = self.nli_model.predict(full_pairs, batch_size=32, show_progress_bar=False) if full_pairs else []
        a_logits = self.nli_model.predict(atom_pairs, batch_size=128, show_progress_bar=False) if atom_pairs else []

        raw_scores = []
        for i in range(len(candidate_ids)):
            # \u8fd9\u91cc\u7d22\u5f15 0, 1, 2 \u5bf9\u5e94\u7684\u903b\u8f91\u8bed\u4e49\u4e0d\u53d8，\u53ea\u662f\u6570\u503c\u4ece [0,1] \u53d8\u4e3a Logits
            s_full = f_logits[i][0] if len(f_logits) > 0 else 0.0
            s_dir, s_ind = 0.0, 0.0
            m = atom_meta[i]
            if m['count'] > 0:
                sl = a_logits[m['start']: m['start'] + m['count']]
                ty = m['types']

                # \u76f4\u63a5\u4ece Logits \u4e2d\u63d0\u53d6\u5bf9\u5e94\u6807\u7b7e\u7684\u5206\u6570
                dirs = [sl[k][0] for k, t in enumerate(ty) if t == 0]
                inds = [sl[k][1] for k, t in enumerate(ty) if t == 1]

                if dirs: s_dir = np.mean(sorted(dirs)[-g:])
                if inds:
                    raw = np.mean(sorted(inds)[-g:])
                    s_ind = raw

            s_atom = (W_DIRECT * s_dir) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))

        # 🔥 \u6539\u52a8\u70b9 2：Z-score \u5f52\u4e00\u5316\u903b\u8f91\u4fdd\u6301\u4e0d\u53d8
        # \u8fd9\u5bf9\u4e8e\u5904\u7406\u65e0\u754c\u9650\u7684 Logits \u81f3\u5173\u91cd\u8981
        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1:
            mean_s, std_s = np.mean(raw_scores), np.std(raw_scores)
            # \u4f7f\u7528 LaTeX \u8868\u793a\u5f52\u4e00\u5316\u516c\u5f0f: $z = \frac{x - \mu}{\sigma}$
            norm_scores = (raw_scores - mean_s) / std_s if std_s > 1e-9 else (raw_scores - mean_s)
        else:
            norm_scores = raw_scores
        return norm_scores.tolist()

    def _print_fold_summary(self, fold_summaries):
        print("\n" + "=" * 90)
        print(f"📊 Final Results ({NUM_FOLDS} Experiments) | Format: Mean (±Std)")
        print("=" * 90)

        def get_stat(vals):
            # \u4e58\u4ee5100\u5e76\u4fdd\u7559\u4e24\u4f4d\u5c0f\u6570
            vals_pct = np.array(vals) * 100
            return f"{np.mean(vals_pct):.2f} (±{np.std(vals_pct):.2f})"

        for k in KS:
            r_stat = get_stat(fold_summaries[k]['recall'])
            n_stat = get_stat(fold_summaries[k]['ndcg'])
            print(f"K={k:<2} | Recall: {r_stat} | NDCG: {n_stat}")
        print("=" * 90)


if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        exp = PileOfLawExperiment(df)
        exp.build_corpus()
        exp.preprocess_queries()
        exp.run_5fold_eval()
    else:
        print(f"❌ File not found: {CSV_PATH}")