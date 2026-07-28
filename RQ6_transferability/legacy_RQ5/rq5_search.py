import os


import pandas as pd
import numpy as np
import ast
import torch
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm

# ================= \u914d\u7f6e\u533a\u57df =================
CSV_PATH = 'RQ5_real.csv'
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K_PER_PASS = 20
KS = [3, 5]

# 🔥 \u7ec6\u81f4\u641c\u7d22\u7f51\u683c
G_RANGE = [1, 2, 3]
W_VEC_RANGE = np.arange(0.0, 1.05, 0.05)  # 0.0, 0.05, ..., 1.0
W_LOGIC_FULL_RANGE = np.arange(0.0, 1.05, 0.05)  # 0.0, 0.05, ..., 1.0

# \u56fa\u5b9a\u9879
W_DIRECT = 0.0
W_INDIRECT = 1.0


# ================= \u8f85\u52a9\u51fd\u6570 =================
def parse_nested_negations(val):
    if pd.isna(val): return []
    s_val = str(val).strip()
    if s_val.startswith('[') and s_val.endswith(']'):
        try:
            parsed = ast.literal_eval(s_val)
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], str): return [parsed]
                return parsed
        except:
            pass
    return []


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


# ================= \u9884\u8ba1\u7b97\u4e0e\u641c\u7d22\u7c7b =================
class TICR_NDCG_Searcher:
    def __init__(self, df):
        self.df = df
        self.corpus_hashes = {}
        self.doc_id_map = {}
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)

    def build_corpus(self):
        print("🏗️ Building Corpus...")
        unique_docs = []
        all_doc_cols = ['col2_conflict_en', 'col3_entail1_en', 'col4_entail2_en', 'col5_entail3_en']
        for _, row in self.df.iterrows():
            for col in all_doc_cols:
                text = str(row.get(col, '')).strip()
                if not text or text.lower() == 'nan': continue
                h = get_text_hash(text)
                if h not in self.corpus_hashes:
                    doc_id = len(unique_docs)
                    unique_docs.append(text)
                    self.corpus_hashes[h] = doc_id
                    self.doc_id_map[doc_id] = {'text': text}
        self.corpus_texts = unique_docs
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True)

    def precompute_data(self):
        print("🛠️ Pre-calculating Scores & Logits...")
        self.cached_queries = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            q_text = str(row.get('col1_query_en', '')).strip()
            gt_text = str(row.get('col2_conflict_en', '')).strip()
            if not q_text: continue

            gt_id = self.corpus_hashes.get(get_text_hash(gt_text))
            raw_negs = parse_nested_negations(row.get('col11_query_negations'))

            # 1. 3M-Pass Max-Pooling \u68c0\u7d22
            atoms = [str(a).strip() for sub in raw_negs for a in sub if str(a).strip()]
            cand_queries = [f"{q_text} {a}" for a in atoms] if atoms else [q_text]

            instruction = "Represent this sentence for searching relevant passages: "
            q_embs_3m = self.embedder.encode([instruction + q for q in cand_queries], convert_to_tensor=True)
            sim_matrix = util.cos_sim(q_embs_3m, self.corpus_embeddings)
            max_sims, _ = torch.max(sim_matrix, dim=0)
            cand_ids = torch.topk(max_sims, k=min(TOP_K_PER_PASS, len(self.corpus_texts))).indices.tolist()

            # 2. \u7ec4\u4ef6\u5206\u9884\u8ba1\u7b97
            q_emb_origin = self.embedder.encode(instruction + q_text, convert_to_tensor=True)
            s_vecs = util.cos_sim(q_emb_origin, self.corpus_embeddings[cand_ids]).squeeze(0).cpu().numpy()

            full_pairs, atom_pairs, atom_meta = [], [], []
            q_negs = flatten_list(raw_negs)
            for cid in cand_ids:
                doc_text = self.doc_id_map[cid]['text']
                full_pairs.append([q_text, doc_text])
                curr_ap = [[doc_text, qn] for qn in q_negs[:10]]
                atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap)})
                atom_pairs.extend(curr_ap)

            f_logits = self.nli_model.predict(full_pairs, show_progress_bar=False)
            a_logits = self.nli_model.predict(atom_pairs, show_progress_bar=False) if atom_pairs else []

            query_data = []
            for i, cid in enumerate(cand_ids):
                query_data.append({
                    'is_gt': cid == gt_id,
                    's_vec': s_vecs[i],
                    's_full_logit': f_logits[i][0],
                    'atom_ent_logits': [a_logits[atom_meta[i]['start'] + k][1] for k in range(atom_meta[i]['count'])]
                })
            self.cached_queries.append(query_data)

    def run_fine_search(self):
        print(f"\n🔍 Searching grid: g={G_RANGE}, step=0.05, optimizing for NDCG@3...")
        results = []

        for g_val in G_RANGE:
            for wv in W_VEC_RANGE:
                wl = 1.0 - wv
                for wlf in W_LOGIC_FULL_RANGE:
                    wla = 1.0 - wlf

                    q_ndcg3, q_r3 = [], []
                    for q_data in self.cached_queries:
                        # \u8ba1\u7b97\u5f53\u524d\u53c2\u6570\u5206\u6570
                        raw_logic = []
                        for item in q_data:
                            s_ind = np.mean(sorted(item['atom_ent_logits'])[-g_val:]) if item[
                                'atom_ent_logits'] else 0.0
                            raw_logic.append((wlf * item['s_full_logit']) + (wla * s_ind))

                        # Z-score \u5f52\u4e00\u5316
                        raw_logic = np.array(raw_logic)
                        std = np.std(raw_logic)
                        norm_logic = (raw_logic - np.mean(raw_logic)) / std if std > 1e-9 else raw_logic - np.mean(
                            raw_logic)

                        # \u878d\u5408\u4e0e NDCG@3 \u8ba1\u7b97
                        ranks = []
                        for i, item in enumerate(q_data):
                            ranks.append({'is_gt': item['is_gt'], 'score': (wv * item['s_vec']) + (wl * norm_logic[i])})
                        ranks.sort(key=lambda x: x['score'], reverse=True)

                        # NDCG@3 \u903b\u8f91
                        top3 = ranks[:3]
                        dcg = sum(1.0 / np.log2(idx + 2) for idx, r in enumerate(top3) if r['is_gt'])
                        idcg = 1.0  # RQ5 \u4e2d GT \u53ea\u6709\u4e00\u4e2a
                        q_ndcg3.append(dcg / idcg)
                        q_r3.append(1.0 if any(r['is_gt'] for r in top3) else 0.0)

                    results.append({
                        'g': g_val, 'W_VEC': wv, 'W_LOGIC_FULL': wlf,
                        'NDCG@3': np.mean(q_ndcg3) * 100, 'R@3': np.mean(q_r3) * 100
                    })

        # \u8f93\u51fa Top 15
        df_res = pd.DataFrame(results).sort_values(by='NDCG@3', ascending=False)
        print("\n" + "=" * 90)
        print(f"🏆 TOP 15 CONFIGURATIONS (Sorted by NDCG@3)")
        print("-" * 90)
        print(df_res.head(15).to_string(index=False))
        print("=" * 90)

        best = df_res.iloc[0]
        print(f"\n✅ Optimal Parameters Found:")
        print(f"-> g: {int(best['g'])}")
        print(f"-> W_VEC: {best['W_VEC']:.2f}, W_LOGIC: {1 - best['W_VEC']:.2f}")
        print(f"-> W_LOGIC_FULL: {best['W_LOGIC_FULL']:.2f}, W_LOGIC_ATOM: {1 - best['W_LOGIC_FULL']:.2f}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH).fillna('')
    searcher = TICR_NDCG_Searcher(df)
    searcher.build_corpus()
    searcher.precompute_data()
    searcher.run_fine_search()