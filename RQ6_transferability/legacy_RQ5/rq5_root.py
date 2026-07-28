import os
# ================= 1. \u914d\u7f6e\u533a\u57df =================
# \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf
import json
import pandas as pd
import numpy as np
import ast
import torch
import re
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
import hashlib


CSV_PATH = 'RQ5_real.csv'
JSON_PATH = 'final_dataset_clean.json'

# \u6a21\u578b\u914d\u7f6e
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# \u8bc4\u6d4b\u53c2\u6570 (from rq5.py)
TOP_K_PER_PASS = 20
KS = [3, 5, 10, 20]
g = 3

# 🔥 \u6838\u5fc3\u53c2\u6570 (\u4e25\u683c\u4fdd\u7559\u60a8\u7684\u903b\u8f91)
W_VEC = 0.9
W_LOGIC = 0.1
W_LOGIC_FULL = 0.5
W_LOGIC_ATOM = 0.5
W_DIRECT = 0.0
W_INDIRECT = 1.0



# ================= 2. \u8f85\u52a9\u5de5\u5177 =================
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

    groups, current_group = [], []
    for line in s_val.split('\n'):
        line = line.strip()
        if not line: continue
        if "Atoms" in line and "Negations:" in line:
            if current_group: groups.append(current_group); current_group = []
            continue
        clean = ""
        if line.startswith("- "):
            clean = line[2:].strip()
        elif re.match(r'^\d+\.\s', line):
            clean = re.sub(r'^\d+\.\s', '', line).strip()
        elif len(line) > 5:
            clean = line
        if clean: current_group.append(clean)
    if current_group: groups.append(current_group)
    return groups


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


# ================= 3. \u6700\u7ec8\u8bc4\u6d4b\u7cfb\u7edf =================
class Final_Clean_Evaluator:
    def __init__(self, df, json_data):
        self.df = df
        self.json_data = json_data

        print(f"🔄 Loading Models on {DEVICE}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        self.nli_model = CrossEncoder(NLI_MODEL_NAME, device=DEVICE)

        self.corpus_texts = []
        self.corpus_embeddings = None
        self.doc_id_map = {}  # doc_id -> text
        self.text_hash_to_id = {}
        self.query_cache = []

    def build_corpus(self):
        print("🏗️ Building Hybrid Corpus (JSON Clean + CSV Conflict)...")
        unique_docs = []

        # 1. \u4f18\u5148\u5165\u5e93 CSV \u4e2d\u7684 Conflict \u5217 (\u786e\u4fdd GT \u5b58\u5728)
        # \u5bf9\u5e94 "RQ5_real\u7684conflict\u5217\u5165\u5e93"
        count_csv = 0
        for idx, row in self.df.iterrows():
            text = str(row.get('col2_conflict_en', '')).strip()
            if not text or text.lower() == 'nan': continue

            h = get_text_hash(text)
            if h not in self.text_hash_to_id:
                doc_id = len(unique_docs)
                unique_docs.append(text)
                self.text_hash_to_id[h] = doc_id
                self.doc_id_map[doc_id] = {'text': text}
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
                self.doc_id_map[doc_id] = {'text': text}
                count_json += 1

        self.corpus_texts = unique_docs
        print(f"   -> Added {count_csv} GT docs from CSV.")
        print(f"   -> Added {count_json} docs from Clean JSON.")
        print(f"🧮 Encoding Total {len(unique_docs)} documents...")
        self.corpus_embeddings = self.embedder.encode(unique_docs, convert_to_tensor=True, show_progress_bar=True)

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

            # \u539f\u5b50\u89e3\u6790
            nested_negs = parse_nested_negations(row.get('col11_query_negations'))

            # \u9884\u8ba1\u7b97\u539f\u59cb Query \u5411\u91cf
            instruction = ""
            q_emb_origin = self.embedder.encode(instruction + q_text, convert_to_tensor=True)

            self.query_cache.append({
                'sent0': q_text,
                'q_emb_origin': q_emb_origin,
                'raw_negs': nested_negs,
                'gt_ids': gt_ids
            })
            if gt_ids: valid_count += 1
        print(f"✅ Processed {len(self.query_cache)} queries (Valid GT found for {valid_count}).")

    def _batch_calculate_logic(self, candidate_ids, q_data):
        full_pairs, atom_pairs, atom_meta = [], [], []
        sent0 = q_data['sent0']
        q_negs = flatten_list(q_data['raw_negs'])

        for doc_id in candidate_ids:
            doc_data = self.doc_id_map[doc_id]
            doc_text = doc_data['text']

            full_pairs.append([sent0, doc_text])
            curr_ap, curr_ty = [], []
            if q_negs:
                for qn in q_negs[:10]:
                    curr_ap.append([doc_text, qn])
                    curr_ty.append(1)

            atom_meta.append({'start': len(atom_pairs), 'count': len(curr_ap), 'types': curr_ty})
            atom_pairs.extend(curr_ap)

        f_probs = torch.softmax(torch.tensor(self.nli_model.predict(full_pairs, show_progress_bar=False)),
                                dim=1).numpy()
        a_probs = []
        if atom_pairs:
            a_probs = torch.softmax(torch.tensor(self.nli_model.predict(atom_pairs, show_progress_bar=False)),
                                    dim=1).numpy()

        raw_scores = []
        for i in range(len(candidate_ids)):
            s_full = f_probs[i][0]  # Index 0 = Contradiction
            s_ind = 0.0

            m = atom_meta[i]
            if m['count'] > 0 and len(a_probs) > 0:
                sl = a_probs[m['start']: m['start'] + m['count']]
                ty = m['types']
                inds = [sl[k][1] for k, t in enumerate(ty) if t == 1]  # Index 1 = Entailment
                if inds:
                    raw = np.mean(sorted(inds)[-g:])
                    s_ind = raw

            s_atom = (W_DIRECT * 0.0) + (W_INDIRECT * s_ind)
            raw_scores.append((W_LOGIC_FULL * s_full) + (W_LOGIC_ATOM * s_atom))

        # Z-Score Normalization
        raw_scores = np.array(raw_scores)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-9:
            return ((raw_scores - np.mean(raw_scores)) / np.std(raw_scores)).tolist()
        return (raw_scores - np.mean(raw_scores)).tolist()

    def run_eval(self):
        print(f"\n🚀 Running Final Evaluation on {len(self.query_cache)} queries...")
        metrics = {k: {'recall': [], 'ndcg': []} for k in KS}

        for q_data in tqdm(self.query_cache, desc="Evaluating"):
            # ======================================================
            # Stage 1: 3M-Pass Max-Pooling Retrieval
            # ======================================================
            cand_queries = []
            if q_data['raw_negs']:
                for sub_list in q_data['raw_negs']:
                    for atom in sub_list:
                        atom_str = str(atom).strip()
                        if atom_str: cand_queries.append(f"{q_data['sent0']} {atom_str}")
            if not cand_queries: cand_queries = [q_data['sent0']]

            q_embs_3m = self.embedder.encode(
                ["Represent this sentence for searching relevant passages: " + q for q in cand_queries],
                convert_to_tensor=True)
            sim_matrix = util.cos_sim(q_embs_3m, self.corpus_embeddings)
            max_sims, _ = torch.max(sim_matrix, dim=0)

            top_vals, top_indices = torch.topk(max_sims, k=min(TOP_K_PER_PASS, len(self.corpus_texts)))
            cand_ids = top_indices.tolist()

            # ======================================================
            # Stage 2: Logic Reranking (TICR)
            # ======================================================
            s_vecs = util.cos_sim(q_data['q_emb_origin'], self.corpus_embeddings[cand_ids]).squeeze(0).cpu().numpy()
            s_logics = self._batch_calculate_logic(cand_ids, q_data)

            final_scores = (W_VEC * s_vecs) + (W_LOGIC * np.array(s_logics))

            final_rank = []
            for idx, cid in enumerate(cand_ids):
                final_rank.append({'cid': cid, 'is_gt': cid in q_data['gt_ids'], 'score': final_scores[idx]})

            final_rank.sort(key=lambda x: x['score'], reverse=True)

            # ======================================================
            # Stage 3: Metrics Calculation
            # ======================================================
            for k in KS:
                top_k = final_rank[:k]
                metrics[k]['recall'].append(1.0 if any(r['is_gt'] for r in top_k) else 0.0)

                dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(top_k) if r['is_gt'])
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(q_data['gt_ids']), k)))
                metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

        # Print Final Table
        print("\n" + "=" * 80)
        print(f"📊 Evaluation Results (Clean JSON + CSV GT) | W_VEC={W_VEC}, W_LOGIC={W_LOGIC}")
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

        evaluator = Final_Clean_Evaluator(df, json_data)
        evaluator.build_corpus()
        evaluator.preprocess_queries()
        evaluator.run_eval()
    else:
        print(f"❌ File missing: Please ensure {CSV_PATH} and {JSON_PATH} exist.")