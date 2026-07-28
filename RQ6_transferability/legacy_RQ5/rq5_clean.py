import os# ================= 1. \u914d\u7f6e\u533a\u57df =================
# \u5f3a\u5236\u4f7f\u7528\u56fd\u5185\u955c\u50cf
import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm



CSV_PATH = 'RQ5_real.csv'
JSON_PATH = 'final_dataset.json'
OUTPUT_JSON = 'final_dataset_dedup.json'

# \u6a21\u578b\u914d\u7f6e
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🔥 \u76f8\u4f3c\u5ea6\u9608\u503c
# 0.95 \u4ee5\u4e0a\u57fa\u672c\u662f\u539f\u53e5\u6539\u4e2a\u522b\u8bcd；0.85-0.90 \u662f\u610f\u601d\u9ad8\u5ea6\u96f7\u540c
SIMILARITY_THRESHOLD = 0.90


# ================= 2. \u6267\u884c\u53bb\u91cd =================

def run_semantic_dedup():
    print(f"🔄 Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    # 1. \u52a0\u8f7d Query \u5e76\u7f16\u7801
    print(f"📖 Reading Queries from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    queries = df['col1_query_en'].dropna().unique().tolist()

    # BGE \u63a8\u8350\u5728 Query \u524d\u52a0\u6307\u4ee4
    instruction = "Represent this sentence for searching relevant passages: "
    query_embeddings = model.encode([instruction + q for q in queries], convert_to_tensor=True)
    print(f"✅ Encoded {len(queries)} unique queries.")

    # 2. \u52a0\u8f7d\u8bed\u6599\u5e93
    if not os.path.exists(JSON_PATH):
        print(f"❌ File not found: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    corpus_texts = [item['content'] for item in corpus_data]

    # 3. \u7f16\u7801\u8bed\u6599\u5e93\u6bb5\u843d (\u5206\u6279\u5904\u7406\u9632\u6b62\u663e\u5b58\u6ea2\u51fa)
    print(f"🧮 Encoding Corpus ({len(corpus_texts)} items)...")
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32)

    # 4. \u8ba1\u7b97\u76f8\u4f3c\u5ea6\u5e76\u7b5b\u9009
    print("🔍 Searching for overly similar segments...")
    # \u8ba1\u7b97\u77e9\u9635：[\u8bed\u6599\u5e93\u6570\u91cf, Query\u6570\u91cf]
    # \u5bf9\u8bed\u6599\u5e93\u4e2d\u6bcf\u4e00\u9879，\u627e\u5230\u5b83\u4e0e\u6240\u6709 Query \u7684\u6700\u5927\u76f8\u4f3c\u5ea6
    cos_sim = util.cos_sim(corpus_embeddings, query_embeddings)
    max_sim_per_doc, _ = torch.max(cos_sim, dim=1)

    max_sim_list = max_sim_per_doc.cpu().tolist()

    # 5. \u6267\u884c\u8fc7\u6ee4
    clean_corpus = []
    removed_count = 0

    for i, item in enumerate(corpus_data):
        if max_sim_list[i] < SIMILARITY_THRESHOLD:
            clean_corpus.append(item)
        else:
            removed_count += 1
            # \u53ef\u9009：\u6253\u5370\u88ab\u5254\u9664\u7684\u4f8b\u5b50\u7528\u4e8e\u68c0\u67e5
            if removed_count < 5:
                print(f"   [Removed] Sim={max_sim_list[i]:.4f} | Content: {item['content'][:100]}...")

    # 6. \u4fdd\u5b58\u7ed3\u679c
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(clean_corpus, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 50)
    print(f"📊 Deduplication Report:")
    print(f"   Original Size: {len(corpus_data)}")
    print(f"   Removed Count: {removed_count}")
    print(f"   Final Size:    {len(clean_corpus)}")
    print(f"✅ Saved to: {OUTPUT_JSON}")
    print("=" * 50)


if __name__ == "__main__":
    run_semantic_dedup()