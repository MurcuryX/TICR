"""Exact BGE baseline on the canonical full clean-valid populations."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from aligned_eval_data import load_aligned

INS = "Represent this sentence for searching relevant passages: "
KS = [3, 5, 10, 20]


def ndcg(order, gt, k):
    dcg = sum(1 / np.log2(i + 2) for i, idx in enumerate(order[:k]) if idx in gt)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / ideal if ideal else 0.0


def evaluate(docs, records, model, device):
    doc_emb = model.encode(
        docs, batch_size=64, convert_to_tensor=True, normalize_embeddings=True,
        show_progress_bar=True, device=device)
    queries = [INS + row["q"] for row in records]
    query_emb = model.encode(
        queries, batch_size=64, convert_to_tensor=True, normalize_embeddings=True,
        show_progress_bar=True, device=device)
    orders = torch.argsort(query_emb @ doc_emb.T, dim=1, descending=True)[:, :20].cpu().numpy()
    per_query = []
    for order, row in zip(orders, records):
        metrics = {}
        for k in KS:
            metrics[f"hit@{k}"] = float(any(idx in row["gt"] for idx in order[:k]))
            metrics[f"ndcg@{k}"] = ndcg(order, row["gt"], k)
        per_query.append(metrics)
    result, block_std = {}, {}
    blocks = np.array_split(np.arange(len(per_query)), 5)
    for metric in per_query[0]:
        values = np.asarray([row[metric] for row in per_query])
        result[metric] = round(100 * float(values.mean()), 4)
        block_std[metric] = round(100 * float(np.std([values[idx].mean() for idx in blocks])), 4)
    return result, block_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="bge_full_clean.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model = SentenceTransformer(os.environ["TICR_EMBED_PATH"], device=args.device)
    payload = {"protocol": {"population": "canonical full clean-valid",
                            "retrieval": "exact cosine", "blocks": 5}}
    for name, filename, kind in [
        ("Bill-Contra", "Bill-Contra.csv", "bill"),
        ("Juris-Logic", "Juris-Logic.csv", "juris"),
    ]:
        docs, records, audit = load_aligned(
            Path(args.data_dir) / filename, kind, paired_only=False)
        result, block_std = evaluate(docs, records, model, args.device)
        payload[name] = {"queries": len(records), "documents": len(docs),
                         "data_audit": audit, "results": result,
                         "block_std": block_std}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
