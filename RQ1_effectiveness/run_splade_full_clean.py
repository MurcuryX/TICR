"""Official SPLADE baseline on the canonical full clean-valid populations."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from aligned_eval_data import load_aligned

KS = [3, 5, 10, 20]


def ndcg(order, gt, k):
    dcg = sum(1 / np.log2(i + 2) for i, idx in enumerate(order[:k]) if idx in gt)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / ideal if ideal else 0.0


@torch.inference_mode()
def encode_sparse(texts, tokenizer, model, device, batch_size, max_length):
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="SPLADE encode"):
        batch = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        weights = torch.log1p(torch.relu(logits))
        weights = weights * batch["attention_mask"].unsqueeze(-1)
        vectors.append(torch.max(weights, dim=1).values.cpu())
    return torch.cat(vectors, dim=0)


def evaluate(docs, records, tokenizer, model, device, batch_size, max_length):
    doc_vectors = encode_sparse(
        docs, tokenizer, model, device, batch_size, max_length)
    query_vectors = encode_sparse(
        [row["q"] for row in records], tokenizer, model, device,
        batch_size, max_length)
    per_query = []
    score_batch = 64
    for start in tqdm(range(0, len(records), score_batch), desc="SPLADE score"):
        scores = query_vectors[start:start + score_batch] @ doc_vectors.T
        orders = torch.topk(scores, 20, dim=1).indices.numpy()
        for order, row in zip(orders, records[start:start + score_batch]):
            metrics = {}
            for k in KS:
                metrics[f"hit@{k}"] = float(
                    any(idx in row["gt"] for idx in order[:k]))
                metrics[f"ndcg@{k}"] = ndcg(order, row["gt"], k)
            per_query.append(metrics)
    blocks = np.array_split(np.arange(len(per_query)), 5)
    result, block_std = {}, {}
    for metric in per_query[0]:
        values = np.asarray([row[metric] for row in per_query])
        result[metric] = round(100 * float(values.mean()), 4)
        block_std[metric] = round(
            100 * float(np.std([values[idx].mean() for idx in blocks])), 4)
    return result, block_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="splade_full_clean.json")
    ap.add_argument("--model", default="naver/splade-cocondenser-ensembledistil")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(args.device).eval()
    payload = {"protocol": {
        "population": "canonical full clean-valid",
        "model": args.model,
        "pooling": "max(log(1 + relu(MLM logits)))",
        "retrieval": "exact sparse dot product",
        "blocks": 5,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }}
    for name, filename, kind in [
        ("Bill-Contra", "Bill-Contra.csv", "bill"),
        ("Juris-Logic", "Juris-Logic.csv", "juris"),
    ]:
        docs, records, audit = load_aligned(
            Path(args.data_dir) / filename, kind, paired_only=False)
        result, block_std = evaluate(
            docs, records, tokenizer, model, args.device,
            args.batch_size, args.max_length)
        payload[name] = {
            "queries": len(records), "documents": len(docs),
            "data_audit": audit, "results": result, "block_std": block_std,
        }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
