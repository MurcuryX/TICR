"""Dependency-free BM25 baseline on the canonical query-level loader."""
import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter

import numpy as np
from aligned_eval_data import load_aligned

TOKEN = re.compile(r"[A-Za-z0-9]+")
KS = [3, 5, 10, 20]


def toks(text):
    return TOKEN.findall(text.lower())


def build_index(docs):
    token_docs = [toks(x) for x in docs]
    lengths = np.array([len(x) for x in token_docs], dtype=float)
    avg = max(lengths.mean(), 1.0)
    df = Counter(t for row in token_docs for t in set(row))
    return token_docs, lengths, avg, df


def score_bm25(index, query, k1=1.2, b=0.75):
    token_docs, lengths, avg, df = index
    q = Counter(toks(query))
    scores = np.zeros(len(token_docs), dtype=float)
    for i, row in enumerate(token_docs):
        tf = Counter(row)
        norm = 1 - b + b * lengths[i] / avg
        for term, qtf in q.items():
            if term not in tf: continue
            idf = math.log(1 + (len(token_docs) - df[term] + 0.5) / (df[term] + 0.5))
            scores[i] += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * norm)
    return scores


def ndcg(order, gt, k):
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(order[:k]) if x in gt)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / ideal if ideal else 0.0


def run(path, kind, paired_only=True):
    docs, records, audit = load_aligned(path, kind, paired_only=paired_only)
    index = build_index(docs)
    values = {f"hit@{k}": [] for k in KS}
    values.update({f"ndcg@{k}": [] for k in KS})
    for record in records:
        order = np.argsort(-score_bm25(index, record["q"]))[:max(KS)]
        for k in KS:
            values[f"hit@{k}"].append(float(any(x in record["gt"] for x in order[:k])))
            values[f"ndcg@{k}"].append(ndcg(order, record["gt"], k))
    metrics = {k: round(100 * float(np.mean(v)), 4) for k, v in values.items()}
    blocks = {k: np.array_split(np.asarray(v, dtype=float), 5) for k, v in values.items()}
    block_means = {k: [round(100 * float(np.mean(b)), 4) for b in bs] for k, bs in blocks.items()}
    block_std = {k: round(float(np.std(m)), 4) for k, m in
                 ((k, np.asarray(v, dtype=float)) for k, v in block_means.items())}
    return {"queries": len(records), "documents": len(docs), "data_audit": audit,
            "metrics": metrics, "block_means": block_means, "block_std": block_std}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="bm25_aligned.json")
    ap.add_argument("--population", choices=["paired", "full_clean_valid"], default="paired")
    args = ap.parse_args()
    paired = args.population == "paired"
    out = {"Bill-Contra": run(Path(args.data_dir) / "Bill-Contra.csv", "bill", paired),
           "Juris-Logic": run(Path(args.data_dir) / "Juris-Logic.csv", "juris", paired)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
