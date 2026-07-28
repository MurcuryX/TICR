"""Unified factorial runner for the dual-path ablation.

All four variants share the same query-level paired data, BGE candidate pool,
candidate depth, NLI logits, and per-query normalization. Only the scoring
term retained after the shared forward/inverse logits is changed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from aligned_eval_data import load_aligned

INS = "Represent this sentence for searching relevant passages: "
K = 20
TOPR = 3


def ndcg(order, gt, k):
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(order[:k]) if x in gt)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / ideal if ideal else 0.0


def rank_variants(origin, direct, inverse, has_inverse=True):
    inv_z = (inverse - inverse.mean()) / (inverse.std() + 1e-9)
    direct_z = (direct - direct.mean()) / (direct.std() + 1e-9)
    if not has_inverse:
        scores = {
            "ticr_full": 0.9 * origin + 0.1 * direct_z,
            "without_forward": origin,
            "without_inverse": 0.9 * origin + 0.1 * direct_z,
            "without_logical_consistency": origin,
        }
        return {name: np.argsort(-value).tolist() for name, value in scores.items()}
    both = 0.5 * direct + 0.5 * inverse
    both_z = (both - both.mean()) / (both.std() + 1e-9)
    scores = {
        "ticr_full": 0.9 * origin + 0.1 * both_z,
        "without_forward": 0.9 * origin + 0.1 * inv_z,
        "without_inverse": 0.9 * origin + 0.1 * direct_z,
        "without_logical_consistency": origin,
    }
    return {name: np.argsort(-value).tolist() for name, value in scores.items()}


def run(name, docs, records, embed, nli, device):
    doc_emb = embed.encode(docs, batch_size=64, convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=True,
                           device=device)
    ranks = {name: [] for name in ["ticr_full", "without_forward",
                                    "without_inverse", "without_logical_consistency"]}
    for record in tqdm(records, desc=name):
        q = record["q"]
        inv_atoms = record["inverse_atoms"]
        q0 = embed.encode(INS + q, convert_to_tensor=True,
                          normalize_embeddings=True, show_progress_bar=False,
                          device=device)
        probes = [INS + f"{q} {atom}" for atom in inv_atoms] or [INS + q]
        p = embed.encode(probes, batch_size=64, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False,
                         device=device)
        candidate_scores = torch.max(p @ doc_emb.T, dim=0).values
        cand = torch.topk(candidate_scores, K).indices.detach().cpu().tolist()
        origin = (q0 @ doc_emb[cand].T).detach().cpu().numpy()
        direct = nli.predict([[q, docs[i]] for i in cand], batch_size=20,
                             show_progress_bar=False, convert_to_numpy=True)[:, 0]
        if inv_atoms:
            pairs = [[docs[i], atom] for i in cand for atom in inv_atoms]
            entail = nli.predict(pairs, batch_size=128, show_progress_bar=False,
                                 convert_to_numpy=True)[:, 1]
            inverse = np.array([
                np.mean(np.sort(entail[i * len(inv_atoms):(i + 1) * len(inv_atoms)])[-TOPR:])
                for i in range(len(cand))
            ])
        else:
            inverse = np.zeros_like(direct)
        ranked = rank_variants(origin, direct, inverse, has_inverse=bool(inv_atoms))
        for variant, local_order in ranked.items():
            ranks[variant].append([cand[i] for i in local_order])
    result = {}
    for variant, orders in ranks.items():
        result[variant] = {}
        for k in [3, 5, 10, 20]:
            result[variant][f"hit@{k}"] = round(100 * np.mean([
                any(x in record["gt"] for x in order[:k])
                for order, record in zip(orders, records)
            ]), 4)
            result[variant][f"ndcg@{k}"] = round(100 * np.mean([
                ndcg(order, record["gt"], k)
                for order, record in zip(orders, records)
            ]), 4)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="dualpath_factorial.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-mode", choices=["contiguous", "interleaved"],
                    default="contiguous")
    ap.add_argument("--population", choices=["paired", "full_clean_valid"], default="paired")
    args = ap.parse_args()
    embed = SentenceTransformer(os.environ["TICR_EMBED_PATH"], device=args.device)
    nli = CrossEncoder(os.environ["TICR_NLI_PATH"], device=args.device)
    output = {"protocol": {"candidate_k": K, "top_inverse_atoms": TOPR,
                            "candidate_construction": "TICR q+inverse max-pool",
                            "shared_nli_logits": True,
                            "variants": ["ticr_full", "without_forward",
                                         "without_inverse", "without_logical_consistency"]}}
    for name, filename, kind in [("Bill-Contra", "Bill-Contra.csv", "bill"),
                                 ("Juris-Logic", "Juris-Logic.csv", "juris")]:
        docs, records, audit = load_aligned(Path(args.data_dir) / filename, kind,
                                             paired_only=args.population == "paired")
        if args.num_shards > 1:
            if args.shard_mode == "contiguous":
                records = list(np.array_split(np.asarray(records, dtype=object),
                                              args.num_shards)[args.shard_index])
            else:
                records = records[args.shard_index::args.num_shards]
        output[name] = {"queries": len(records), "documents": len(docs),
                        "data_audit": audit, "population": args.population,
                        "results": run(name, docs, records, embed, nli, args.device)}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
