"""Compute-matched candidate-construction controls for TICR."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from aligned_eval_data import load_aligned

INS = "Represent this sentence for searching relevant passages: "
K, TOPR = 20, 3
W_VEC, W_LOGIC = .9, .1


def cycle(values, n, fallback):
    values = [x for x in values if x] or [fallback]
    return [values[i % len(values)] for i in range(n)]


def ndcg(rank, gt, k):
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(rank[:k]) if x in gt)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / idcg if idcg else 0


def rerank(q, inverse, cand, docs, origin, nli):
    direct = nli.predict([[q, docs[i]] for i in cand], batch_size=64,
                         show_progress_bar=False, convert_to_numpy=True)[:, 0]
    pairs = [(i, atom) for i in cand for atom in inverse]
    entail = nli.predict([[docs[i], atom] for i, atom in pairs], batch_size=512,
                         show_progress_bar=False, convert_to_numpy=True)[:, 1]
    local = np.array([np.mean(np.sort(entail[j*len(inverse):(j+1)*len(inverse)])[-TOPR:])
                      for j in range(len(cand))])
    raw = .5 * direct + .5 * local
    z = (raw - raw.mean()) / (raw.std() + 1e-9)
    final = W_VEC * np.asarray(origin) + W_LOGIC * z
    return [x for _, x in sorted(zip(final, cand), reverse=True)]


def variants(record, paraphrases, shuffled):
    q, inverse = record["q"], record["inverse_atoms"]
    n = len(inverse)
    positive = cycle(record["positive_atoms"], n, q)
    para = cycle(paraphrases, n, q)
    foreign = cycle(shuffled, n, q)
    return {
        "original_repeated": [q] * n,
        "decomposition_matched": [f"{q} {x}" for x in positive],
        "lexical_negation_matched": [f"{q} NOT ({x})" for x in positive],
        "llm_multiquery_matched": para,
        "shuffled_inversion_matched": [f"{q} {x}" for x in foreign],
        "ticr_inversion": [f"{q} {x}" for x in inverse],
    }


def run(name, docs, records, shuffle_pool, para_map, embed, nli, device):
    doc_emb = embed.encode(docs, convert_to_tensor=True, normalize_embeddings=True,
                           batch_size=64, show_progress_bar=True, device=device)
    keys = ["original_repeated", "decomposition_matched", "lexical_negation_matched",
            "llm_multiquery_matched", "shuffled_inversion_matched", "ticr_inversion"]
    ranks, latency = {k: [] for k in keys}, {k: [] for k in keys}
    total_probes = {k: 0 for k in keys}
    pool_index = {record["q"]: i for i, record in enumerate(shuffle_pool)}
    for record in tqdm(records, desc=name):
        index = pool_index[record["q"]]
        foreign = shuffle_pool[(index + 1) % len(shuffle_pool)]["inverse_atoms"]
        probes = variants(record, para_map.get(record["q"], []), foreign)
        q0 = embed.encode(INS + record["q"], convert_to_tensor=True,
                          normalize_embeddings=True, show_progress_bar=False, device=device)
        for key in keys:
            total_probes[key] += len(probes[key])
            torch.cuda.synchronize(); start = time.perf_counter()
            p = embed.encode([INS + x for x in probes[key]], convert_to_tensor=True,
                             normalize_embeddings=True, batch_size=64,
                             show_progress_bar=False, device=device)
            scores = torch.max(p @ doc_emb.T, dim=0).values
            cand = torch.topk(scores, K).indices.detach().cpu().tolist()
            origin = (q0 @ doc_emb[cand].T).detach().cpu().numpy()
            ranks[key].append(rerank(record["q"], record["inverse_atoms"], cand,
                                     docs, origin, nli))
            torch.cuda.synchronize(); latency[key].append((time.perf_counter() - start) * 1000)
    out = {}
    for key in keys:
        out[key] = {f"hit@{k}": round(100*np.mean([
            any(x in record["gt"] for x in rank[:k]) for rank, record in zip(ranks[key], records)]), 4)
            for k in [3, 5, 10, 20]}
        out[key].update({f"ndcg@{k}": round(100*np.mean([
            ndcg(rank, record["gt"], k) for rank, record in zip(ranks[key], records)]), 4)
            for k in [3, 5, 10, 20]})
        out[key]["mean_latency_ms_excl_generation"] = round(float(np.mean(latency[key])), 3)
        out[key]["total_candidate_probes"] = total_probes[key]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--paraphrases", default="multiquery_paraphrases.json")
    ap.add_argument("--out", default="compute_matched_controls.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", choices=["Bill-Contra", "Juris-Logic"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    embed = SentenceTransformer(os.environ["TICR_EMBED_PATH"], device=args.device)
    nli = CrossEncoder(os.environ["TICR_NLI_PATH"], device=args.device)
    generated = json.load(open(args.paraphrases, encoding="utf-8"))
    output = {"protocol": {"candidate_k": K, "same_probe_count_per_query": True,
                            "same_dual_path_reranker": True, "same_nli_calls_per_query": True,
                            "generation_latency_excluded": True}}
    for name, filename, kind in [("Bill-Contra", "Bill-Contra.csv", "bill"),
                                  ("Juris-Logic", "Juris-Logic.csv", "juris")]:
        if args.dataset and name != args.dataset:
            continue
        docs, all_records, audit = load_aligned(Path(args.data_dir) / filename, kind)
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must satisfy 0 <= index < num-shards")
        records = all_records[args.shard_index::args.num_shards]
        para_map = {x["q"]: x["views"] for x in generated[name]["records"]}
        output[name] = {"queries": len(records), "documents": len(docs), "data_audit": audit,
                        "shard_index": args.shard_index, "num_shards": args.num_shards,
                        "results": run(name, docs, records, all_records, para_map, embed, nli, args.device)}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
