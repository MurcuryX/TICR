"""Auditable Stage-1 controls for TICR using the released cached atoms.

This script deliberately evaluates retrieval only: all variants use the same
frozen BGE encoder, corpus, exact cosine search, and cutoff.  It does not
invoke an LLM or an NLI reranker.  The cached decomposition/inversion fields
in the two source CSV files are the inputs to every atomic variant.
"""
import argparse
import ast
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from aligned_eval_data import load_aligned

INSTRUCTION = "Represent this sentence for searching relevant passages: "


def parsed(value):
    try:
        x = ast.literal_eval(value) if value and value.strip() else []
        return x if isinstance(x, list) else []
    except (ValueError, SyntaxError):
        return []


def flatten(items):
    out = []
    for item in items or []:
        if isinstance(item, list):
            out.extend(flatten(item))
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def inv_by_atom(items, r):
    """Use the first r cached inversions for every source atom."""
    if not items:
        return []
    if isinstance(items[0], str):
        return [x for x in items[:r] if x.strip()]
    return [x for group in items for x in (group[:r] if isinstance(group, list) else []) if x.strip()]


def load_bill(path):
    docs, queries, seen = [], [], {}

    def add_doc(text):
        if not text:
            return None
        if text not in seen:
            seen[text] = len(docs)
            docs.append(text)
        return seen[text]

    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            conflicts = [row.get("Conflict1 (Contradiction)", ""), row.get("Conflict2 (Contradiction)", "")]
            entails = [row.get("Paraphrase_Structure_1 (Entailment)", ""), row.get("Paraphrase_Structure_2 (Entailment)", "")]
            conflict_ids = {add_doc(x) for x in conflicts if x}
            entail_ids = {add_doc(x) for x in entails if x}
            q = row.get("Original_Text", "").strip()
            if q and conflict_ids:
                queries.append({"q": q, "positive_atoms": flatten(parsed(row.get("Original_Text_atoms", ""))),
                                "inverse_atoms": inv_by_atom(parsed(row.get("Neg_Original_Text_atoms", "")), 3),
                                "gt": conflict_ids, "hard_entail": entail_ids})
    return docs, queries


def load_juris(path):
    docs, queries, seen = [], [], {}

    def add_doc(text):
        if text not in seen:
            seen[text] = len(docs)
            docs.append(text)
        return seen[text]

    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            conflict = row.get("conflict", "").strip()
            entail = row.get("entail", "").strip()
            q = row.get("query", "").strip()
            if q and conflict:
                gt = {add_doc(conflict)}
                hard_entail = {add_doc(entail)} if entail else set()
                queries.append({"q": q, "positive_atoms": flatten(parsed(row.get("query_atoms", ""))),
                                "inverse_atoms": inv_by_atom(parsed(row.get("negs_sourse_atoms", "")), 3),
                                "gt": gt, "hard_entail": hard_entail})
    return docs, queries


def match_count(values, n, fallback):
    values = [x for x in values if x] or [fallback]
    return [values[i % len(values)] for i in range(n)]


def probes(qdata, variant, shuffled_inversions=None):
    q = qdata["q"]
    if variant == "original":
        return [q]
    if variant == "decomposition_only":
        return [f"{q} {atom}" for atom in qdata["positive_atoms"]] or [q]
    if variant == "direct_negation":
        return [f"{q} NOT ({atom})" for atom in qdata["positive_atoms"]] or [q]
    if variant == "shuffled_inversion":
        foreign = match_count(shuffled_inversions or [], len(qdata["inverse_atoms"]), q)
        return [f"{q} {atom}" for atom in foreign]
    if variant == "query_plus_inverse":
        return [f"{q} {atom}" for atom in qdata["inverse_atoms"]] or [q]
    if variant == "inverse_only":
        return list(qdata["inverse_atoms"]) or [q]
    raise ValueError(variant)


def rank_and_probe_metrics(model, docs, queries, device, name):
    doc_emb = model.encode(docs, batch_size=64, convert_to_tensor=True, normalize_embeddings=True,
                           show_progress_bar=True, device=device)
    variants = ["original", "decomposition_only", "direct_negation",
                "shuffled_inversion", "inverse_only", "query_plus_inverse"]
    result = {v: {"hit@20": [], "mrr": [], "pair_margin": [], "pair_win_rate": []} for v in variants}
    for q_index, qdata in enumerate(tqdm(queries, desc=f"{name}: controls")):
        shuffled = queries[(q_index + 1) % len(queries)]["inverse_atoms"]
        for variant in variants:
            batch = [INSTRUCTION + x for x in probes(qdata, variant, shuffled)]
            q_emb = model.encode(batch, batch_size=64, convert_to_tensor=True, normalize_embeddings=True,
                                 show_progress_bar=False, device=device)
            scores = torch.max(q_emb @ doc_emb.T, dim=0).values.detach().cpu().numpy()
            order = np.argsort(-scores)
            ranks = np.where(np.isin(order, list(qdata["gt"])))[0]
            result[variant]["hit@20"].append(float(np.any(ranks < 20)))
            result[variant]["mrr"].append(1.0 / (int(ranks.min()) + 1) if len(ranks) else 0.0)
            if qdata["hard_entail"]:
                margin = max(scores[list(qdata["gt"])]) - max(scores[list(qdata["hard_entail"])])
                result[variant]["pair_margin"].append(float(margin))
                result[variant]["pair_win_rate"].append(float(margin > 0))
    return {v: {m: round(float(np.mean(values)) * (100 if m != "pair_margin" else 1), 4)
                for m, values in measures.items() if values}
            for v, measures in result.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--output", default="stage1_controls.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model = SentenceTransformer(os.environ.get("TICR_MODEL_PATH", "BAAI/bge-base-en-v1.5"), device=args.device)
    payload = {}
    for name, loader, filename in [("Bill-Contra", load_bill, "Bill-Contra.csv"),
                                   ("Juris-Logic", load_juris, "Juris-Logic.csv")]:
        kind = "bill" if name == "Bill-Contra" else "juris"
        docs, queries, metadata = load_aligned(Path(args.data_dir) / filename, kind)
        print(f"{name}: {len(queries)} queries, {len(docs)} documents")
        payload[name] = {"queries": len(queries), "documents": len(docs), "data_audit": metadata,
                         "results": rank_and_probe_metrics(model, docs, queries, args.device, name)}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
