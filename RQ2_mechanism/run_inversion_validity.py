"""Automatic NLI audit of cached atomic inversions."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder
from aligned_eval_data import canonical_text, load_aligned


def parse(value):
    try:
        value = ast.literal_eval(value) if value and value.strip() else []
        return value if isinstance(value, list) else []
    except (ValueError, SyntaxError):
        return []


def paired_atoms(row, kind):
    atoms_key = "Original_Text_atoms" if kind == "bill" else "query_atoms"
    inverse_key = "Neg_Original_Text_atoms" if kind == "bill" else "negs_sourse_atoms"
    atoms, inversions = parse(row.get(atoms_key, "")), parse(row.get(inverse_key, ""))
    atoms = [x.strip() for x in atoms if isinstance(x, str) and x.strip()]
    if inversions and isinstance(inversions[0], str):
        inversions = [inversions]
    pairs = []
    for atom, group in zip(atoms, inversions):
        if isinstance(group, list):
            pairs.extend((atom, x.strip()) for x in group[:3]
                         if isinstance(x, str) and x.strip())
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="inversion_validity.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    nli = CrossEncoder(os.environ["TICR_NLI_PATH"], device=args.device)
    output = {"protocol": {"diagnostic": "automatic, not human annotation",
                            "criterion": "NLI argmax label is contradiction",
                            "score": "contradiction logit margin over max(entailment, neutral)"}}
    for name, filename, kind in [("Bill-Contra", "Bill-Contra.csv", "bill"),
                                  ("Juris-Logic", "Juris-Logic.csv", "juris")]:
        _, canonical_records, audit = load_aligned(
            Path(args.data_dir) / filename, kind, paired_only=True)
        allowed = {record["q"] for record in canonical_records}
        by_query = {}
        with open(Path(args.data_dir) / filename, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                q = canonical_text(row.get("Original_Text" if kind == "bill" else "query", ""))
                if q in allowed:
                    by_query.setdefault(q, [])
                    seen = set(by_query[q])
                    for pair in paired_atoms(row, kind):
                        pair = tuple(canonical_text(x) for x in pair)
                        if all(pair) and pair not in seen:
                            by_query[q].append(pair)
                            seen.add(pair)
        pairs = [pair for values in by_query.values() for pair in values]
        logits = nli.predict([[a, inv] for a, inv in pairs], batch_size=128,
                             show_progress_bar=True, convert_to_numpy=True)
        valid = np.argmax(logits, axis=1) == 0
        margins = logits[:, 0] - np.max(logits[:, 1:], axis=1)
        query_rates = [np.mean([valid[i] for i in range(start, start + len(values))])
                       for start, values in _offsets(by_query.values()) if values]
        output[name] = {"queries": len(by_query), "data_audit": audit,
                        "paired_atomic_inversions": len(pairs),
                        "contradiction_argmax_rate": round(float(np.mean(valid))*100, 4),
                        "mean_contradiction_margin": round(float(np.mean(margins)), 4),
                        "median_query_validity_rate": round(float(np.median(query_rates))*100, 4),
                        "queries_all_inversions_valid_rate": round(float(np.mean(np.array(query_rates)==1))*100, 4)}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


def _offsets(groups):
    start = 0
    for group in groups:
        yield start, group
        start += len(group)


if __name__ == "__main__":
    main()
