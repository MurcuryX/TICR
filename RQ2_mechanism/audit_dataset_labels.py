"""Audit query-level counts and conflict/entailment label collisions.

This is a read-only audit. It does not silently relabel ambiguous examples.
"""
import argparse
import ast
import csv
import json
from collections import defaultdict


def parse(value):
    try:
        return ast.literal_eval(value) if value else []
    except (ValueError, SyntaxError):
        return []


def audit(path, kind):
    qcol = "Original_Text" if kind == "bill" else "query"
    groups = defaultdict(list)
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    for i, row in enumerate(rows):
        groups[row.get(qcol, "").strip()].append((i, row))
    missing_gt, missing_inv, collisions = [], [], []
    for q, items in groups.items():
        gt, entail, inversions = set(), set(), set()
        for _, row in items:
            if kind == "bill":
                gt.update(x.strip() for x in [row.get("Conflict1 (Contradiction)", ""), row.get("Conflict2 (Contradiction)", "")] if x.strip())
                entail.update(x.strip() for x in [row.get("Paraphrase_Structure_1 (Entailment)", ""), row.get("Paraphrase_Structure_2 (Entailment)", "")] if x.strip())
                raw = row.get("Neg_Original_Text_atoms", "")
            else:
                gt.add(row.get("conflict", "").strip())
                entail.add(row.get("entail", "").strip())
                raw = row.get("negs_sourse_atoms", "")
            value = parse(raw)
            for group in value if isinstance(value, list) else []:
                inversions.update(x.strip() for x in (group if isinstance(group, list) else [group]) if isinstance(x, str) and x.strip())
        gt.discard(""); entail.discard("")
        if not gt: missing_gt.append(q)
        elif not inversions: missing_inv.append(q)
        overlap = sorted(gt & entail)
        if overlap: collisions.append({"query": q, "rows": [i for i, _ in items], "texts": overlap})
    return {"rows": len(rows), "unique_queries": len(groups),
            "missing_ground_truth": len(missing_gt),
            "missing_inversion_among_gt": len(missing_inv),
            "label_collisions": collisions}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bill", required=True)
    ap.add_argument("--juris", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = {"Bill-Contra": audit(args.bill, "bill"),
              "Juris-Logic": audit(args.juris, "juris")}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
