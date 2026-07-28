"""Automatic meaning-preservation audit for generated paraphrase controls."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder
from aligned_eval_data import canonical_text, load_aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrases", default="multiquery_paraphrases.json")
    ap.add_argument("--out", default="paraphrase_validity.json")
    ap.add_argument("--validated-out", default="multiquery_paraphrases_validated.json")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    data = json.load(open(args.paraphrases, encoding="utf-8"))
    nli = CrossEncoder(os.environ["TICR_NLI_PATH"], device=args.device)
    out = {"protocol": {"diagnostic": "automatic, not human annotation",
                         "criterion": "entailment argmax in both directions"}}
    validated = {"protocol": data["protocol"] | {"filter": "bidirectional NLI entailment"}}
    specs = [("Bill-Contra", "Bill-Contra.csv", "bill"),
             ("Juris-Logic", "Juris-Logic.csv", "juris")]
    for name, filename, kind in specs:
        _, canonical_records, audit = load_aligned(
            Path(args.data_dir) / filename, kind, paired_only=True)
        views_by_query = {}
        for row in data[name]["records"]:
            key = canonical_text(row["q"])
            views_by_query.setdefault(key, [])
            for view in row["views"]:
                view = canonical_text(view)
                if view and view not in views_by_query[key]:
                    views_by_query[key].append(view)
        rows = [{"q": record["q"], "views": views_by_query.get(record["q"], [])}
                for record in canonical_records]
        pairs = [(row["q"], view) for row in rows for view in row["views"]]
        forward = nli.predict(pairs, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
        backward = nli.predict([[b, a] for a, b in pairs], batch_size=128,
                               show_progress_bar=True, convert_to_numpy=True)
        f_valid, b_valid = np.argmax(forward, axis=1) == 1, np.argmax(backward, axis=1) == 1
        both = f_valid & b_valid
        cursor = 0
        filtered_rows = []
        for row in rows:
            count = len(row["views"])
            keep = [view for view, ok in zip(row["views"], both[cursor:cursor+count]) if ok]
            filtered_rows.append({"q": row["q"], "views": keep})
            cursor += count
        validated[name] = {"data_audit": audit, "records": filtered_rows,
                           "queries_with_valid_view": sum(bool(x["views"]) for x in filtered_rows)}
        out[name] = {
            "queries": len(rows),
            "queries_with_3_paraphrases": sum(len(x["views"]) == 3 for x in rows),
            "generated_paraphrases": len(pairs),
            "forward_entailment_rate": round(float(np.mean(f_valid))*100, 4),
            "backward_entailment_rate": round(float(np.mean(b_valid))*100, 4),
            "bidirectional_entailment_rate": round(float(np.mean(both))*100, 4),
        }
    json.dump(out, open(args.out, "w"), indent=2)
    json.dump(validated, open(args.validated_out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
