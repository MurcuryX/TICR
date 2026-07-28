"""Merge deterministic query shards from compute-matched evaluation."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bill", required=True)
    ap.add_argument("--juris", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bill = json.load(open(args.bill))
    shards = [json.load(open(x)) for x in args.juris]
    first = shards[0]["Juris-Logic"]
    names = list(first["results"])
    total = sum(x["Juris-Logic"]["queries"] for x in shards)
    merged_results = {}
    for name in names:
        fields = first["results"][name]
        merged_results[name] = {}
        for field in fields:
            values = [(x["Juris-Logic"]["results"][name][field], x["Juris-Logic"]["queries"])
                      for x in shards]
            if field == "total_candidate_probes":
                merged_results[name][field] = sum(v for v, _ in values)
            else:
                merged_results[name][field] = round(sum(v*n for v, n in values) / total, 4)
    probe_totals = {x["total_candidate_probes"] for x in merged_results.values()}
    if len(probe_totals) != 1:
        raise RuntimeError(f"Juris variants are not compute matched: {probe_totals}")
    bill_probes = {x["total_candidate_probes"] for x in bill["Bill-Contra"]["results"].values()}
    if len(bill_probes) != 1:
        raise RuntimeError(f"Bill variants are not compute matched: {bill_probes}")
    output = {"protocol": bill["protocol"], "Bill-Contra": bill["Bill-Contra"],
              "Juris-Logic": {"queries": total, "documents": first["documents"],
                               "data_audit": first["data_audit"], "results": merged_results}}
    json.dump(output, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
