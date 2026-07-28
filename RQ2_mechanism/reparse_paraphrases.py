"""Reparse saved raw generations after parser improvements, without regeneration."""
import argparse
import json

from generate_multiquery_paraphrases import parse_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    data = json.load(open(args.path, encoding="utf-8"))
    for name in ["Bill-Contra", "Juris-Logic"]:
        for row in data[name]["records"]:
            row["views"] = parse_views(row["raw"], row["q"])
        data[name]["queries_with_3_views"] = sum(len(x["views"]) == 3 for x in data[name]["records"])
    with open(args.path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
