"""Canonical query-level loading for TICR control experiments."""
import ast
import csv
import re
import unicodedata
from collections import OrderedDict


def _parse(value):
    try:
        parsed = ast.literal_eval(value) if value and value.strip() else []
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def canonical_text(value):
    """Canonicalize text before query grouping and document labeling."""
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def _flatten(items):
    out = []
    for item in items or []:
        if isinstance(item, list):
            out.extend(_flatten(item))
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _inversions(items, r=3):
    if not items:
        return []
    if isinstance(items[0], str):
        return [x.strip() for x in items[:r] if isinstance(x, str) and x.strip()]
    return [x.strip() for group in items if isinstance(group, list)
            for x in group[:r] if isinstance(x, str) and x.strip()]


def _extend_unique(target, values):
    present = set(target)
    for value in values:
        if value not in present:
            target.append(value)
            present.add(value)


def load_aligned(path, kind, paired_only=True):
    """Group duplicate rows by query and select either full or paired data.

    Documents and relevance labels are unioned across all rows belonging to a
    query. Positive atoms and cached inversions are deduplicated in source order.
    With ``paired_only=True`` every returned query has both a nonempty
    ground-truth set and inversions. With ``paired_only=False`` the returned
    population contains every clean valid query; callers must use the explicit
    no-inversion fallback defined by their protocol.
    """
    docs, seen = [], {}
    groups = OrderedDict()
    raw_rows = 0

    def add_doc(text):
        text = canonical_text(text)
        if not text:
            return None
        if text not in seen:
            seen[text] = len(docs)
            docs.append(text)
        return seen[text]

    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            if kind == "bill":
                q = canonical_text(row.get("Original_Text", ""))
                conflicts = [row.get("Conflict1 (Contradiction)", ""), row.get("Conflict2 (Contradiction)", "")]
                entails = [row.get("Paraphrase_Structure_1 (Entailment)", ""), row.get("Paraphrase_Structure_2 (Entailment)", "")]
                positive = _flatten(_parse(row.get("Original_Text_atoms", "")))
                inverse = _inversions(_parse(row.get("Neg_Original_Text_atoms", "")))
            else:
                q = canonical_text(row.get("query", ""))
                conflicts = [row.get("conflict", "")]
                entails = [row.get("entail", "")]
                positive = _flatten(_parse(row.get("query_atoms", "")))
                inverse = _inversions(_parse(row.get("negs_sourse_atoms", "")))
            if not q:
                continue
            group = groups.setdefault(q, {"q": q, "positive_atoms": [], "inverse_atoms": [],
                                          "gt": set(), "hard_entail": set()})
            for text in conflicts:
                idx = add_doc(text)
                if idx is not None:
                    group["gt"].add(idx)
            for text in entails:
                idx = add_doc(text)
                if idx is not None:
                    group["hard_entail"].add(idx)
            _extend_unique(group["positive_atoms"], positive)
            _extend_unique(group["inverse_atoms"], inverse)

    valid_gt = [g for g in groups.values() if g["gt"]]
    # A document cannot be both a contradiction and an entailment for the same
    # query.  Exclude such ambiguous query groups from paired diagnostics and
    # report them explicitly instead of silently choosing one label.
    collisions = [g for g in valid_gt if g["gt"] & g["hard_entail"]]
    clean_gt = [g for g in valid_gt if not (g["gt"] & g["hard_entail"])]
    paired = [g for g in clean_gt if g["inverse_atoms"]]
    selected = paired if paired_only else clean_gt
    metadata = {
        "raw_rows": raw_rows,
        "unique_queries": len(groups),
        "excluded_missing_ground_truth": len(groups) - len(valid_gt),
        "excluded_missing_inversion": len(clean_gt) - len(paired),
        "excluded_label_collision": len(collisions),
        "valid_query_population": len(valid_gt),
        "clean_valid_query_population": len(clean_gt),
        "paired_query_population": len(paired),
        "evaluated_queries": len(selected),
        "population_mode": "paired" if paired_only else "full_clean_valid",
            "aggregation": (
                "NFKC + whitespace canonical query; union labels/atoms/inversions; "
                + ("paired clean-valid subset" if paired_only else "full clean-valid population")
            ),
    }
    return docs, selected, metadata
