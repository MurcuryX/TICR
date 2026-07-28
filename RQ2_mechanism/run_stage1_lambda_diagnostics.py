"""Canonical Stage-1 diagnostics for inversion composition and lambda scans."""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from aligned_eval_data import load_aligned

INS = "Represent this sentence for searching relevant passages: "
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

def norm(x):
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-12)

def summarize(rows):
    return {k: round(float(np.mean(v)) * (100 if k in {"hit20", "mrr", "pair_win", "overlap20"} else 1), 4)
            for k, v in rows.items()}

def run(model, docs, records, device):
    d = model.encode(docs, batch_size=64, convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=True, device=device)
    names = ["inverse_only", "q_plus_inverse", "inverse_prefix_q"] + [f"vec_lambda_{x:g}" for x in LAMBDAS]
    allrows = {n: {k: [] for k in ["hit20", "mrr", "pair_margin", "pair_win", "overlap20"]} for n in names}
    for rec in tqdm(records, desc="lambda diagnostics"):
        q = model.encode([INS + rec["q"]], convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False, device=device)[0]
        inv = model.encode([INS + x for x in rec["inverse_atoms"]], convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=False, device=device)
        pos = model.encode([INS + x for x in rec["positive_atoms"]], convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=False, device=device) if rec["positive_atoms"] else None
        probes = {
            "inverse_only": inv,
            "q_plus_inverse": model.encode([INS + rec["q"] + " " + x for x in rec["inverse_atoms"]], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=device),
            "inverse_prefix_q": model.encode([INS + x + " " + rec["q"] for x in rec["inverse_atoms"]], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=device),
        }
        for lam in LAMBDAS:
            probes[f"vec_lambda_{lam:g}"] = norm(q.unsqueeze(0) + lam * inv)
        original = torch.topk(q @ d.T, 20).indices.detach().cpu().numpy()
        for name, p in probes.items():
            scores = torch.max(p @ d.T, dim=0).values
            order = torch.argsort(scores, descending=True).detach().cpu().numpy()
            ranks = np.where(np.isin(order, list(rec["gt"]))) [0]
            allrows[name]["hit20"].append(float(len(ranks) and ranks.min() < 20))
            allrows[name]["mrr"].append(float(1.0 / (ranks.min() + 1)) if len(ranks) else 0.0)
            allrows[name]["overlap20"].append(float(len(set(order[:20]) & set(original[:20])) / 20.0))
            if rec["hard_entail"]:
                margin = max(scores[list(rec["gt"])]) - max(scores[list(rec["hard_entail"])])
                allrows[name]["pair_margin"].append(float(margin))
                allrows[name]["pair_win"].append(float(margin > 0))
    return {name: summarize(rows) for name, rows in allrows.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="stage1_lambda_diagnostics.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model = SentenceTransformer(os.environ.get("TICR_MODEL_PATH", "BAAI/bge-base-en-v1.5"), device=args.device)
    payload = {"protocol": {"lambdas": LAMBDAS, "population": "canonical paired subset", "metrics": "Hit@20, MRR, pair margin, pair win, original-top20 overlap"}}
    for name, fn, kind in [("Bill-Contra", "Bill-Contra.csv", "bill"), ("Juris-Logic", "Juris-Logic.csv", "juris")]:
        docs, records, audit = load_aligned(Path(args.data_dir) / fn, kind)
        payload[name] = {"queries": len(records), "data_audit": audit, "results": run(model, docs, records, args.device)}
    Path(args.out).write_text(json.dumps(payload, indent=2))

if __name__ == "__main__": main()
