"""Generate a deterministic three-view LLM paraphrase control."""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from aligned_eval_data import load_aligned


SYSTEM = ("You rewrite legal and regulatory statements without changing their meaning. "
          "Do not negate, weaken, strengthen, add, or remove any condition.")
USER = ("Return exactly three meaning-preserving paraphrases of the statement below as a JSON list "
        "of three strings, with no commentary.\n\nSTATEMENT:\n{query}")


def parse_views(text, original):
    match = re.search(r"\[[\s\S]*\]", text)
    try:
        values = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        values = []
    if not values:
        values = []
        for token in re.findall(r'"(?:\\.|[^"\\])*"', text):
            try:
                values.append(json.loads(token))
            except json.JSONDecodeError:
                pass
    clean = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, str) and value.strip() and value.strip() != original:
            clean.append(value.strip())
    return clean[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="multiquery_paraphrases.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(args.device)
    model.eval()

    existing = json.load(open(args.out, encoding="utf-8")) if args.resume and Path(args.out).exists() else {}
    payload = {"protocol": {"model": args.model, "decoding": "greedy",
                             "views_requested": 3, "prompt": USER, "system": SYSTEM}}
    for name, filename, kind in [("Bill-Contra", "Bill-Contra.csv", "bill"),
                                  ("Juris-Logic", "Juris-Logic.csv", "juris")]:
        _, records, audit = load_aligned(Path(args.data_dir) / filename, kind)
        old = {x["q"]: x for x in existing.get(name, {}).get("records", [])}
        generated_rows = dict(old)
        pending = [x for x in records if len(old.get(x["q"], {}).get("views", [])) < 3]
        for start in tqdm(range(0, len(pending), args.batch_size), desc=name):
            batch = pending[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": USER.format(query=x["q"])}],
                tokenize=False, add_generation_prompt=True) for x in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                               max_length=1536).to(args.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
            prompt_width = inputs["input_ids"].shape[1]
            for record, tokens in zip(batch, output):
                decoded = tokenizer.decode(tokens[prompt_width:], skip_special_tokens=True)
                generated_rows[record["q"]] = {"q": record["q"],
                                                "views": parse_views(decoded, record["q"]),
                                                "raw": decoded}
        rows = [generated_rows[x["q"]] for x in records]
        payload[name] = {"data_audit": audit, "records": rows,
                         "queries_with_3_views": sum(len(x["views"]) == 3 for x in rows)}

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
