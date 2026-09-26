#!/usr/bin/env python3
"""
Honesty steering vectors, as a comparison to the TruthfulQA truthfulness vectors.

Method (Zou et al. 2023 "Representation Engineering"; also used by Goldowsky-Dill et al. 2025):
take TRUE factual statements and pair two prompts that differ ONLY in the instruction,

    user:  "Pretend you're an honest person making statements about the world."
    model: <first part of a true statement>

    user:  "Pretend you're a dishonest person making statements about the world."
    model: <the same first part of the same statement>

and read the residual stream at the last token. Because the assistant text is identical,
the difference isolates "intending to be honest vs. deceptive" rather than differences
in answer content (the confound in TruthfulQA contrast pairs, where the truthful answer is
often a hedge like "Nothing in particular happens").

Statements are truncated at several points (never the last few tokens), so the model is
mid-statement, i.e. still deciding what to say.

Vector = mean(honest - dishonest)  ->  positive coeff steers towards honesty, same sign
convention as the truthfulness vectors. Saved as 1-D CPU tensors, so they plug directly into
run_inference_steering.sh as the 6th argument.

Usage:
  python truthfulness/train_honesty_vectors.py                        # base gemma-2-9b-it
  python truthfulness/train_honesty_vectors.py \
      --model_name bcywinski/gemma-2-9b-it-user-female --tag honesty_female
"""
import argparse
import csv
import io
import json
import os
import random
import sys
import urllib.request

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FACTS_URL = (
    "https://raw.githubusercontent.com/andyzoujm/representation-engineering/"
    "main/data/facts/facts_true_false.csv"
)
FACTS_CACHE = os.path.join(REPO_ROOT, "truthfulness", "data", "facts_true_false.csv")

HONEST = "Pretend you're an honest person making statements about the world."
DISHONEST = "Pretend you're a dishonest person making statements about the world."


def load_true_facts():
    if not os.path.exists(FACTS_CACHE):
        os.makedirs(os.path.dirname(FACTS_CACHE), exist_ok=True)
        print(f"Downloading facts dataset -> {FACTS_CACHE}")
        with urllib.request.urlopen(FACTS_URL) as r:
            data = r.read().decode("utf-8")
        with open(FACTS_CACHE, "w", encoding="utf-8") as f:
            f.write(data)
    with open(FACTS_CACHE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["statement"].strip() for r in rows if r["label"].strip() == "1"]


def make_pairs(tokenizer, statements, max_prefixes=8, drop_last=5):
    """For each statement, build (honest_text, dishonest_text) pairs at several truncation points."""
    def prompt(instruction):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

    honest_prefix, dishonest_prefix = prompt(HONEST), prompt(DISHONEST)
    pairs = []
    for s in statements:
        ids = tokenizer.encode(s, add_special_tokens=False)
        last = max(1, len(ids) - drop_last)  # prefixes of length 1..last
        cuts = sorted(set(
            round(1 + i * (last - 1) / max(max_prefixes - 1, 1)) for i in range(max_prefixes)
        ))
        for k in cuts:
            partial = tokenizer.decode(ids[:k])
            pairs.append((honest_prefix + partial, dishonest_prefix + partial))
    return pairs


@torch.no_grad()
def last_token_activations(model, tokenizer, texts, layers, batch_size=8):
    """Residual stream after each layer at the final real token. Returns {layer: [n, d]} (float32, CPU)."""
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # so position -1 is the last real token for every row
    captured, handles = {}, []
    for L in layers:
        def hook(module, inp, out, L=L):
            captured[L] = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float().cpu()
        handles.append(model.model.layers[L].register_forward_hook(hook))

    out = {L: [] for L in layers}
    try:
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[i:i + batch_size],
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False,  # chat template already contains <bos>
            )
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
            for L in layers:
                out[L].append(captured[L])
    finally:
        for h in handles:
            h.remove()
        tokenizer.padding_side = old_side
    return {L: torch.cat(v) for L, v in out.items()}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20, 23, 26])
    ap.add_argument("--output_dir", default="truthfulness/vectors")
    ap.add_argument("--tag", default="honesty", help="Filename prefix: <tag>_l<layer>.pt")
    ap.add_argument("--train_fraction", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from sampling.utils import load_model_and_tokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    facts = load_true_facts()
    random.shuffle(facts)
    n_train = int(len(facts) * args.train_fraction)
    train_facts, test_facts = facts[:n_train], facts[n_train:]  # split by statement: no leakage
    print(f"{len(facts)} true facts -> {len(train_facts)} train / {len(test_facts)} held-out")

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    train_pairs = make_pairs(tokenizer, train_facts)
    test_pairs = make_pairs(tokenizer, test_facts)
    print(f"{len(train_pairs)} train pairs, {len(test_pairs)} held-out pairs")
    print("Example honest text:\n" + train_pairs[0][0])

    def acts(pairs):
        h = last_token_activations(model, tokenizer, [p[0] for p in pairs], args.layers, args.batch_size)
        d = last_token_activations(model, tokenizer, [p[1] for p in pairs], args.layers, args.batch_size)
        return h, d

    tr_h, tr_d = acts(train_pairs)
    te_h, te_d = acts(test_pairs)

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {"model_name": args.model_name, "n_train_pairs": len(train_pairs),
               "n_test_pairs": len(test_pairs), "layers": {}}
    print(f"\n{'layer':>5} {'norm':>9} {'held-out acc':>13} {'cos(truthful)':>14}")
    for L in args.layers:
        vec = (tr_h[L] - tr_d[L]).mean(0)
        unit = vec / vec.norm()
        acc = ((te_h[L] - te_d[L]) @ unit > 0).float().mean().item()

        cos = None
        tpath = os.path.join(args.output_dir, f"truthfulness_l{L}.pt")
        if os.path.exists(tpath):
            t = torch.load(tpath, map_location="cpu", weights_only=True).float()
            cos = torch.nn.functional.cosine_similarity(vec, t, dim=0).item()

        path = os.path.join(args.output_dir, f"{args.tag}_l{L}.pt")
        torch.save(vec, path)
        summary["layers"][L] = {"norm": vec.norm().item(), "heldout_pair_acc": acc,
                                "cos_with_truthfulness": cos, "path": path}
        cos_s = "   —   " if cos is None else f"{cos:+.3f}"
        print(f"{L:>5} {vec.norm().item():>9.2f} {100 * acc:>12.1f}% {cos_s:>14}")

    with open(os.path.join(args.output_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved vectors + {args.tag}_summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()
