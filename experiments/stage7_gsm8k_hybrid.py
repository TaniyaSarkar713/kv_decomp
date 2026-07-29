# stage7_gsm8k_hybrid.py
# Stage 7: repaired downstream harness + the hybrid old-cache on GSM8K.
# (Experiment 0 completion + Experiment 3's task-level question from
# RESULTS_AND_NEXT_EXPERIMENTS.md.)
#
# The stage-4 GSM8K run compressed the cache DURING the multi-token prefill
# (via a cache-update hook), altering prompt self-attention, and had no exact
# hot window; joint rank-48 collapsed 38% -> 0%. This stage repairs the
# protocol using the stage-6 path:
#   dense prefill -> compress ONLY the aged old block once (last W tokens
#   exact) -> greedy generation appends exact tokens.
#
# Configs: full cache; a copy-control (identity re-build of the cache) that
# MUST reproduce full's generations token-for-token (Exp-0 gate -- abort
# otherwise); fixed-basis latent r=16/32; latent r=16 + SAGE influence
# residual at 2%/5% of RoPE-pair groups; latent r=16 + energy residual at 5%
# (task-level selector comparison); KIVI-style 2-bit on the old block.
# 8-shot CoT prompts (~1.3K tokens) so the old block dominates the prompt.
# Calibration (fixed bases, query quantiles) comes from stage 6 -- WikiText
# train -- so this also tests domain transfer of the offline basis.

import json
import re
import torch
from datasets import load_dataset
from kv_common import load_model
from kv_hybrid import Hybrid

torch.manual_seed(0)
tok, model = load_model()
hy = Hybrid(model)
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

N_PROBLEMS = 100
N_SHOTS = 8
MAX_NEW = 256
W = 64

gsm = load_dataset("openai/gsm8k", "main")
train, test = gsm["train"], gsm["test"]


def clean(ans):
    return re.sub(r"<<[^>]*>>", "", ans)


shots = "\n\n".join(
    f"Question: {train[i]['question']}\nAnswer: {clean(train[i]['answer'])}"
    for i in range(N_SHOTS)
)
INSTR = ("Solve the math problem. Reason step by step like the examples, and end "
         "your answer with '#### <final number>'.\n\n")


def build_ids(question):
    user = INSTR + shots + f"\n\nQuestion: {question}\nAnswer:"
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)


def gold_of(ans):
    return float(ans.split("####")[-1].strip().replace(",", "").replace("$", ""))


def extract(text):
    m = re.findall(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


# name -> (spec for Hybrid.compress, needs abar)
CONFIGS = [
    ("full", {"mode": "copy"}, False),
    ("copy-control", {"mode": "copy"}, False),
    ("latent-r16", {"mode": "latent", "r": 16}, False),
    ("latent-r16+inf2", {"mode": "latent", "r": 16,
                         "residual": ("influence", 0.02)}, True),
    ("latent-r16+inf5", {"mode": "latent", "r": 16,
                         "residual": ("influence", 0.05)}, True),
    ("latent-r16+egy5", {"mode": "latent", "r": 16,
                         "residual": ("energy", 0.05)}, False),
    ("latent-r32", {"mode": "latent", "r": 32}, False),
    ("kivi-2bit-old", {"mode": "kivi", "bits": 2, "group": 32}, False),
]
NEED_Q = any(nq for _, _, nq in CONFIGS)

problems = [(test[i]["question"], gold_of(test[i]["answer"]))
            for i in range(N_PROBLEMS)]
sample_len = build_ids(problems[0][0]).shape[1]
print(f"{N_PROBLEMS} problems, {N_SHOTS}-shot, prefill ~{sample_len} tokens "
      f"(old block ~{sample_len - 1 - W}), W={W}, max_new={MAX_NEW}\n")

stats = {name: {"correct": 0, "unparsed": 0, "n_toks": [], "match_full": 0,
                "records": []} for name, _, _ in CONFIGS}
copy_mismatch = 0

for j, (q, gold) in enumerate(problems):
    ids = build_ids(q)
    base, q_list = hy.prefill(ids, capture_q=NEED_Q)
    abar = hy.attention_mass(base, q_list, W) if NEED_Q else None
    full_gen = None
    for name, spec, _ in CONFIGS:
        cache = hy.build_cache(hy.compress(base, spec, W, abar=abar))
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=MAX_NEW,
                past_key_values=cache, use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0, ids.shape[1]:].tolist()
        del cache, out
        if name == "full":
            full_gen = gen
        if name == "copy-control" and gen != full_gen:
            copy_mismatch += 1
        text = tok.decode(torch.tensor(gen), skip_special_tokens=True)
        pred = extract(text)
        ok = pred is not None and abs(pred - gold) < 1e-6
        s = stats[name]
        s["correct"] += int(ok)
        s["unparsed"] += int(pred is None)
        s["n_toks"].append(len(gen))
        s["match_full"] += int(gen == full_gen)
        s["records"].append({"i": j, "pred": pred, "gold": gold, "correct": ok,
                             "new_tokens": len(gen)})
    del base, q_list, abar
    torch.cuda.empty_cache()

    if j == 4:  # Exp-0 gate, checked early: identity re-build must be exact
        assert copy_mismatch == 0, (
            f"copy-control diverged from full on {copy_mismatch}/5 problems -- "
            "cache re-build path is broken; later comparisons uninterpretable")
        print("  [gate] copy-control == full on first 5 problems\n")
    if (j + 1) % 10 == 0:
        line = f"  [{j + 1:>3}/{N_PROBLEMS}] " + "  ".join(
            f"{name} {s['correct']}/{j + 1}"
            for name, s in list(stats.items())[:6])
        print(line)

S_old = sample_len - 1 - W
results = {"meta": {"n_problems": N_PROBLEMS, "n_shots": N_SHOTS, "W": W,
                    "max_new": MAX_NEW, "prefill_tokens": sample_len,
                    "copy_mismatch": copy_mismatch}}
print(f"\n{'config':>16} | {'acc':>7} | {'==full':>6} | {'unparsed':>8} | "
      f"{'new toks':>8} | {'old-block bytes/tok/layer':>25}")
for name, spec, _ in CONFIGS:
    s = stats[name]
    acc = s["correct"] / N_PROBLEMS
    bpt = hy.old_block_bytes(spec, S_old)
    results[name] = {
        "accuracy": acc, "correct": s["correct"], "n": N_PROBLEMS,
        "unparsed": s["unparsed"],
        "mean_new_tokens": sum(s["n_toks"]) / len(s["n_toks"]),
        "match_full_frac": s["match_full"] / N_PROBLEMS,
        "old_block_bytes_per_tok_layer": bpt,
        "old_block_compression": (2 * hy.H * hy.D * 2) / bpt,
        "records": s["records"],
    }
    print(f"{name:>16} | {s['correct']:>3}/{N_PROBLEMS} | "
          f"{s['match_full']:>6} | {s['unparsed']:>8} | "
          f"{results[name]['mean_new_tokens']:>8.0f} | "
          f"{bpt:>10.0f} ({results[name]['old_block_compression']:.2f}x)")

assert copy_mismatch == 0, f"copy-control diverged on {copy_mismatch} problems"
print("\n[gate] copy-control reproduced full-cache generations exactly "
      f"on all {N_PROBLEMS} problems")

with open("results/stage7_gsm8k_hybrid.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved: results/stage7_gsm8k_hybrid.json")
