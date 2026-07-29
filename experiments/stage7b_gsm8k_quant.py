# stage7b_gsm8k_quant.py
# Stage 7 addendum: the missing task-level datapoints for the packed-quant
# frontier. Stage 8 showed packed Kint4/Vint4 is near-lossless on WikiText
# KL/PPL at 3.2x, but stage 7 showed 2-bit KIVI-style quant halves GSM8K
# accuracy (17/100 vs 39/100) while latent-r16 + influence 5% preserves it
# (38/100 at 2.86x). This runs GSM8K on: packed int4/int4, packed int2/int2,
# and the asymmetric Klat16/Vint4 + K-influence residual hybrid.
# Same repaired harness as stage 7 (dense prefill, W=64 exact, old block
# coded once, 8-shot, greedy, 100 problems).
# K per-channel quant groups run along tokens; the trailing S_old % 32 tokens
# stay native (KIVI residual practice); V groups run along channels.

import json
import re
import torch
from datasets import load_dataset
from kv_common import load_model
from kv_hybrid import Hybrid, pack_quant, unpack_quant

torch.manual_seed(0)
tok, model = load_model()
hy = Hybrid(model)
H, D, P = hy.H, hy.D, hy.P
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

N_PROBLEMS = 100
N_SHOTS = 8
MAX_NEW = 256
W = 64
GROUP = 32

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


def quant_roundtrip(X, ti, bits):
    """X [H, S, D] fp32 -> packed round trip; K trailing tokens stay native."""
    S = X.shape[1]
    if ti == 0:                                  # K: per-channel, token groups
        nq = (S // GROUP) * GROUP
        packed, mn, sc, nb, meta = pack_quant(X[:, :nq], GROUP, 1, bits=bits)
        Xh = torch.cat([unpack_quant(packed, mn, sc, meta, GROUP, bits=bits),
                        X[:, nq:]], dim=1)
        nb += H * (S - nq) * D * 2
        return Xh, nb
    packed, mn, sc, nb, meta = pack_quant(X, GROUP, 2, bits=bits)  # V per-token
    return unpack_quant(packed, mn, sc, meta, GROUP, bits=bits), nb


def compress(base, spec, abar):
    out, total, S_old = [], 0, None
    for li, (K, V) in enumerate(base):
        S = K.shape[-2]
        S_old = S - W
        Kf, Vf = K[0, :, :S_old].float(), V[0, :, :S_old].float()
        if spec == "full":
            out.append((K, V))
            total += 2 * H * S_old * D * 2
            continue
        if spec in ("int4", "int2"):
            bits = int(spec[-1])
            Kh, bK = quant_roundtrip(Kf, 0, bits)
            Vh, bV = quant_roundtrip(Vf, 1, bits)
            total += bK + bV
        else:  # "klat16-vint4-infK5"
            Kh = hy.project_fixed(Kf, li, 0, 16)
            eK, _ = hy.residual(Kf - Kh, torch.zeros_like(Vf), li,
                                "influence", 0.05, abar)
            Kh = Kh + eK
            Vh, bV = quant_roundtrip(Vf, 1, 4)
            total += H * S_old * 16 * 2 + 0.05 * H * S_old * P * 8 + bV
        Kc = torch.cat([Kh.unsqueeze(0).to(K.dtype), K[:, :, S_old:]], dim=-2)
        Vc = torch.cat([Vh.unsqueeze(0).to(V.dtype), V[:, :, S_old:]], dim=-2)
        out.append((Kc, Vc))
    return out, total / (S_old * hy.L)


CONFIGS = ["full", "int4", "int2", "klat16-vint4-infK5"]
NEED_Q = True

problems = [(test[i]["question"], gold_of(test[i]["answer"]))
            for i in range(N_PROBLEMS)]
print(f"{N_PROBLEMS} problems, {N_SHOTS}-shot, W={W}, max_new={MAX_NEW}\n")

stats = {c: {"correct": 0, "unparsed": 0, "n_toks": [], "bpt": []}
         for c in CONFIGS}
for j, (q, gold) in enumerate(problems):
    ids = build_ids(q)
    base, q_list = hy.prefill(ids, capture_q=NEED_Q)
    abar = hy.attention_mass(base, q_list, W)
    for name in CONFIGS:
        tensors, bpt = compress(base, name, abar)
        cache = hy.build_cache(tensors)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=MAX_NEW,
                past_key_values=cache, use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0, ids.shape[1]:]
        del cache, tensors, out
        text = tok.decode(gen, skip_special_tokens=True)
        pred = extract(text)
        ok = pred is not None and abs(pred - gold) < 1e-6
        s = stats[name]
        s["correct"] += int(ok)
        s["unparsed"] += int(pred is None)
        s["n_toks"].append(len(gen))
        s["bpt"].append(bpt)
    del base, q_list, abar
    torch.cuda.empty_cache()
    if (j + 1) % 10 == 0:
        print(f"  [{j + 1:>3}/{N_PROBLEMS}] " + "  ".join(
            f"{c} {stats[c]['correct']}/{j + 1}" for c in CONFIGS))

results = {"meta": {"n_problems": N_PROBLEMS, "n_shots": N_SHOTS, "W": W,
                    "group": GROUP}}
print(f"\n{'config':>20} | {'acc':>7} | {'unparsed':>8} | {'new toks':>8} | "
      f"{'bytes/tok/layer':>15}")
for name in CONFIGS:
    s = stats[name]
    b = sum(s["bpt"]) / len(s["bpt"])
    results[name] = {
        "accuracy": s["correct"] / N_PROBLEMS, "correct": s["correct"],
        "n": N_PROBLEMS, "unparsed": s["unparsed"],
        "mean_new_tokens": sum(s["n_toks"]) / len(s["n_toks"]),
        "bytes_per_tok_layer": b, "compression": 2048 / b}
    print(f"{name:>20} | {s['correct']:>3}/{N_PROBLEMS} | {s['unparsed']:>8} | "
          f"{results[name]['mean_new_tokens']:>8.0f} | "
          f"{b:>8.0f} ({2048 / b:.2f}x)")

with open("results/stage7b_gsm8k_quant.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved: results/stage7b_gsm8k_quant.json")
