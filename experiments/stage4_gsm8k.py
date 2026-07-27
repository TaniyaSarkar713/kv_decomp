# stage4_gsm8k.py
# Stage 4 downstream task: GSM8K reasoning accuracy with a compressed KV cache.
# KL on one next token underestimates reasoning damage (handout Part VI lesson 4,
# GEAR): chained generation compounds per-step error. Protocol: 4-shot CoT
# prompts (exemplars from the train split, calculator annotations stripped),
# greedy decoding, answer = last '#### <number>'. The cache is compressed once
# after prefill (the ~700-token few-shot prompt); the model's own generated
# reasoning tokens append exactly, matching Stages 2-4.

import json
import re
import torch
from datasets import load_dataset
from transformers import DynamicCache
from kv_common import load_model, LowRankCache, KIVICache

torch.manual_seed(0)
tok, model = load_model()
cfg = model.config
print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

N_PROBLEMS = 50
N_SHOTS = 4
MAX_NEW = 256

gsm = load_dataset("openai/gsm8k", "main")
train, test = gsm["train"], gsm["test"]


def clean(ans):  # strip calculator annotations like <<48/2=24>>
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


CONFIGS = [
    ("full", lambda: DynamicCache(config=cfg)),
    ("kv-r48", lambda: LowRankCache(rank=48, config=cfg)),
    ("kv-r32", lambda: LowRankCache(rank=32, config=cfg)),
    ("kv-r16", lambda: LowRankCache(rank=16, config=cfg)),
    ("v-only-r8", lambda: LowRankCache(rank=8, compress_keys=False, config=cfg)),
    ("kivi-2bit", lambda: KIVICache(group=32, residual=32, bits=2, config=cfg)),
]

problems = [(test[i]["question"], gold_of(test[i]["answer"])) for i in range(N_PROBLEMS)]
sample_len = build_ids(problems[0][0]).shape[1]
print(f"{N_PROBLEMS} problems, {N_SHOTS}-shot, prefill ~{sample_len} tokens, "
      f"max_new_tokens={MAX_NEW}\n")

results = {}
for name, ctor in CONFIGS:
    correct, n_toks, unparsed, records = 0, [], 0, []
    for j, (q, gold) in enumerate(problems):
        ids = build_ids(q)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=MAX_NEW,
                past_key_values=ctor(), use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        pred = extract(text)
        ok = pred is not None and abs(pred - gold) < 1e-6
        correct += int(ok)
        unparsed += int(pred is None)
        n_toks.append(len(gen))
        records.append({"i": j, "pred": pred, "gold": gold, "correct": ok,
                        "new_tokens": len(gen)})
        if (j + 1) % 10 == 0:
            print(f"  [{name}] {j + 1}/{N_PROBLEMS}  acc so far {correct}/{j + 1}")
    acc = correct / N_PROBLEMS
    results[name] = {
        "accuracy": acc, "correct": correct, "n": N_PROBLEMS,
        "unparsed": unparsed, "mean_new_tokens": sum(n_toks) / len(n_toks),
        "records": records,
    }
    print(f"{name:>10}: accuracy {correct}/{N_PROBLEMS} = {acc:.1%}, "
          f"mean new tokens {sum(n_toks)/len(n_toks):.0f}, unparsed {unparsed}")
    with open("results/gsm8k_partial.json", "w") as f:
        json.dump(results, f, indent=2)

with open("results/stage4_gsm8k_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved: results/stage4_gsm8k_results.json")
