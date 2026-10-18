import json
import random
import re
import time
from decimal import Decimal
from pathlib import Path

import torch
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL = "Qwen/Qwen3-1.7B-Base"
DATASET = "apple/GSM-Symbolic"
BATCH_SIZE = 8
MAX_TOKENS = 384
SEED = 2026

Path("results").mkdir(exist_ok=True)
torch.manual_seed(SEED)

ANSWER = re.compile(r"####[ \t]*(-?[\d,]+(?:\.\d+)?)")
STOP = re.compile(r"####[ \t]*-?[\d,]+(?:\.\d+)?[ \t]*\r?\n")

def parse(text):
    m = ANSWER.search(text)
    return str(Decimal(m.group(1).replace(",", ""))) if m else None

class StopAnswers(StoppingCriteria):
    def __init__(self, tok, prompt_len, count):
        self.tok = tok
        self.prompt_len = prompt_len
        self.lengths = [None] * count

    def __call__(self, input_ids, scores, **kwargs):
        n = input_ids.shape[1] - self.prompt_len
        for i in range(len(self.lengths)):
            if self.lengths[i] is not None:
                continue
            text = self.tok.decode(
                input_ids[i, self.prompt_len:],
                skip_special_tokens=True
            )
            if STOP.search(text):
                self.lengths[i] = n
        return torch.tensor(
            [x is not None for x in self.lengths],
            device=input_ids.device,
            dtype=torch.bool
        )

print("Fetching pinned dataset version...", flush=True)
revision = HfApi().dataset_info(DATASET).sha
print("Dataset revision:", revision)

datasets = {}
for variant in ["p1", "p2"]:
    print("Loading:", variant, flush=True)
    datasets[variant] = load_dataset(
        DATASET, name=variant, split="test", revision=revision
    )
    print("Rows:", len(datasets[variant]))
    print("Columns:", datasets[variant].column_names)

required = {"id", "instance", "question", "answer"}
for variant, ds in datasets.items():
    assert required.issubset(ds.column_names), (
        f"Unexpected {variant} schema"
    )


# Template IDs are specific to each variant.
# Choose 20 templates independently for P1 and P2.
# Track the underlying GSM8K original IDs to protect
# related problem families from future evaluation leakage.

dev_ids_by_variant = {}
original_by_template = {}

for variant, ds in datasets.items():
    mapping = {}

    for row in ds:
        template_id = row["id"]
        original_id = row["original_id"]

        if template_id in mapping:
            assert mapping[template_id] == original_id, (
                f"Inconsistent original ID in {variant}"
            )
        else:
            mapping[template_id] = original_id

    original_by_template[variant] = mapping

    ids = sorted(mapping)
    assert len(ids) >= 20, "Too few templates"

    selected = sorted(
        random.Random(f"{SEED}-{variant}").sample(ids, 20)
    )

    dev_ids_by_variant[variant] = selected

    print(
        f"{variant.upper()}: {len(ids)} templates; "
        f"selected {len(selected)} for development"
    )

all_originals = set()
exposed_originals = set()

for variant, mapping in original_by_template.items():
    all_originals.update(mapping.values())

    for template_id in dev_ids_by_variant[variant]:
        exposed_originals.add(mapping[template_id])

reserved_originals = all_originals - exposed_originals

split_info = {
    "dataset": DATASET,
    "revision": revision,
    "development_templates": dev_ids_by_variant,
    "development_original_ids": sorted(exposed_originals),
    "reserved_original_ids": sorted(reserved_originals),
    "instances_per_template": [0, 1, 2, 3, 4],
    "seed": SEED,
    "paired_variants": False
}

split_path = Path("results/gsm_symbolic_pilot_split_v2.json")

if split_path.exists():
    assert json.loads(split_path.read_text()) == split_info, (
        "Existing split differs. Refusing to overwrite."
    )
else:
    split_path.write_text(json.dumps(split_info, indent=2))

print("Development question families:", len(exposed_originals))
print("Reserved question families:", len(reserved_originals))

print("Preparing identical GSM8K few-shot prompt...", flush=True)
gsm = load_dataset(
    "openai/gsm8k", "main", split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=SEED)

prefix = (
    "Solve the math problems step by step. "
    "Finish each solution with #### followed by the numeric answer.\n\n"
)
for ex in gsm.select(range(2)):
    solution = re.sub(r"<<[^<>]*>>", "", ex["answer"])
    prefix += f"Question: {ex['question']}\nSolution: {solution}\n\n"

print("Loading original Qwen model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cuda:0"
)
model = PeftModel.from_pretrained(
    model, "lora_verified_seed_2028"
)
model.eval()

for variant, ds in datasets.items():
    print(f"\n=== EVALUATING {variant.upper()} ===", flush=True)

    rows = [
        row for row in ds
        if row["id"] in dev_ids_by_variant[variant] and row["instance"] in range(5)
    ]
    rows.sort(key=lambda row: (row["id"], row["instance"]))
    assert len(rows) == 100, (
        f"Expected 100 {variant} questions, got {len(rows)}"
    )

    prompts = [
        prefix + f"Question: {r['question']}\nSolution:"
        for r in rows
    ]
    gold = [parse(r["answer"]) for r in rows]
    assert all(x is not None for x in gold), "Unparseable reference"

    correct = 0
    parsed = 0
    records = []
    start_time = time.perf_counter()

    for start in range(0, 100, BATCH_SIZE):
        end = min(start + BATCH_SIZE, 100)
        inputs = tok(
            prompts[start:end],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        prompt_len = inputs["input_ids"].shape[1]
        stopper = StopAnswers(tok, prompt_len, end - start)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=tok.pad_token_id
            )

        generated = output[:, prompt_len:]

        for j in range(end - start):
            idx = start + j
            token_ids = generated[j].tolist()

            if stopper.lengths[j] is not None:
                count = stopper.lengths[j]
            elif tok.eos_token_id in token_ids:
                count = token_ids.index(tok.eos_token_id) + 1
            else:
                count = len(token_ids)

            response = tok.decode(
                token_ids[:count], skip_special_tokens=True
            )
            prediction = parse(response)
            ok = prediction == gold[idx]

            correct += int(ok)
            parsed += int(prediction is not None)

            records.append({
                "variant": variant,
                "id": rows[idx]["id"],
                "instance": rows[idx]["instance"],
                "question": rows[idx]["question"],
                "gold": gold[idx],
                "prediction": prediction,
                "correct": ok,
                "tokens": count,
                "response": response
            })

        print(
            f"{variant.upper()} {end}/100 | "
            f"Correct: {correct} | "
            f"Accuracy: {100 * correct / end:.1f}%",
            flush=True
        )
        del inputs, output, generated

    elapsed = time.perf_counter() - start_time
    destination = Path(
        f"results/gsm_symbolic_{variant}_dev_pilot_v2_lora_seed_2028.jsonl"
    )
    temp = destination.with_suffix(".tmp")

    with temp.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    temp.replace(destination)

    print(f"\n{variant.upper()} FINAL: {correct}/100")
    print(f"Accuracy: {correct}%")
    print(f"Answers parsed: {parsed}/100")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")
    print("Saved:", destination)

print("\n=== GSM-SYMBOLIC PILOT COMPLETE ===")
print(
    "Peak PyTorch VRAM:",
    round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    "GB"
)
