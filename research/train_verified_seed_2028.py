import hashlib
import json
import random
import re
import time
from decimal import Decimal
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
SEED = 2028

EPOCHS = 2
LEARNING_RATE = 0.0001
GRAD_ACCUM = 4
MAX_SEQUENCE_LENGTH = 1024

SOURCE = Path("results/verified_training.jsonl")
OUTPUT = Path("lora_verified_seed_2028")
TEMP_OUTPUT = Path("lora_verified_seed_2028.part")
METADATA = Path("results/lora_verified_seed_2028_metrics.json")

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available. Training stopped.")

if OUTPUT.exists() or TEMP_OUTPUT.exists():
    raise FileExistsError(
        "Adapter output already exists. "
        "Stopping to avoid overwriting previous work."
    )

if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

print("Reading verified training dataset...", flush=True)

raw = [
    json.loads(line)
    for line in SOURCE.read_text().splitlines()
    if line.strip()
]

assert len(raw) == 43, (
    f"Expected 43 training examples, found {len(raw)}"
)

indices = [x["dataset_index"] for x in raw]

assert len(set(indices)) == len(indices), (
    "Duplicate training question indices"
)
assert all(222 <= idx < 272 for idx in indices), (
    "Training data outside the approved partition"
)

data = load_dataset(
    "openai/gsm8k",
    "main",
    split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=2026)

for x in raw:
    assert x["question"] == data[x["dataset_index"]]["question"], (
        "Dataset question mismatch"
    )

    match = re.search(
        r"(?m)^####[ \t]*(-?[\d,]+(?:\.\d+)?)[ \t]*$",
        x["solution"].strip()
    )
    assert match, (
        f"Missing final answer for index {x['dataset_index']}"
    )

    number = Decimal(match.group(1).replace(",", ""))
    assert number == Decimal(x["answer"]), (
        f"Unverified answer at index {x['dataset_index']}"
    )

print("All 43 records passed structural checks.", flush=True)

prefix = (
    "Solve the math problems step by step. "
    "Finish each solution with #### followed by the numeric answer.\n\n"
)

for ex in data.select(range(2)):
    solution = re.sub(r"<<[^<>]*>>", "", ex["answer"])
    prefix += (
        f"Question: {ex['question']}\n"
        f"Solution: {solution}\n\n"
    )

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer has no EOS token.")

training_examples = []

for x in raw:
    prompt = (
        prefix
        + f"Question: {x['question']}\nSolution:"
    )

    target = " " + x["solution"].strip() + "\n"

    prompt_ids = tokenizer.encode(
        prompt, add_special_tokens=False
    )
    target_ids = tokenizer.encode(
        target, add_special_tokens=False
    )

    input_ids = (
        prompt_ids
        + target_ids
        + [tokenizer.eos_token_id]
    )

    labels = (
        [-100] * len(prompt_ids)
        + target_ids
        + [tokenizer.eos_token_id]
    )

    assert len(input_ids) == len(labels)
    assert any(label != -100 for label in labels)

    if len(input_ids) > MAX_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Example {x['dataset_index']} contains "
            f"{len(input_ids)} tokens, exceeding "
            f"the {MAX_SEQUENCE_LENGTH}-token limit. "
            "Stopping rather than truncating a solution."
        )

    training_examples.append({
        "input_ids": input_ids,
        "labels": labels
    })

print(
    "Maximum training sequence length:",
    max(len(x["input_ids"]) for x in training_examples),
    flush=True
)

print("Loading original Qwen model...", flush=True)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda:0"
)

base.config.use_cache = False
base.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base, config)
model.train()
model.print_trainable_parameters()

trainable = [
    p for p in model.parameters() if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable,
    lr=LEARNING_RATE,
    weight_decay=0.0
)

scaler = torch.amp.GradScaler("cuda")

optimizer.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()

started = time.perf_counter()
epoch_losses = []
updates = 0

print("\n=== STARTING VERIFIED SELF-TRAINING ===", flush=True)

for epoch in range(1, EPOCHS + 1):
    order = list(range(len(training_examples)))
    random.Random(SEED + epoch).shuffle(order)

    losses = []

    for j, idx in enumerate(order):
        ex = training_examples[idx]

        inputs = torch.tensor(
            [ex["input_ids"]],
            dtype=torch.long,
            device="cuda"
        )

        labels = torch.tensor(
            [ex["labels"]],
            dtype=torch.long,
            device="cuda"
        )

        group_start = (j // GRAD_ACCUM) * GRAD_ACCUM
        group_size = min(
            GRAD_ACCUM,
            len(order) - group_start
        )

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                labels=labels
            )
            loss = outputs.loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}, "
                f"example {j + 1}."
            )

        losses.append(loss.item())
        scaler.scale(loss / group_size).backward()

        if (
            (j + 1) % GRAD_ACCUM == 0
            or (j + 1) == len(order)
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

        del outputs, loss, inputs, labels

        if (j + 1) % 10 == 0 or j + 1 == len(order):
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Examples {j + 1}/{len(order)} | "
                f"Mean loss {sum(losses)/len(losses):.4f} | "
                f"Optimizer updates {updates}",
                flush=True
            )

    average = sum(losses) / len(losses)
    epoch_losses.append(average)
    print(
        f"Epoch {epoch} finished. Mean loss: {average:.4f}",
        flush=True
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - started
peak = torch.cuda.max_memory_reserved() / 1024**3

print("Saving LoRA adapter...", flush=True)
model.save_pretrained(str(TEMP_OUTPUT))
TEMP_OUTPUT.rename(OUTPUT)

metrics = {
    "model": MODEL_ID,
    "seed": SEED,
    "training_examples": len(raw),
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "gradient_accumulation": GRAD_ACCUM,
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "lora_rank": 4,
    "lora_alpha": 8,
    "target_modules": ["q_proj", "v_proj"],
    "optimizer_updates": updates,
    "epoch_mean_losses": epoch_losses,
    "elapsed_minutes": round(elapsed / 60, 2),
    "peak_pytorch_vram_gb": round(peak, 2),
    "source_sha256": hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest(),
    "script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "adapter_path": str(OUTPUT)
}

METADATA.write_text(json.dumps(metrics, indent=2))

print("\n=== SELF-TRAINING PILOT COMPLETE ===")
print("Examples:", len(raw))
print("Epochs:", EPOCHS)
print("Optimizer updates:", updates)
print("Mean loss by epoch:", epoch_losses)
print("Elapsed:", round(elapsed / 60, 2), "minutes")
print("Peak PyTorch VRAM:", round(peak, 2), "GB")
print("Adapter saved:", OUTPUT)
print("Metrics saved:", METADATA)
