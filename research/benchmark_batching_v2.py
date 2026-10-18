import json
import re
import time
from decimal import Decimal
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
SEED = 2026
MAX_NEW_TOKENS = 256
BATCH_SIZES = [1, 8, 16, 32, 64]

ANSWER = re.compile(r"####[ \t]*(-?[\d,]+(?:\.\d+)?)")
STOP = re.compile(r"####[ \t]*-?[\d,]+(?:\.\d+)?[ \t]*\r?\n")

Path("results").mkdir(exist_ok=True)
torch.manual_seed(SEED)

def extract_answer(text):
    match = ANSWER.search(text)
    if not match:
        return None
    return str(Decimal(match.group(1).replace(",", "")))

class StopPerSequence(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length, batch_size):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_lengths = [None] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        length = input_ids.shape[1] - self.prompt_length

        for i in range(len(self.stop_lengths)):
            if self.stop_lengths[i] is not None:
                continue

            generated = self.tokenizer.decode(
                input_ids[i, self.prompt_length:],
                skip_special_tokens=True
            )
            if STOP.search(generated):
                self.stop_lengths[i] = length

        return torch.tensor(
            [x is not None for x in self.stop_lengths],
            dtype=torch.bool,
            device=input_ids.device
        )

print("Loading GSM8K training-pool questions...", flush=True)

data = load_dataset(
    "openai/gsm8k",
    "main",
    split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=SEED)

fewshot = data.select(range(2))
questions = data.select(range(272, 336))

prefix = (
    "Solve the math problems step by step. "
    "Finish each solution with #### followed by the numeric answer.\n\n"
)

for ex in fewshot:
    solution = re.sub(r"<<[^<>]*>>", "", ex["answer"])
    prefix += f"Question: {ex['question']}\nSolution: {solution}\n\n"

prompts = [
    prefix + f"Question: {ex['question']}\nSolution:"
    for ex in questions
]

gold = [extract_answer(ex["answer"]) for ex in questions]

print("Loading Qwen model...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda:0"
)
model.eval()

# Short GPU warmup
warmup = tokenizer(prompts[0], return_tensors="pt").to("cuda")
with torch.inference_mode():
    model.generate(
        **warmup,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
torch.cuda.synchronize()
del warmup

baseline_predictions = None
results = []

for batch_size in BATCH_SIZES:
    print(f"\n=== TESTING BATCH SIZE {batch_size} ===", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions = []
    real_tokens = 0
    padded_slots = 0

    started = time.perf_counter()

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            prompt_length = inputs["input_ids"].shape[1]

            stopper = StopPerSequence(
                tokenizer, prompt_length, len(batch)
            )

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([stopper]),
                    pad_token_id=tokenizer.pad_token_id
                )

            generated = outputs[:, prompt_length:]
            padded_slots += generated.numel()

            for i in range(len(batch)):
                token_ids = generated[i].tolist()
                count = len(token_ids)

                if stopper.stop_lengths[i] is not None:
                    count = min(count, stopper.stop_lengths[i])
                elif tokenizer.eos_token_id in token_ids:
                    count = token_ids.index(tokenizer.eos_token_id) + 1

                real_tokens += count

                response = tokenizer.decode(
                    token_ids[:count],
                    skip_special_tokens=True
                )
                predictions.append(extract_answer(response))

            del inputs, outputs, generated

            if batch_size == 1 and (start + 1) % 16 == 0:
                print(f"  Completed {start + 1}/64", flush=True)

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_reserved() / 1024**3

        correct = sum(
            pred == target for pred, target in zip(predictions, gold)
        )

        if batch_size == 1:
            baseline_predictions = predictions.copy()

        agreement = sum(
            a == b for a, b in zip(predictions, baseline_predictions)
        )

        result = {
            "batch_size": batch_size,
            "seconds": round(elapsed, 2),
            "real_tokens": real_tokens,
            "padded_slots": padded_slots,
            "real_tokens_per_second": round(real_tokens / elapsed, 2),
            "slots_per_second": round(padded_slots / elapsed, 2),
            "correct": correct,
            "total": len(gold),
            "agreement_with_batch1": agreement,
            "peak_pytorch_vram_gb": round(peak_vram, 2),
            "predictions": predictions
        }

        results.append(result)

        print(f"Time: {elapsed:.1f} seconds")
        print(f"Real generated tokens: {real_tokens}")
        print(f"Padded token slots: {padded_slots}")
        print(f"Real tokens/sec: {real_tokens / elapsed:.1f}")
        print(f"Accuracy: {correct}/{len(gold)}")
        print(f"Agreement with batch 1: {agreement}/{len(gold)}")
        print(f"Peak PyTorch VRAM: {peak_vram:.2f} GB")

        # Save results atomically after every successful batch size.
        temporary = Path("results/batching_v2_results.tmp")
        destination = Path("results/batching_v2_results.json")
        temporary.write_text(json.dumps(results, indent=2))
        temporary.replace(destination)

    except torch.cuda.OutOfMemoryError:
        print(f"OUT OF MEMORY at batch size {batch_size}.", flush=True)
        print("Stopping larger-batch tests safely.", flush=True)
        break

print("\n=== BENCHMARK FINISHED ===")
print("Results: results/batching_v2_results.json")
