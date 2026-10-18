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
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256

ANSWER = re.compile(r"####[ \t]*(-?[\d,]+(?:\.\d+)?)")
STOP = re.compile(r"####[ \t]*-?[\d,]+(?:\.\d+)?[ \t]*\r?\n")

Path("results").mkdir(exist_ok=True)
torch.manual_seed(SEED)

def extract_answer(text):
    match = ANSWER.search(text)
    if match is None:
        return None
    return str(Decimal(match.group(1).replace(",", "")))

class StopPerSequence(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length, batch_size):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_lengths = [None] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        current_length = input_ids.shape[1] - self.prompt_length

        for i in range(len(self.stop_lengths)):
            if self.stop_lengths[i] is not None:
                continue

            response = self.tokenizer.decode(
                input_ids[i, self.prompt_length:],
                skip_special_tokens=True
            )

            if STOP.search(response):
                self.stop_lengths[i] = current_length

        return torch.tensor(
            [length is not None for length in self.stop_lengths],
            dtype=torch.bool,
            device=input_ids.device
        )

print("Reading original baseline...", flush=True)

old_path = Path("results/gsm8k_dev_baseline.jsonl")
if not old_path.is_file():
    raise FileNotFoundError(f"Missing baseline: {old_path}")

with old_path.open() as f:
    original = [json.loads(line) for line in f]

assert len(original) == 200, "Original baseline must have 200 records."

print("Loading the exact same GSM8K data...", flush=True)

data = load_dataset(
    "openai/gsm8k",
    "main",
    split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=SEED)

fewshot = data.select(range(2))
development = data.select(range(22, 222))

prefix = (
    "Solve the math problems step by step. "
    "Finish each solution with #### followed by the numeric answer.\n\n"
)

for example in fewshot:
    solution = re.sub(r"<<[^<>]*>>", "", example["answer"])
    prefix += (
        f"Question: {example['question']}\n"
        f"Solution: {solution}\n\n"
    )

prompts = []
gold = []

for i, example in enumerate(development):
    old = original[i]
    expected_index = i + 22

    assert old["index"] == expected_index, "Index mismatch"
    assert old["question"] == example["question"], "Question mismatch"

    answer = extract_answer(example["answer"])
    assert old["gold"] == answer, "Ground-truth mismatch"

    prompts.append(
        prefix + f"Question: {example['question']}\nSolution:"
    )
    gold.append(answer)

print("All 200 questions match the original baseline.")
print("Loading original Qwen model...", flush=True)

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

correct = 0
agreement = 0
changed = 0
improved = 0
regressed = 0
all_results = []
useful_tokens = 0

torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()

for start in range(0, len(prompts), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(prompts))

    inputs = tokenizer(
        prompts[start:end],
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    prompt_length = inputs["input_ids"].shape[1]

    stopper = StopPerSequence(
        tokenizer,
        prompt_length,
        end - start
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

    for j in range(end - start):
        index = start + j
        tokens = generated[j].tolist()

        if stopper.stop_lengths[j] is not None:
            count = min(len(tokens), stopper.stop_lengths[j])
        elif tokenizer.eos_token_id in tokens:
            count = tokens.index(tokenizer.eos_token_id) + 1
        else:
            count = len(tokens)

        useful_tokens += count

        response = tokenizer.decode(
            tokens[:count],
            skip_special_tokens=True
        )

        prediction = extract_answer(response)
        is_correct = prediction == gold[index]

        old_prediction = original[index]["prediction"]
        old_correct = original[index]["correct"]

        same = prediction == old_prediction

        correct += int(is_correct)
        agreement += int(same)
        changed += int(not same)
        improved += int(is_correct and not old_correct)
        regressed += int(old_correct and not is_correct)

        all_results.append({
            "index": index + 22,
            "question": development[index]["question"],
            "gold": gold[index],
            "prediction": prediction,
            "correct": is_correct,
            "original_prediction": old_prediction,
            "same_as_original": same,
            "generated_tokens": count,
            "response": response
        })

    print(
        f"{end:03d}/200 | "
        f"Correct: {correct} | "
        f"Accuracy: {100 * correct / end:.1f}% | "
        f"Agreement: {agreement}/{end}",
        flush=True
    )

    del inputs, outputs, generated

torch.cuda.synchronize()
elapsed = time.perf_counter() - started

# Save without overwriting the original baseline.
destination = Path("results/gsm8k_dev_batch16.jsonl")
temporary = Path("results/gsm8k_dev_batch16.jsonl.tmp")

with temporary.open("w") as f:
    for record in all_results:
        f.write(json.dumps(record) + "\n")

temporary.replace(destination)

original_correct = sum(int(x["correct"]) for x in original)

print("\n=== BATCH-16 DEVELOPMENT BASELINE ===")
print(f"Original batch-1 accuracy: {original_correct}/200")
print(f"New batch-16 accuracy: {correct}/200")
print(f"New percentage accuracy: {100 * correct / 200:.1f}%")
print(f"Prediction agreement: {agreement}/200")
print(f"Changed predictions: {changed}")
print(f"Previously wrong, now correct: {improved}")
print(f"Previously correct, now wrong: {regressed}")
print(f"Useful generated tokens: {useful_tokens}")
print(f"Elapsed: {elapsed / 60:.1f} minutes")
print(
    "Peak PyTorch VRAM:",
    round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    "GB"
)
print("Saved:", destination)
