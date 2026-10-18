import json
import re
import time
from decimal import Decimal
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
SEED = 2026
PILOT_SIZE = 20

torch.manual_seed(SEED)
Path("results").mkdir(exist_ok=True)

print("Downloading/loading GSM8K training split...")
data = load_dataset(
    "openai/gsm8k", "main", split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=SEED)

# First 2 questions: examples shown to the model.
# Next 200: reserved development questions.
# All remaining questions: available for later training.
examples = data.select(range(2))
pilot = data.select(range(2, 2 + PILOT_SIZE))

def gold_answer(text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    return Decimal(match.group(1).replace(",", ""))

def predicted_answer(text):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    return Decimal(match.group(1).replace(",", "")) if match else None

prefix = (
    "Solve the math problems step by step. "
    "Finish each solution with #### followed by the numeric answer.\n\n"
)

for example in examples:
    solution = re.sub(r"<<[^<>]*>>", "", example["answer"])
    prefix += (
        f"Question: {example['question']}\n"
        f"Solution: {solution}\n\n"
    )

print("Loading original base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda:0"
)
model.eval()

correct = 0
parsed = 0
started = time.time()

with open("results/gsm8k_baseline_pilot.jsonl", "w") as file:
    for i, example in enumerate(pilot, 1):
        prompt = prefix + f"Question: {example['question']}\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        gold = gold_answer(example["answer"])
        prediction = predicted_answer(response)
        is_correct = prediction == gold

        correct += int(is_correct)
        parsed += int(prediction is not None)

        record = {
            "question": example["question"],
            "gold": str(gold),
            "prediction": str(prediction) if prediction is not None else None,
            "correct": is_correct,
            "response": response
        }
        file.write(json.dumps(record) + "\n")
        file.flush()

        print(
            f"{i:02d}/{PILOT_SIZE} | "
            f"Expected: {gold} | "
            f"Predicted: {prediction} | "
            f"Correct: {is_correct}",
            flush=True
        )

print("\n=== BASELINE PILOT RESULTS ===")
print(f"Correct: {correct}/{PILOT_SIZE}")
print(f"Accuracy: {100 * correct / PILOT_SIZE:.1f}%")
print(f"Answers parsed: {parsed}/{PILOT_SIZE}")
print(f"Elapsed: {(time.time() - started):.1f} seconds")
print(
    "Peak reserved PyTorch VRAM:",
    round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    "GB"
)
print("Detailed results: results/gsm8k_baseline_pilot.jsonl")
