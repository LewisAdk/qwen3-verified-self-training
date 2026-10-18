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
MAX_TOKENS = 256

Path("results").mkdir(exist_ok=True)
torch.manual_seed(SEED)

def extract_answer(text):
    match = re.search(
        r"####[ \t]*(-?[\d,]+(?:\.\d+)?)", text
    )
    if not match:
        return None
    return Decimal(match.group(1).replace(",", ""))

class StopAfterAnswer(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = self.tokenizer.decode(
            input_ids[0, self.prompt_length:],
            skip_special_tokens=True
        )
        return bool(re.search(
            r"####[ \t]*-?[\d,]+(?:\.\d+)?[ \t]*\n",
            generated
        ))

print("Loading GSM8K...")
data = load_dataset(
    "openai/gsm8k",
    "main",
    split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=SEED)

examples = data.select(range(2))
development = data.select(range(22, 222))

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

print("Loading original Qwen model...")
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

output_path = "results/gsm8k_dev_baseline.jsonl"

with open(output_path, "w") as file:
    for i, example in enumerate(development, 1):
        prompt = prefix + (
            f"Question: {example['question']}\nSolution:"
        )
        inputs = tokenizer(
            prompt, return_tensors="pt"
        ).to("cuda")

        stopping = StoppingCriteriaList([
            StopAfterAnswer(
                tokenizer, inputs["input_ids"].shape[1]
            )
        ])

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                stopping_criteria=stopping,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = (
            outputs.shape[1] - inputs["input_ids"].shape[1]
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        gold = extract_answer(example["answer"])
        prediction = extract_answer(response)
        is_correct = prediction == gold

        correct += int(is_correct)
        parsed += int(prediction is not None)

        record = {
            "index": i + 21,
            "question": example["question"],
            "gold": str(gold),
            "prediction": str(prediction)
                if prediction is not None else None,
            "correct": is_correct,
            "generated_tokens": generated_tokens,
            "response": response,
        }

        file.write(json.dumps(record) + "\n")
        file.flush()

        print(
            f"{i:03d}/200 | "
            f"Correct: {correct} | "
            f"Running accuracy: {100*correct/i:.1f}%",
            flush=True
        )

print("\n=== DEVELOPMENT BASELINE ===")
print(f"Correct: {correct}/200")
print(f"Accuracy: {100*correct/200:.1f}%")
print(f"Parsed: {parsed}/200")
print(f"Elapsed: {(time.time()-started)/60:.1f} minutes")
print(
    "Peak PyTorch VRAM:",
    round(torch.cuda.max_memory_reserved()/1024**3, 2),
    "GB"
)
print("Saved:", output_path)
