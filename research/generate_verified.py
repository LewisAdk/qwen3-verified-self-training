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
    StoppingCriteriaList
)

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
SEED = 2026
MAX_TOKENS = 256
CANDIDATES = 2

torch.manual_seed(SEED)
Path("results").mkdir(exist_ok=True)

ANSWER = re.compile(
    r"(?m)^####[ \t]*(-?[\d,]+(?:\.\d+)?)[ \t]*(?:\r?\n|$)"
)
STOP = re.compile(
    r"(?m)^####[ \t]*-?[\d,]+(?:\.\d+)?[ \t]*\r?\n"
)

def parse_number(text):
    match = ANSWER.search(text)
    if not match:
        return None
    return Decimal(match.group(1).replace(",", ""))

class StopAfterAnswer(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(
            input_ids[0, self.prompt_length:],
            skip_special_tokens=True
        )
        return bool(STOP.search(text))

print("Loading GSM8K training data...")
data = load_dataset(
    "openai/gsm8k", "main", split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=SEED)

examples = data.select(range(2))
problems = data.select(range(222, 272))

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

print("Loading original model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda:0"
)
model.eval()

total_correct = 0
verified_problems = 0
started = time.time()

with (
    open("results/all_candidates.jsonl", "w") as all_file,
    open("results/verified_training.jsonl", "w") as train_file
):
    for i, example in enumerate(problems, 1):
        question = example["question"]
        gold_match = re.search(
            r"####\s*(-?[\d,]+(?:\.\d+)?)",
            example["answer"]
        )
        gold = Decimal(gold_match.group(1).replace(",", ""))
        selected = None

        for candidate in range(1, CANDIDATES + 1):
            prompt = prefix + f"Question: {question}\nSolution:"
            inputs = tokenizer(
                prompt, return_tensors="pt"
            ).to("cuda")

            stopping = StoppingCriteriaList([
                StopAfterAnswer(
                    tokenizer, inputs["input_ids"].shape[1]
                )
            ])

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    stopping_criteria=stopping,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            prediction = parse_number(response)
            correct = prediction == gold
            total_correct += int(correct)

            record = {
                "dataset_index": i + 221,
                "candidate": candidate,
                "question": question,
                "gold": str(gold),
                "prediction": str(prediction)
                    if prediction is not None else None,
                "correct": correct,
                "response": response
            }

            all_file.write(json.dumps(record) + "\n")
            all_file.flush()

            # Keep at most one verified solution per problem.
            if correct and selected is None:
                match = ANSWER.search(response)
                solution = response[:match.end()].strip()

                selected = {
                    "dataset_index": i + 221,
                    "question": question,
                    "solution": solution,
                    "answer": str(gold),
                    "candidate": candidate
                }

        if selected is not None:
            train_file.write(json.dumps(selected) + "\n")
            train_file.flush()
            verified_problems += 1

        print(
            f"{i:02d}/50 | "
            f"Verified candidates: {total_correct}/{i*2} | "
            f"Training examples: {verified_problems}",
            flush=True
        )

print("\n=== GENERATION COMPLETE ===")
print(f"Candidates generated: {len(problems)*CANDIDATES}")
print(f"Correct candidates: {total_correct}")
print(f"Verified training examples: {verified_problems}")
print(f"Elapsed: {(time.time()-started)/60:.1f} minutes")
print("Saved: results/all_candidates.jsonl")
print("Saved: results/verified_training.jsonl")
