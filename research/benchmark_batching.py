import re
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-1.7B-Base"
MAX_NEW_TOKENS = 128

print("Loading training-pool questions...", flush=True)
data = load_dataset(
    "openai/gsm8k", "main", split="train",
    revision="740312add88f781978c0658806c59bc2815b9866"
).shuffle(seed=2026)

prefix = (
    "Solve the math problems step by step. "
    "Finish each solution with #### followed by the numeric answer.\n\n"
)

for ex in data.select(range(2)):
    solution = re.sub(r"<<[^<>]*>>", "", ex["answer"])
    prefix += f"Question: {ex['question']}\nSolution: {solution}\n\n"

prompts = [
    prefix + f"Question: {ex['question']}\nSolution:"
    for ex in data.select(range(272, 288))
]

print("Loading Qwen model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cuda:0"
)
model.eval()

# Warm up GPU before timing.
inputs = tokenizer(prompts[0], return_tensors="pt").to("cuda")
with torch.inference_mode():
    model.generate(
        **inputs, max_new_tokens=8, do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
torch.cuda.synchronize()
del inputs

for batch_size in [1, 2, 4, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total_tokens = 0

    print(f"\nTesting batch size {batch_size}...", flush=True)
    start = time.perf_counter()

    try:
        for i in range(0, len(prompts), batch_size):
            inputs = tokenizer(
                prompts[i:i + batch_size],
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            total_tokens += outputs.shape[0] * new_tokens
            del inputs, outputs

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_reserved() / (1024 ** 3)

        print(f"Batch size: {batch_size}")
        print(f"Elapsed: {elapsed:.1f} seconds")
        print(f"Generated token slots: {total_tokens}")
        print(f"Throughput: {total_tokens / elapsed:.1f} tokens/sec")
        print(f"Peak PyTorch VRAM: {peak:.2f} GB")

    except torch.cuda.OutOfMemoryError:
        print(f"Batch size {batch_size}: OUT OF MEMORY")
        torch.cuda.empty_cache()

print("\n=== BATCH BENCHMARK COMPLETE ===")
