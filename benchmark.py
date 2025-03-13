import torch
import time
from transformers import AutoTokenizer
from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Llamba model and tokenizer clearly
model = LlambaLMHeadModel.from_pretrained("AvivBick/Llamba-3B", strict=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# Define input
prompt = "Hello, my name is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Warm-up clearly
for _ in range(5):
    _ = model.generate(input_ids, max_length=100)

torch.cuda.synchronize()

# Benchmark clearly
start_time = time.time()
for _ in range(20):
    output = model.generate(input_ids, max_length=100)

torch.cuda.synchronize()
elapsed_time = (time.time() - start_time) / 20

# Output clearly
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}\n")
print(f"Average inference latency: {elapsed_time * 1000:.2f} ms")
