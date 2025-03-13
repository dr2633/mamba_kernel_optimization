import torch
from mem0 import Mem0
import time

# Load model and move to GPU
model = Mem0.from_pretrained('path/to/model').cuda().eval()

# Prepare dummy input matching expected model dimensions
input_tensor = torch.randn(batch_size, seq_length, model_dim).cuda()

# Warm-up
for _ in range(10):
    _ = model(input_tensor)

# Timed benchmark
torch.cuda.synchronize()
import time
start_time = time.time()

iterations = 100
for _ in range(iterations):
    _ = model(input_tensor)
torch.cuda.synchronize()

end_time = time.time()
elapsed_time = (end_time - start_time) / iterations

print(f"Average inference latency: {elapsed_time*1000:.2f} ms")
