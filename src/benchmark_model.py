
import time
from pathlib import Path

import torch

from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

torch.set_num_threads(2)

model = create_model(num_classes=100)
model.eval()

batch_size = 1
dummy_input = torch.randn(batch_size, 3, 32, 32)

warmup_runs = 10
benchmark_runs = 50

with torch.inference_mode():
    for _ in range(warmup_runs):
        model(dummy_input)

    start_time = time.perf_counter()

    for _ in range(benchmark_runs):
        model(dummy_input)

    end_time = time.perf_counter()

total_time = end_time - start_time
latency_ms = (total_time / benchmark_runs) * 1000
throughput = batch_size * benchmark_runs / total_time

print("Baseline ResNet-18 CPU Benchmark")
print(f"Device: CPU")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"Batch size: {batch_size}")
print(f"Warm-up runs: {warmup_runs}")
print(f"Benchmark runs: {benchmark_runs}")
print(f"Average latency: {latency_ms:.2f} ms/image")
print(f"Throughput: {throughput:.2f} images/second")
