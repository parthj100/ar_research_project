"""
Quantize distilled custom students and measure size/CPU latency.
Supports:
- mobilenetv3_small_100 (mobilenetv3)
- efficientnet_b0
- mobilevit_xxs (optional)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import time
import json
import timm

from scripts.train_distill_custom import DistillableStudent


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_size_mb(state_dict_path: Path):
    return state_dict_path.stat().st_size / (1024 * 1024)


def measure_latency(model: nn.Module, device: torch.device, runs: int = 100, warmup: int = 10):
    model.eval()
    input_tensor = torch.randn(1, 8, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)["logits"] if isinstance(model(input_tensor), dict) else model(input_tensor)
    torch.cuda.empty_cache() if device.type == "cuda" else None
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(input_tensor)["logits"] if isinstance(model(input_tensor), dict) else model(input_tensor)
            latencies.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def quantize_and_measure(model_name: str, checkpoint: Path, num_actions: int = 12, out_dir: Path = None):
    device_cpu = torch.device("cpu")
    # Ensure quantization engine is set
    torch.backends.quantized.engine = "qnnpack"

    # Build model and load weights
    student = DistillableStudent(model_name=model_name, num_actions=num_actions, teacher_embed_dim=512)
    state = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    student.load_state_dict(state, strict=False)
    student.eval()

    # Original size/latency
    size_mb = count_params(student)
    ckpt_size_mb = get_size_mb(checkpoint)
    latency_cpu = measure_latency(student, device_cpu)

    # Quantize (dynamic on Linear)
    quantized = torch.quantization.quantize_dynamic(
        student, {nn.Linear}, dtype=torch.qint8
    )
    q_latency_cpu = measure_latency(quantized, device_cpu)

    results = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "params": size_mb,
        "ckpt_size_mb": ckpt_size_mb,
        "latency_cpu_ms": latency_cpu,
        "quantized_latency_cpu_ms": q_latency_cpu,
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "quantization_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        torch.save(quantized.state_dict(), out_dir / "quantized_state_dict.pt")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["mobilevit_xxs", "mobilenetv3", "efficientnet_b0"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-actions", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="results/quantized_custom")
    args = parser.parse_args()

    res = quantize_and_measure(
        model_name=args.model,
        checkpoint=Path(args.checkpoint),
        num_actions=args.num_actions,
        out_dir=Path(args.output_dir) / args.model,
    )
    print(json.dumps(res, indent=2))

