"""A2 — hardware probe for the training box.

Measures, on whatever accelerator `get_device()` picks:

  * device properties (name, VRAM, SM count, clocks, bf16 support)
  * sustained FP32 / TF32 / BF16 matmul throughput (TFLOPS)
  * host->device and device->host bandwidth, pageable and pinned

Nothing here touches production code.  Run:

    ssh -o BatchMode=yes jashm@192.168.1.7 \
      'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/win_bootstrap.sh \
       run scripts/profiling/a2_hw_probe.py'
"""
from __future__ import annotations

import json
import time

import torch


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def matmul_tflops(
    dev: torch.device, n: int, dtype: torch.dtype, tf32: bool, iters: int = 50
) -> float:
    """Sustained TFLOPS for an n x n x n matmul."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(10):                       # warmup + clock ramp
        c = a @ b
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    _sync(dev)
    dt = time.perf_counter() - t0
    del a, b, c
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return 2.0 * n**3 * iters / dt / 1e12


def conv_tflops(dev: torch.device, in_ch: int, iters: int = 30,
                amp: bool = False) -> float:
    """Sustained TFLOPS for the shape the TCN actually runs.

    Conv1d(in_ch -> 64, k=3) over [B*N, in_ch, L].  This is the kernel that
    dominates the model, so its achieved rate matters more than matmul peak.
    """
    b, length = 128 * 504, 60
    x = torch.randn(b, in_ch, length, device=dev)
    conv = torch.nn.Conv1d(in_ch, 64, 3, padding=2).to(dev)
    ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp)
    with torch.no_grad(), ctx:
        for _ in range(5):
            y = conv(x)
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            y = conv(x)
        _sync(dev)
    dt = time.perf_counter() - t0
    flops = 2.0 * b * 64 * in_ch * 3 * length * iters
    del x, y, conv
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return flops / dt / 1e12


def pcie_bandwidth(dev: torch.device, mb: int = 232, iters: int = 30) -> dict:
    """H2D / D2H bandwidth at the size of one production minibatch."""
    n = mb * 1024 * 1024 // 4
    out: dict[str, float] = {}
    for pinned in (False, True):
        host = torch.empty(n, dtype=torch.float32, pin_memory=pinned)
        gpu = torch.empty(n, dtype=torch.float32, device=dev)
        for _ in range(3):
            gpu.copy_(host)
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            gpu.copy_(host)
        _sync(dev)
        h2d = mb / 1024 * iters / (time.perf_counter() - t0)
        for _ in range(3):
            host.copy_(gpu)
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            host.copy_(gpu)
        _sync(dev)
        d2h = mb / 1024 * iters / (time.perf_counter() - t0)
        tag = "pinned" if pinned else "pageable"
        out[f"h2d_GiBs_{tag}"] = h2d
        out[f"d2h_GiBs_{tag}"] = d2h
        del host, gpu
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep: dict = {"torch": torch.__version__, "device": str(dev)}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        rep["gpu"] = {
            "name": p.name,
            "vram_GiB": round(p.total_memory / 2**30, 3),
            "sm_count": p.multi_processor_count,
            "capability": f"{p.major}.{p.minor}",
            "bf16": torch.cuda.is_bf16_supported(),
            "cuda": torch.version.cuda,
        }

    rep["throughput_TFLOPS"] = {}
    for n in (2048, 4096, 8192):
        rep["throughput_TFLOPS"][f"fp32_notf32_{n}"] = matmul_tflops(
            dev, n, torch.float32, tf32=False
        )
        rep["throughput_TFLOPS"][f"fp32_tf32_{n}"] = matmul_tflops(
            dev, n, torch.float32, tf32=True
        )
        rep["throughput_TFLOPS"][f"bf16_{n}"] = matmul_tflops(
            dev, n, torch.bfloat16, tf32=False
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rep["conv1d_TFLOPS"] = {
        "in_ch_15_fp32": conv_tflops(dev, 15),
        "in_ch_16_fp32": conv_tflops(dev, 16),
        "in_ch_64_fp32": conv_tflops(dev, 64),
        # The 15 -> 16 padding claim in R3 is about tensor cores, which need
        # bf16/TF32.  Measure the same convs under autocast so the sizing is
        # against the precision the training loop actually uses.
        "in_ch_15_bf16": conv_tflops(dev, 15, amp=True),
        "in_ch_16_bf16": conv_tflops(dev, 16, amp=True),
        "in_ch_64_bf16": conv_tflops(dev, 64, amp=True),
    }
    rep["pcie"] = pcie_bandwidth(dev)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
