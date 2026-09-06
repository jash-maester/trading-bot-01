"""R4 VRAM/throughput probe: largest batch_days that fits under 8 GiB."""
import time

import torch

from trader.data.universe import active_tickers
from trader.models.signal import SignalConfig, SignalModel, horizon_key

N = len(active_tickers())
F = 15
L = 60
H = (5, 20)
dev = torch.device("cuda")
print(f"N={N} F={F} L={L} device={torch.cuda.get_device_name(0)}", flush=True)
print(f"{'batch':>6} {'peak GiB':>9} {'s/step':>8} {'verdict':>9}", flush=True)
for B in [4, 8, 12, 16, 24, 32]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        cfg = SignalConfig(in_features=F, embed_dim=128, num_channels=[128]*4,
                           kernel_size=3, dropout=0.1, head_hidden=32, horizons=H)
        m = SignalModel(cfg).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x = torch.randn(B, L, N, F, device=dev)
        y = {horizon_key(h): torch.randn(B, N, device=dev) for h in H}
        msk = torch.ones(B, N, dtype=torch.bool, device=dev)
        ts = []
        for i in range(4):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            out = m(x, msk)
            loss = sum(((out[k] - y[k])**2).mean() for k in y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            if i:
                ts.append(time.time() - t0)
        pk = torch.cuda.max_memory_allocated() / 2**30
        v = "FITS" if pk < 6.5 else ("TIGHT" if pk < 7.5 else "SPILLS")
        print(f"{B:>6} {pk:>9.2f} {sum(ts)/len(ts):>8.3f} {v:>9}", flush=True)
        del m, opt, x, y, msk, out, loss
    except torch.cuda.OutOfMemoryError:
        print(f"{B:>6} {'-':>9} {'-':>8} {'OOM':>9}", flush=True)
        break
