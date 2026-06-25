"""Latency benchmark — reproduces the *method* behind Table I / Fig. 2-3.

Measures mean wall-clock per inference for:
  native  — the whole model in one process (no boundary)
  split   — early/late layers in the host + the GPZ block via the enclave path

and reports the split overhead. On a commodity CPU the "enclave" is the software
stub, so the absolute overhead is NOT the paper's number — the paper's ~12% is
measured on SGX2 / OP-TEE hardware where the cost is dominated by real
secure/normal-world context switches (Fig. 3). Set ENCLAVE_CROSSING_US to model
a per-crossing switch cost.

    python eval/bench_latency.py --config partition/configs/resnet18.yaml --runs 100
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from partition.splitter import build, load_config          # noqa: E402
from host.secure_stub import EnclaveStub                    # noqa: E402


def _time(fn, x, runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(x)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        samples.append(time.perf_counter() - t0)
    return statistics.mean(samples)


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sm = build(args.config)
    enclave = EnclaveStub(gpz_forward=sm.enclave)
    enclave.remote_attest()

    x = torch.randn(*cfg["input_shape"])
    torch.set_grad_enabled(False)

    native = lambda inp: sm.forward_native(inp)
    split = lambda inp: sm.tail(enclave.ecall_run_gpz(sm.head(inp)))

    t_native = _time(native, x, args.runs, args.warmup)
    t_split = _time(split, x, args.runs, args.warmup)
    overhead = (t_split - t_native) / t_native * 100

    print(f"Model:                  {cfg['model']}")
    print(f"Native latency (mean):  {t_native * 1e3:8.3f} ms")
    print(f"Split  latency (mean):  {t_split * 1e3:8.3f} ms")
    print(f"Overhead (this host):   {overhead:6.1f} %")
    print(f"Paper overhead (HW):    {cfg.get('paper_overhead_pct', '—')} %  "
          f"(SGX2 / OP-TEE; this host's number reflects the software stub)")


if __name__ == "__main__":
    main()
