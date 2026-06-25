"""Run a single EnclaveSplit inference.

Untrusted host (this process) runs the early/late layers; the Golden Partition
Zone block runs inside the enclave — here the in-process secure stub, on real
hardware the attested Open Enclave secure world.

    python host/run_split_inference.py --config partition/configs/resnet18.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from partition.splitter import build, load_config          # noqa: E402
from host.secure_stub import EnclaveStub                    # noqa: E402


def run(config_path: str):
    import torch

    cfg = load_config(config_path)
    sm = build(config_path)
    enclave = EnclaveStub(gpz_forward=sm.enclave)
    enclave.remote_attest()                                 # once per session

    x = torch.randn(*cfg["input_shape"])

    # ── Normal world ──────────────────────────────────────────────────────────
    activations = sm.head(x)                                # early/late layers (host)
    # ── Secure world (single ECALL into the GPZ block) ────────────────────────
    gpz_out = enclave.ecall_run_gpz(activations)
    # ── Back in the normal world ──────────────────────────────────────────────
    logits = sm.tail(gpz_out)

    print(f"Model:            {cfg['model']}")
    print(f"Enclave (GPZ):    {cfg['split']['enclave_modules']}")
    print(f"Real TEE:         {enclave.is_real_tee()}  (software stub on this host)")
    print(f"Activation shape: {tuple(activations.shape)}  ← only this crosses the boundary")
    print(f"Output shape:     {tuple(logits.shape)}")
    if cfg["task"] == "classification":
        topk = torch.topk(logits.softmax(-1), k=3, dim=-1)
        print(f"Top-3 class ids:  {topk.indices[0].tolist()}")
    return logits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
