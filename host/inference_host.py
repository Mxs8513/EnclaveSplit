"""Normal-world EnclaveSplit inference driver.

This is the public entry point for the host side of the prototype. It runs the
non-sensitive model prefix in the normal world, sends only the boundary
activation to the enclave path, then consumes the protected GPZ output.

Without SGX/OP-TEE hardware this uses host.secure_stub.EnclaveStub, which is a
software stand-in for the enclave boundary and is not a security boundary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from host.run_split_inference import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one split inference.")
    parser.add_argument(
        "--config",
        default="partition/configs/resnet18.yaml",
        help="Path to a model split config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
