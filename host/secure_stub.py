"""In-process secure stub for the enclave (software reference path).

When no TEE is present, this stands in for the Open Enclave *secure world* so the
EnclaveSplit pipeline runs end-to-end on a commodity CPU. It mimics the enclave's
contract — "decrypt" the Golden Partition Zone weights, run *only* the
decision-level block, hand the result back to the untrusted host — but it is
explicitly **not** a security boundary.

On real hardware this exact step is the attested C/C++ enclave in ../enclave,
loaded and measured via the Open Enclave SDK (ECALL/OCALL on SGX, SMC on
TrustZone). The point of the stub is to let you reproduce the *method* and
*measurement harness* without SGX2 / OP-TEE hardware.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class EnclaveStub:
    """Software stand-in for the attested enclave.

    Holds the GPZ-stage callable. On a real deployment the weights inside would
    arrive encrypted (E_k(W)) and be decrypted only after remote attestation;
    here we just call the in-memory block and account for the secure/normal-world
    transition cost the paper attributes most overhead to (Fig. 3).
    """
    gpz_forward: "callable"
    attested: bool = field(default=False)
    # Per-crossing fixed cost (microseconds) representing the ECALL/OCALL or SMC
    # context switch. Configurable so the harness can model different platforms;
    # absolute numbers on real hardware come from the enclave build, not this.
    crossing_us: float = float(os.environ.get("ENCLAVE_CROSSING_US", "0"))

    def remote_attest(self) -> None:
        """One-time, amortized per session (see paper §V-B)."""
        self.attested = True

    def ecall_run_gpz(self, activations):
        """The single ECALL: run the decision-level layers in the secure world."""
        if not self.attested:
            raise RuntimeError("enclave not attested — refusing to provision secrets")
        if self.crossing_us:
            time.sleep(self.crossing_us / 1e6)   # model the world-switch cost
        out = self.gpz_forward(activations)
        if self.crossing_us:
            time.sleep(self.crossing_us / 1e6)   # return crossing
        return out

    def is_real_tee(self) -> bool:
        return False
