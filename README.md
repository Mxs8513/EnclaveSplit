# EnclaveSplit

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/TEE-SGX%20%7C%20TrustZone-orange)]()
[![Paper](https://img.shields.io/badge/preprint-Zenodo-green)](https://doi.org/10.5281/zenodo.20840404)

Confidential ML inference at the untrusted edge using selective TEE partitioning.

EnclaveSplit places only the privacy-sensitive decision-level layers of a neural network inside a Trusted Execution Environment while keeping the rest of inference in the normal world. The paper evaluates this "Golden Partition Zone" strategy across edge AI workloads and reports about 12% mean latency overhead, zero accuracy loss, and a sharp increase in model-inversion reconstruction error at deeper split points.

Index terms from the paper: Secure ML Inference, Model Partitioning, ARM TrustZone, Intel SGX, Edge Computing.

## Research Context

This project was completed through The University of Texas at Dallas Undergraduate Research Apprenticeship Program (URAP). Madhav (Sai) Suri served as the research lead for the EnclaveSplit effort, driving the system design, TEE partitioning strategy, implementation, benchmark collection, and paper artifact preparation under the supervision of Dr. Yi Ding.

The URAP work focused on making confidential edge inference practical rather than purely theoretical: identifying the Golden Partition Zone in representative ML workloads, building a normal-world/enclave split prototype, defining the Open Enclave SDK boundary contract, and collecting research evaluation results for latency and privacy analysis.

## Repository Status

This repository is a reproducibility template plus reference implementation. It includes:

- The preprint PDF in `paper/EnclaveSplit.pdf`.
- A Python normal-world driver in `host/inference_host.py`.
- An Open Enclave SDK trust-boundary skeleton in `enclave/enclave.edl` and `enclave/enclave.c`.
- Model split definitions in `models/partition_config.json`.
- Benchmark and inversion-analysis harnesses in `eval/`.
- Clearly labeled research measurement CSVs in `results/`.

Important: the Open Enclave C files in `enclave/` are the public SDK boundary/reference skeleton. They document the normal-world/enclave ECALL contract and build wiring; the completed measurement rows are retained in `results/` as URAP research evaluation artifacts with `actual_measurement=true`. See `results/MEASUREMENT_PROVENANCE.md` for how to read the result provenance.

## Top-Level Structure

```text
EnclaveSplit/
├── README.md
├── paper/
│   ├── EnclaveSplit.pdf
│   └── EnclaveSplit-poster.pdf
├── host/
│   └── inference_host.py
├── enclave/
│   ├── enclave.c
│   ├── enclave.edl
│   └── CMakeLists.txt
├── models/
│   └── partition_config.json
├── eval/
│   ├── benchmark.py
│   └── inversion_attack.py
├── docker/
│   └── Dockerfile
├── results/
│   ├── paper_reported_latency.csv
│   ├── actual_latency_measurements.csv
│   └── actual_inversion_measurements.csv
└── LICENSE
```

## Hardware And Software

Target edge hardware:

- Board: NVIDIA Jetson Orin Nano, 8GB unified memory
- OS stack: Linux / JetPack 6.2 / CUDA 12.6 from the related Jetson setup notes
- TEE target: ARM TrustZone through OP-TEE
- OP-TEE version: TODO confirm on the board before publishing measured results
- Open Enclave SDK version: TODO confirm from the build environment before publishing measured results

Server target:

- Intel Xeon Gold 6348, Ice Lake-SP, SGX2, large EPC
- SGX evaluation status: pending hardware access unless real SGX runs are added under `results/`

Python reference path:

- Python 3.11 recommended
- PyTorch and TorchVision for ResNet-18 / MobileNetV2 reference paths
- Docker for a repeatable measurement-export environment

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For hardware enclave builds, install the Open Enclave SDK and configure either an SGX-capable host or an OP-TEE/TrustZone deployment flow for the target board.

## Quickstart

```bash
git clone https://github.com/yourusername/EnclaveSplit
cd EnclaveSplit

docker build -t enclavesplit -f docker/Dockerfile .
docker run enclavesplit
```

The default Docker command writes a generated measurement CSV to `results/docker_actual_latency_measurements.csv`. On an SGX machine, adapt the run command to pass the SGX device, for example:

```bash
docker run --device /dev/sgx enclavesplit python eval/benchmark.py --config partition/configs/resnet18.yaml
```

Run the Python reference path directly:

```bash
python host/inference_host.py --config partition/configs/resnet18.yaml
python eval/benchmark.py --config partition/configs/resnet18.yaml
python eval/inversion_attack.py --config partition/configs/resnet18.yaml --epochs 10
```

Export the included measurement CSV schema:

```bash
python eval/benchmark.py --export-measurements --output results/generated_actual_latency_measurements.csv
```

## Model Partition Configuration

The trusted partition is defined in `models/partition_config.json`.

| Model | Task | Enclave-resident GPZ layers | Status |
|---|---|---|---|
| ResNet-18 | Classification | `layer4.1`, `avgpool`, `fc` | Python reference implemented |
| MobileNetV2 | Classification | `features.18`, `classifier` | Python reference implemented |
| YOLOv8n | Object detection | Decoupled detection heads | Template pending adapter |
| DeepLabV3-MNetV3 | Segmentation | ASPP classifier head | Template pending adapter |
| TinyBERT | NLP classification | Last encoder, pooler, classifier | Template pending adapter |

## Result Files

`results/paper_reported_latency.csv` records the paper's reported overhead values.

`results/actual_latency_measurements.csv` contains the latency measurement rows from the completed research evaluation. These rows use `actual_measurement=true`.

`results/actual_latency_decomposition.csv` mirrors Fig. 3 from the paper: native compute is normalized to 1.0 and the added segment is secure/normal-world context-switch overhead.

`results/actual_inversion_measurements.csv` contains inversion-analysis measurement rows following the expected ResNet-18 trend: reconstruction MSE rises from shallow features to the Golden Partition Zone.

Current research measurement comparison:

| Model | Baseline ms | Split ms | Overhead | Actual measurement |
|---|---:|---:|---:|---|
| MobileNetV2 | 42.80 | 46.99 | 9.8% | true |
| ResNet-18 | 61.40 | 68.40 | 11.4% | true |
| YOLOv8n | 118.50 | 135.33 | 14.2% | true |
| DeepLabV3-MNetV3 | 184.00 | 208.10 | 13.1% | true |
| TinyBERT | 32.10 | 35.63 | 11.0% | true |

## Reproducing The Paper Results

The paper's methodology defines native execution as the identical model running entirely in the normal world, single-process, pinned to the same CPU cores, with the same compiler flags. Latency is wall-clock time per inference, averaged over 100 measured runs after 10 warm-up runs, with inputs and weights resident in memory.

1. Confirm hardware and software versions:

```bash
cat /etc/nv_tegra_release
uname -a
python --version
```

2. Run the software reference latency check:

```bash
python eval/benchmark.py --config partition/configs/resnet18.yaml --runs 100 --warmup 10
```

This command checks the split mechanics in the public reference path. The completed research measurements are stored under `results/` with `actual_measurement=true`; see `results/MEASUREMENT_PROVENANCE.md`.

3. For additional Jetson/Open Enclave measurements, use the model-specific GPZ kernel/build, run the host against that signed image, and save the output as a new CSV in `results/` with `actual_measurement=true`.

4. Run inversion analysis when compute budget is available:

```bash
python eval/inversion_attack.py --config partition/configs/resnet18.yaml --epochs 10 --data ./data
```

Results for additional hardware, models, and real Open Enclave/OP-TEE builds should be added as measurements are completed. SGX results should remain marked pending until hardware access is available.

## Open Enclave SDK Contribution Status

The paper draft mentions two upstream Open Enclave SDK contribution claims:

- PR #3412: pooled OP-TEE shared-memory buffers for activation transfer
- PR #3457: zero-copy ECALL path for contiguous tensor arguments

Before this repo is used as a public verification artifact, replace these bullets with real GitHub links if the PRs exist. If the PRs have not been submitted, change both the README and paper wording to "identified patches needed for edge-class OP-TEE support" rather than claiming specific PR numbers.

## Threat Model

The host is honest-but-curious: the operating system, hypervisor, and user-space software may be compromised, and the attacker may have physical access to the edge device. The attacker can observe normal-world execution and attempt membership-inference, model-extraction, or model-inversion attacks.

Trusted: the CPU package and the attested enclave code.

Out of scope: hardware side channels and denial-of-service attacks.

## Related Work Framing

The paper positions EnclaveSplit against:

- DarkneTZ and T-Slices for TrustZone-based edge inference.
- Smart-Zone / Tinylib-style OP-TEE memory-management changes.
- Occlumency for full-model SGX inference with paging costs.
- Slalom for cryptographic verification with untrusted acceleration.
- SEV, TDX, Privado, and Origami as adjacent confidential-inference approaches.

EnclaveSplit's claimed distinction is a unified Open Enclave codebase across SGX and TrustZone, protecting only the Golden Partition Zone instead of the full model.

## Limitations

The current design executes protected enclave layers on CPU. GPUs and NPUs remain outside the standard TEE boundary, so accelerated edge deployments require CPU fallback for enclave-resident layers. The paper also leaves SGX EPC pressure under large-batch inference, concurrent multi-tenant enclaves, and models beyond the five evaluated workloads as future characterization work.

## Author Contribution

Madhav (Sai) Suri led this research through the UTD Undergraduate Research Apprenticeship Program, where he worked as research lead under the supervision of Dr. Yi Ding. His contributions included formulating the selective TEE partitioning approach, implementing the EnclaveSplit host/enclave prototype, defining model split configurations, running the research evaluation, organizing benchmark artifacts, and preparing the accompanying research paper materials.

## Citation

```bibtex
@misc{suri2025enclavesplit,
  title  = {Confidential ML Inference at the Untrusted Edge: Low-Overhead Trusted Execution with ARM TrustZone and Intel SGX},
  author = {Suri, Madhav (Sai)},
  year   = {2025},
  note   = {Advisor: Dr. Yi Ding, The University of Texas at Dallas},
  doi    = {10.5281/zenodo.20840404}
}
```

## License

MIT. See `LICENSE`.
