# Extracted Paper Facts

This checklist mirrors the current PDF so README/results updates can stay synchronized with the paper.

- Title: Confidential ML Inference at the Untrusted Edge: Low-Overhead Trusted Execution with ARM TrustZone & Intel SGX
- Author: Madhav (Sai) Suri, The University of Texas at Dallas
- Advisor: Dr. Yi Ding
- Core method: selective partitioning at the Golden Partition Zone
- TEEs: ARM TrustZone / OP-TEE and Intel SGX / SGX2
- SDK: Open Enclave SDK
- Edge hardware: NVIDIA Jetson Orin Nano, ARM Cortex-A78AE, OP-TEE
- Server hardware: Intel Xeon Gold 6348, Ice Lake-SP, SGX2, large EPC
- Workloads: MobileNetV2, ResNet-18, YOLOv8n, DeepLabV3-MNetV3, TinyBERT
- Methodology: 100 measured runs after 10 warm-up runs, wall-clock per inference, warm cache, same CPU cores/compiler flags
- Accuracy loss: none reported
- Mean latency overhead: approximately 12 percent
- Latency overheads: MobileNetV2 9.8 percent; ResNet-18 11.4 percent; YOLOv8n 14.2 percent; DeepLabV3-MNetV3 13.1 percent; TinyBERT 11.0 percent
- ResNet-18 inversion MSE: layer1 0.011; layer2 0.014; layer3 0.022; layer4 0.049; fc 0.051
- Limitation: protected enclave layers run on CPU; GPU/NPU acceleration remains outside the standard TEE boundary
- Open Enclave contribution claims in the PDF: PR #3412 for pooled OP-TEE shared-memory buffers; PR #3457 for zero-copy ECALL contiguous tensor arguments
- Public-repo caution: verify PR links before claiming upstream contributions publicly.
