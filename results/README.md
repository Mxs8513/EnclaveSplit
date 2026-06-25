# Results

This directory contains the current actual data tables for the repository.

- `paper_reported_latency.csv` records the values currently described in the paper/README.
- `actual_latency_predictions.csv` contains the current hardware-measured latency comparison data.
- `actual_latency_decomposition.csv` contains the current hardware-measured normalized latency decomposition.
- `actual_inversion_predictions.csv` contains the current hardware-measured inversion-attack data.

For additional runs, add new CSVs with `actual_measurement=true`, the exact command, hardware, JetPack version, OP-TEE/Open Enclave version, timestamp, and commit SHA.
