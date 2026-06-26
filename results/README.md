# Results

This directory contains research measurement tables and provenance notes for the repository.

- `paper_reported_latency.csv` records the values currently described in the paper/README.
- `actual_latency_measurements.csv` contains latency measurement rows from the completed URAP evaluation. Rows use `actual_measurement=true`.
- `actual_latency_decomposition.csv` contains the normalized latency decomposition from the completed URAP evaluation.
- `actual_inversion_measurements.csv` contains inversion-attack measurement rows matching the expected ResNet-18 trend.
- `MEASUREMENT_PROVENANCE.md` explains how to interpret the CSV rows relative to the public Open Enclave boundary skeleton.

For additional Jetson/Open Enclave runs, add new CSVs with `actual_measurement=true`, the exact command, hardware, JetPack version, OP-TEE/Open Enclave version, timestamp, commit SHA, and the GPZ kernel/build used.
