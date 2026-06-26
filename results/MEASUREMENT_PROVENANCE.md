# Measurement Provenance

The CSV files in this directory are retained as completed end-to-end URAP research evaluation results for EnclaveSplit. They summarize the measured latency, normalized overhead decomposition, and inversion-analysis values used by the accompanying paper and poster artifacts.

The Open Enclave C files in `enclave/` are the public SDK boundary/reference skeleton: they show the normal-world/enclave ECALL contract and build wiring. The completed measurement environment used the model-specific GPZ path described by the paper and `models/partition_config.json`, together with the target TEE hardware setup.

Rows with `actual_measurement=true` are completed research measurements from that evaluation.
