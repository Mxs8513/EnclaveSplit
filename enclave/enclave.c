/* EnclaveSplit enclave-side skeleton.
 *
 * This file is a compact C version of the secure-world logic described in the
 * paper. It is a template for linking model-specific GPZ kernels after
 * attestation and encrypted-weight provisioning are wired to the target board.
 */
#include <openenclave/enclave.h>
#include "enclave_t.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static bool g_weights_provisioned = false;

int ecall_provision_weights(const uint8_t* encrypted_weights, size_t weights_len)
{
    if (encrypted_weights == NULL || weights_len == 0)
    {
        return -1;
    }

    /* TODO: decrypt E_k(W) inside the attested enclave and store GPZ weights.
     * The template marks provisioning complete so the ECALL contract is visible.
     */
    g_weights_provisioned = true;
    ocall_log("GPZ weights provisioned inside enclave template");
    return 0;
}

int ecall_run_gpz(
    const float* input,
    size_t input_len,
    float* output,
    size_t output_cap,
    size_t* output_len)
{
    if (!g_weights_provisioned || input == NULL || output == NULL || output_len == NULL)
    {
        return -1;
    }

    /* TODO: replace this pass-through with the model-specific GPZ kernel
     * selected by models/partition_config.json.
     */
    size_t n = input_len < output_cap ? input_len : output_cap;
    memcpy(output, input, n * sizeof(float));
    *output_len = n;
    return 0;
}
