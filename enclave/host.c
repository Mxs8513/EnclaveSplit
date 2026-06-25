/* EnclaveSplit normal-world Open Enclave launcher template. */
#include <openenclave/host.h>
#include "enclave_u.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void ocall_log(const char* message)
{
    printf("[enclave] %s\n", message);
}

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <signed-enclave-image>\n", argv[0]);
        return 1;
    }

    oe_enclave_t* enclave = NULL;
    oe_result_t result = oe_create_enclave_enclave(
        argv[1],
        OE_ENCLAVE_TYPE_AUTO,
        OE_ENCLAVE_FLAG_DEBUG,
        NULL,
        0,
        &enclave);

    if (result != OE_OK)
    {
        fprintf(stderr, "oe_create_enclave_enclave failed: %s\n", oe_result_str(result));
        return 1;
    }

    uint8_t encrypted_weights[] = {0x01};
    int status = 0;
    result = ecall_provision_weights(enclave, &status, encrypted_weights, sizeof(encrypted_weights));
    if (result != OE_OK || status != 0)
    {
        fprintf(stderr, "ecall_provision_weights failed\n");
        oe_terminate_enclave(enclave);
        return 1;
    }

    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0};
    size_t output_len = 0;
    result = ecall_run_gpz(enclave, &status, input, 4, output, 4, &output_len);
    if (result != OE_OK || status != 0)
    {
        fprintf(stderr, "ecall_run_gpz failed\n");
        oe_terminate_enclave(enclave);
        return 1;
    }

    printf("GPZ output length: %zu\n", output_len);
    oe_terminate_enclave(enclave);
    return 0;
}
