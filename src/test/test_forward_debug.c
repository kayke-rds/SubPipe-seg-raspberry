/* test_forward_debug.c -- roda o forward com dump de cada tensor. */
#include <stdio.h>
#include <stdlib.h>
#include "fastscnn_forward.h"

extern void fastscnn_enable_dump(void);

int main(void)
{
    if (fastscnn_init("fastscnn_weights.bin", "fastscnn_ppm_fp32.bin") < 0) {
        fprintf(stderr, "init falhou\n");
        return 1;
    }
    fastscnn_enable_dump();

    size_t n = (size_t)3 * 320 * 544;
    float *x = malloc(n * sizeof(float));
    FILE *f = fopen("test_forward_input.bin", "rb");
    if (!f || fread(x, sizeof(float), n, f) != n) {
        perror("input"); return 1;
    }
    fclose(f);

    fastscnn_forward(x);

    const int8_t *y = fastscnn_output();
    size_t ny = (size_t)320 * 544;
    f = fopen("test_forward_c.bin", "wb");
    fwrite(y, 1, ny, f);
    fclose(f);

    free(x);
    return 0;
}
