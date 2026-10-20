#include <stdio.h>
#include <stdlib.h>
#include "fastscnn_forward.h"

int main(void)
{
    if (fastscnn_init("fastscnn_weights.bin", "fastscnn_ppm_fp32.bin") < 0) return 1;

    size_t n = 3 * 320 * 544;
    float *x = malloc(n * sizeof(float));
    FILE *f = fopen("test_forward_input.bin", "rb");
    if (!f || fread(x, sizeof(float), n, f) != n) { perror("input"); return 1; }
    fclose(f);

    fastscnn_forward(x);

    const int8_t *y = fastscnn_output();
    size_t ny = 320 * 544;
    f = fopen("test_forward_c.bin", "wb");
    fwrite(y, 1, ny, f);
    fclose(f);

    int mn = 127, mx = -128; long s = 0;
    for (size_t i = 0; i < ny; i++) { s += y[i]; if (y[i] < mn) mn = y[i]; if (y[i] > mx) mx = y[i]; }
    printf("out: min=%d max=%d mean=%.2f\n", mn, mx, (double)s / ny);

    free(x);
    return 0;
}
