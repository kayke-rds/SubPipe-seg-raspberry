#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels_int8.h"

typedef struct {
    int conv_idx;
    int h_idx;
    int Cin, Hin, Win;
    int Cout, Hout, Wout;
} TestMeta;

static int read_meta(const char *path, TestMeta *m)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 0; }
    char key[32];
    int  val;
    m->h_idx = -1;
    while (fscanf(f, "%31s %d", key, &val) == 2) {
        if      (!strcmp(key, "conv_idx")) m->conv_idx = val;
        else if (!strcmp(key, "h_idx"))    m->h_idx    = val;
        else if (!strcmp(key, "Cin"))      m->Cin      = val;
        else if (!strcmp(key, "Hin"))      m->Hin      = val;
        else if (!strcmp(key, "Win"))      m->Win      = val;
        else if (!strcmp(key, "Cout"))     m->Cout     = val;
        else if (!strcmp(key, "Hout"))     m->Hout     = val;
        else if (!strcmp(key, "Wout"))     m->Wout     = val;
    }
    fclose(f);
    return 1;
}

static uint8_t *load_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read: %s\n", path);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    if (out_size) *out_size = (size_t)sz;
    return buf;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    TestMeta m = {0};
    if (!read_meta("test_meta.txt", &m)) return 1;

    if (m.h_idx < 0) {
        fprintf(stderr, "conv %d (ONNX) é do PPM, fora do .h\n", m.conv_idx);
        return 0;
    }

    printf("conv_idx=%d  h_idx=%d  Cin=%d Hin=%d Win=%d  Cout=%d Hout=%d Wout=%d\n",
           m.conv_idx, m.h_idx, m.Cin, m.Hin, m.Win, m.Cout, m.Hout, m.Wout);

    size_t ws_size;
    uint8_t *weights = load_file("fastscnn_weights.bin", &ws_size);
    if (!weights) return 1;

    size_t xs_size;
    int8_t *X = (int8_t *)load_file("test_input.bin", &xs_size);
    if (!X) { free(weights); return 1; }

    if (xs_size != (size_t)m.Cin * m.Hin * m.Win) {
        fprintf(stderr, "input size mismatch: %zu vs esperado %zu\n",
                xs_size, (size_t)m.Cin * m.Hin * m.Win);
        free(X); free(weights); return 1;
    }

    int8_t *Y = calloc((size_t)m.Cout * m.Hout * m.Wout, 1);
    if (!Y) { free(X); free(weights); return 1; }

    const FastSCNNConvDesc *d = &FASTSCNN_CONV[m.h_idx];
    printf("descriptor: Cout=%u Cin=%u k=%u s=%u p=%u g=%u relu=%u\n",
           d->Cout, d->Cin, d->k, d->stride, d->pad, d->groups, d->has_relu);
    printf("            X_scale=%.6e X_zp=%d Y_scale=%.6e Y_zp=%d\n",
           d->X_scale, d->X_zp, d->Y_scale, d->Y_zp);

    conv2d_int8(X, m.Hin, m.Win,
                Y, m.Hout, m.Wout,
                d, weights);

    FILE *f = fopen("test_output_c.bin", "wb");
    if (!f) { perror("test_output_c.bin"); return 1; }
    fwrite(Y, 1, (size_t)m.Cout * m.Hout * m.Wout, f);
    fclose(f);

    long sum = 0;
    int mn = 127, mx = -128;
    for (int i = 0; i < m.Cout * m.Hout * m.Wout; i++) {
        sum += Y[i];
        if (Y[i] < mn) mn = Y[i];
        if (Y[i] > mx) mx = Y[i];
    }
    printf("Y_c: min=%d max=%d mean=%.4f\n",
           mn, mx, (double)sum / (m.Cout * m.Hout * m.Wout));

    free(Y); free(X); free(weights);
    return 0;
}
