/* fastscnn_runtime.c -- Fast-SCNN INT8 (40 convs) + PPM em FP32.
 *
 * Ordem de execução (a numeração em `t=` casa com birth/death do tensors_c.csv):
 *   t=0..4    learning_to_downsample
 *   t=5..15   bottleneck1 (2 residuais)
 *   t=16..26  bottleneck2 (2 residuais)
 *   t=27..37  bottleneck3 (2 residuais) + b3_2_add
 *   t=38      PPM (FP32): pools, 5 convs 1x1, upsamples, concat, quantize
 *   t=39..43  feature_fusion (upsample, dwconv, conv_lower, ff_high, ff_add)
 *   t=44..48  classifier
 *   t=49      upsample final ×8
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fastscnn_forward.h"
#include "fastscnn_mem.h"
#include "fastscnn_weights.h"
#include "fastscnn_ppm_fp32.h"
#include "kernels_int8.h"

/* ---- estado global ---- */
static uint8_t *g_weights = NULL;
static uint8_t *g_ppm     = NULL;
static int      g_dump    = 0;

static uint8_t  g_arena_storage[FASTSCNN_ARENA_BYTES + 64] __attribute__((aligned(64)));
static uint8_t *g_arena = g_arena_storage;     /* start alinhado a 64 B */

static int8_t   g_input_i8[3 * 320 * 544] __attribute__((aligned(64)));

void fastscnn_enable_dump(void) { g_dump = 1; }

#define A          (g_arena)
#define TI8(id)    ((int8_t *) (A + FASTSCNN_TENSOR[id].off))
#define TF32(id)   ((float  *) (A + FASTSCNN_TENSOR[id].off))

const int8_t *fastscnn_logits(void) { return TI8(T_cls_out); }

static void dump_tensor(int id, int count)
{
    if (!g_dump) return;
    char p[64]; snprintf(p, sizeof(p), "dump_T%03d.bin", id);
    FILE *f = fopen(p, "wb");
    if (f) { fwrite(TI8(id), 1, (size_t)count, f); fclose(f); }
}

/* ============================================================
 *  Helpers
 * ============================================================ */

static void quantize_input(const float *in, int8_t *out, int n,
                           float scale, int zp)
{
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(in[i] / scale) + zp;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out[i] = (int8_t)q;
    }
}

static void resize_nearest_i8(const int8_t *in, int c, int h, int w,
                              int8_t *out, int oh, int ow)
{
    for (int ch = 0; ch < c; ch++) {
        const int8_t *ip = in  + (size_t)ch * h * w;
        int8_t       *op = out + (size_t)ch * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            int iy = (int)((long)oy * h / oh);
            for (int ox = 0; ox < ow; ox++) {
                int ix = (int)((long)ox * w / ow);
                op[oy * ow + ox] = ip[iy * w + ix];
            }
        }
    }
}

static void resize_nearest_fp32(const float *in, int c, int h, int w,
                                float *out, int oh, int ow)
{
    for (int ch = 0; ch < c; ch++) {
        const float *ip = in  + (size_t)ch * h * w;
        float       *op = out + (size_t)ch * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            int iy = (int)((long)oy * h / oh);
            for (int ox = 0; ox < ow; ox++) {
                int ix = (int)((long)ox * w / ow);
                op[oy * ow + ox] = ip[iy * w + ix];
            }
        }
    }
}

/* Add com requantização entre escalas diferentes.
 * out = quantize( dequantize(a) + dequantize(b), S_o, Z_o )
 * relu=1 aplica max(0, ·) antes de quantizar. */
static void add_i8_requant(int8_t *out, const int8_t *a, const int8_t *b, int n,
                           float S_a, int Z_a,
                           float S_b, int Z_b,
                           float S_o, int Z_o, int relu)
{
    for (int i = 0; i < n; i++) {
        float va = ((float)a[i] - (float)Z_a) * S_a;
        float vb = ((float)b[i] - (float)Z_b) * S_b;
        float v  = va + vb;
        if (relu && v < 0.f) v = 0.f;
        int q = (int)lrintf(v / S_o) + Z_o;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out[i] = (int8_t)q;
    }
}

static void dequantize_i8(const int8_t *x, float *y, int n, float scale, int zp)
{
    for (int i = 0; i < n; i++) y[i] = ((float)x[i] - (float)zp) * scale;
}

static void adaptive_avgpool_fp32(const float *X, int C, int Hin, int Win,
                                  float *Y, int n)
{
    for (int c = 0; c < C; c++) {
        const float *x = X + (size_t)c * Hin * Win;
        float       *y = Y + (size_t)c * n * n;
        for (int oy = 0; oy < n; oy++) {
            int y0 = (int)floorf((float)oy       * Hin / n);
            int y1 = (int)ceilf ((float)(oy + 1) * Hin / n);
            for (int ox = 0; ox < n; ox++) {
                int x0 = (int)floorf((float)ox       * Win / n);
                int x1 = (int)ceilf ((float)(ox + 1) * Win / n);
                float s = 0.f; int cnt = 0;
                for (int iy = y0; iy < y1; iy++)
                    for (int ix = x0; ix < x1; ix++) { s += x[iy * Win + ix]; cnt++; }
                y[oy * n + ox] = s / (float)cnt;
            }
        }
    }
}

static void conv1x1_fp32(const float *X, int Cin, int HW,
                         float *Y, int Cout,
                         const float *W, const float *B, int relu)
{
    for (int co = 0; co < Cout; co++) {
        float *y = Y + co * HW;
        const float *w = W + co * Cin;
        float b = B[co];
        for (int p = 0; p < HW; p++) y[p] = b;
        for (int ci = 0; ci < Cin; ci++) {
            const float *x = X + ci * HW;
            float wc = w[ci];
            for (int p = 0; p < HW; p++) y[p] += wc * x[p];
        }
        if (relu) for (int p = 0; p < HW; p++) if (y[p] < 0.f) y[p] = 0.f;
    }
}

/* ============================================================
 *  PPM (FP32) -- input é o b3_2_add INT8 (FASTSCNN_PPM_IN_*)
 * ============================================================ */
static void ppm_forward_from(const int8_t *x_i8)
{
    const int C = 128, H = 10, Wd = 17, HW = H * Wd;

    float *x_fp32   = TF32(T_ppm_x_fp32);
    float *pool1    = TF32(T_ppm_pool1);
    float *pool2    = TF32(T_ppm_pool2);
    float *pool3    = TF32(T_ppm_pool3);
    float *pool6    = TF32(T_ppm_pool6);
    float *c1       = TF32(T_ppm_conv1_out);
    float *c2       = TF32(T_ppm_conv2_out);
    float *c3       = TF32(T_ppm_conv3_out);
    float *c6       = TF32(T_ppm_conv4_out);
    float *up1      = TF32(T_ppm_up1);
    float *up2      = TF32(T_ppm_up2);
    float *up3      = TF32(T_ppm_up3);
    float *up6      = TF32(T_ppm_up6);
    float *cat      = TF32(T_ppm_cat);
    float *out_f    = TF32(T_ppm_out_fp32);

    dequantize_i8(x_i8, x_fp32, C * HW,
                  FASTSCNN_PPM_IN_SCALE, FASTSCNN_PPM_IN_ZP);

    adaptive_avgpool_fp32(x_fp32, C, H, Wd, pool1, 1);
    adaptive_avgpool_fp32(x_fp32, C, H, Wd, pool2, 2);
    adaptive_avgpool_fp32(x_fp32, C, H, Wd, pool3, 3);
    adaptive_avgpool_fp32(x_fp32, C, H, Wd, pool6, 6);

    const float *Wp = (const float *)g_ppm;
    #define PW(i)  (Wp + FASTSCNN_PPM_CONV[i].off_W / 4)
    #define PB(i)  (Wp + FASTSCNN_PPM_CONV[i].off_B / 4)
    conv1x1_fp32(pool1, C,   1, c1, 32, PW(0), PB(0), 1);
    conv1x1_fp32(pool2, C,   4, c2, 32, PW(1), PB(1), 1);
    conv1x1_fp32(pool3, C,   9, c3, 32, PW(2), PB(2), 1);
    conv1x1_fp32(pool6, C,  36, c6, 32, PW(3), PB(3), 1);
    #undef PW
    #undef PB

    resize_nearest_fp32(c1, 32, 1, 1, up1, H, Wd);
    resize_nearest_fp32(c2, 32, 2, 2, up2, H, Wd);
    resize_nearest_fp32(c3, 32, 3, 3, up3, H, Wd);
    resize_nearest_fp32(c6, 32, 6, 6, up6, H, Wd);

    memcpy(cat +   0*HW, x_fp32, C*HW * sizeof(float));
    memcpy(cat + 128*HW, up1,    32*HW * sizeof(float));
    memcpy(cat + 160*HW, up2,    32*HW * sizeof(float));
    memcpy(cat + 192*HW, up3,    32*HW * sizeof(float));
    memcpy(cat + 224*HW, up6,    32*HW * sizeof(float));

    conv1x1_fp32(cat, 256, HW, out_f, 128,
                 Wp + FASTSCNN_PPM_CONV[4].off_W / 4,
                 Wp + FASTSCNN_PPM_CONV[4].off_B / 4, 1);

    int8_t *out_i8 = TI8(T_ppm_out_i8);
    const float s_out = FASTSCNN_PPM_OUT_SCALE;
    const int   z_out = FASTSCNN_PPM_OUT_ZP;
    for (int i = 0; i < 128 * HW; i++) {
        int q = (int)lrintf(out_f[i] / s_out) + z_out;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out_i8[i] = (int8_t)q;
    }
}

/* ============================================================
 *  Macro de conv INT8
 * ============================================================ */
#define RUN(X, hX, wX, id, hY, wY, ci) do {                                     \
    conv2d_int8((X), (hX), (wX), TI8(id), (hY), (wY),                           \
                &FASTSCNN_CONV[ci], g_weights);                                 \
    if (g_dump) {                                                               \
        char _p[64]; snprintf(_p, sizeof(_p), "dump_T%03d.bin", (int)(id));     \
        FILE *_f = fopen(_p, "wb");                                             \
        if (_f) {                                                               \
            fwrite(TI8(id), 1, (size_t)(hY)*(wY)*FASTSCNN_CONV[ci].Cout, _f);   \
            fclose(_f);                                                         \
        }                                                                       \
    }                                                                           \
} while (0)

/* ============================================================
 *  Forward completo
 * ============================================================ */
void fastscnn_forward(const float *input_nchw)
{
    /* t=0: quantiza input */
    quantize_input(input_nchw, g_input_i8, 3 * 320 * 544,
                   FASTSCNN_CONV[0].X_scale, FASTSCNN_CONV[0].X_zp);

    /* --- Learning to downsample (t=0..4) --- */
    RUN(g_input_i8,         320, 544, T_l2d_conv,   159, 271,  0);
    RUN(TI8(T_l2d_conv),    159, 271, T_l2d_ds1_dw,  80, 136,  1);
    RUN(TI8(T_l2d_ds1_dw),   80, 136, T_l2d_ds1_pw,  80, 136,  2);
    RUN(TI8(T_l2d_ds1_pw),   80, 136, T_l2d_ds2_dw,  40,  68,  3);
    RUN(TI8(T_l2d_ds2_dw),   40,  68, T_l2d_ds2_pw,  40,  68,  4);

    /* --- Bottleneck1 (t=5..15, 2 residuais) --- */
    RUN(TI8(T_l2d_ds2_pw),   40, 68, T_b1_0_expand,  40, 68,  5);
    RUN(TI8(T_b1_0_expand),  40, 68, T_b1_0_dw,      20, 34,  7);
    RUN(TI8(T_b1_0_dw),      20, 34, T_b1_0_project, 20, 34,  8);
    RUN(TI8(T_b1_0_project), 20, 34, T_b1_1_expand,  20, 34,  9);
    RUN(TI8(T_b1_1_expand),  20, 34, T_b1_1_dw,      20, 34, 10);
    RUN(TI8(T_b1_1_dw),      20, 34, T_b1_1_project, 20, 34, 11);

    /* t=11: b1_1_add = b1_0_project + b1_1_project */
    add_i8_requant(TI8(T_b1_1_add),
                   TI8(T_b1_0_project), TI8(T_b1_1_project), 64*20*34,
                   FASTSCNN_CONV[ 8].Y_scale, FASTSCNN_CONV[ 8].Y_zp,
                   FASTSCNN_CONV[11].Y_scale, FASTSCNN_CONV[11].Y_zp,
                   FASTSCNN_CONV[12].X_scale, FASTSCNN_CONV[12].X_zp, 0);

    dump_tensor(T_b1_1_add, 64*20*34);

    RUN(TI8(T_b1_1_add),     20, 34, T_b1_2_expand,  20, 34, 12);
    RUN(TI8(T_b1_2_expand),  20, 34, T_b1_2_dw,      20, 34, 13);
    RUN(TI8(T_b1_2_dw),      20, 34, T_b1_2_project, 20, 34, 14);

    /* t=15: b1_2_add = b1_1_add + b1_2_project */
    add_i8_requant(TI8(T_b1_2_add),
                   TI8(T_b1_1_add),     TI8(T_b1_2_project), 64*20*34,
                   FASTSCNN_CONV[12].X_scale, FASTSCNN_CONV[12].X_zp,
                   FASTSCNN_CONV[14].Y_scale, FASTSCNN_CONV[14].Y_zp,
                   FASTSCNN_CONV[15].X_scale, FASTSCNN_CONV[15].X_zp, 0);

    dump_tensor(T_b1_2_add, 64*20*34);

    /* --- Bottleneck2 (t=16..26) --- */
    RUN(TI8(T_b1_2_add),     20, 34, T_b2_0_expand,  20, 34, 15);
    RUN(TI8(T_b2_0_expand),  20, 34, T_b2_0_dw,      10, 17, 16);
    RUN(TI8(T_b2_0_dw),      10, 17, T_b2_0_project, 10, 17, 17);
    RUN(TI8(T_b2_0_project), 10, 17, T_b2_1_expand,  10, 17, 18);
    RUN(TI8(T_b2_1_expand),  10, 17, T_b2_1_dw,      10, 17, 19);
    RUN(TI8(T_b2_1_dw),      10, 17, T_b2_1_project, 10, 17, 20);

    /* t=22: b2_1_add */
    add_i8_requant(TI8(T_b2_1_add),
                   TI8(T_b2_0_project), TI8(T_b2_1_project), 96*10*17,
                   FASTSCNN_CONV[17].Y_scale, FASTSCNN_CONV[17].Y_zp,
                   FASTSCNN_CONV[20].Y_scale, FASTSCNN_CONV[20].Y_zp,
                   FASTSCNN_CONV[21].X_scale, FASTSCNN_CONV[21].X_zp, 0);

    dump_tensor(T_b2_1_add, 96*10*17);

    RUN(TI8(T_b2_1_add),     10, 17, T_b2_2_expand,  10, 17, 21);
    RUN(TI8(T_b2_2_expand),  10, 17, T_b2_2_dw,      10, 17, 22);
    RUN(TI8(T_b2_2_dw),      10, 17, T_b2_2_project, 10, 17, 23);

    /* t=26: b2_2_add */
    add_i8_requant(TI8(T_b2_2_add),
                   TI8(T_b2_1_add),     TI8(T_b2_2_project), 96*10*17,
                   FASTSCNN_CONV[21].X_scale, FASTSCNN_CONV[21].X_zp,
                   FASTSCNN_CONV[23].Y_scale, FASTSCNN_CONV[23].Y_zp,
                   FASTSCNN_CONV[24].X_scale, FASTSCNN_CONV[24].X_zp, 0);

    dump_tensor(T_b2_2_add, 96*10*17);

    /* --- Bottleneck3 (t=27..37) --- */
    RUN(TI8(T_b2_2_add),     10, 17, T_b3_0_expand,  10, 17, 24);
    RUN(TI8(T_b3_0_expand),  10, 17, T_b3_0_dw,      10, 17, 25);
    RUN(TI8(T_b3_0_dw),      10, 17, T_b3_0_project, 10, 17, 26);
    RUN(TI8(T_b3_0_project), 10, 17, T_b3_1_expand,  10, 17, 27);
    RUN(TI8(T_b3_1_expand),  10, 17, T_b3_1_dw,      10, 17, 28);
    RUN(TI8(T_b3_1_dw),      10, 17, T_b3_1_project, 10, 17, 29);

    /* t=33: b3_1_add */
    add_i8_requant(TI8(T_b3_1_add),
                   TI8(T_b3_0_project), TI8(T_b3_1_project), 128*10*17,
                   FASTSCNN_CONV[26].Y_scale, FASTSCNN_CONV[26].Y_zp,
                   FASTSCNN_CONV[29].Y_scale, FASTSCNN_CONV[29].Y_zp,
                   FASTSCNN_CONV[30].X_scale, FASTSCNN_CONV[30].X_zp, 0);

    dump_tensor(T_b3_1_add, 128*10*17);

    RUN(TI8(T_b3_1_add),     10, 17, T_b3_2_expand,  10, 17, 30);
    RUN(TI8(T_b3_2_expand),  10, 17, T_b3_2_dw,      10, 17, 31);
    RUN(TI8(T_b3_2_dw),      10, 17, T_b3_2_project, 10, 17, 32);

    /* t=37: b3_2_add -> escala do PPM_IN */
    add_i8_requant(TI8(T_b3_2_add),
                   TI8(T_b3_1_add),     TI8(T_b3_2_project), 128*10*17,
                   FASTSCNN_CONV[30].X_scale, FASTSCNN_CONV[30].X_zp,
                   FASTSCNN_CONV[32].Y_scale, FASTSCNN_CONV[32].Y_zp,
                   FASTSCNN_PPM_IN_SCALE,      FASTSCNN_PPM_IN_ZP, 0);

    dump_tensor(T_b3_2_add, 128*10*17);

    /* --- PPM FP32 (t=38) --- */
    ppm_forward_from(TI8(T_b3_2_add));

    dump_tensor(T_ppm_out_i8, 128*10*17);

    if (g_dump) {
        FILE *f = fopen("dump_T_ppm_out.bin", "wb");
        if (f) { fwrite(TI8(T_ppm_out_i8), 1, 128*10*17, f); fclose(f); }
    }

    /* --- FeatureFusion (t=39..43) --- */
    resize_nearest_i8(TI8(T_ppm_out_i8), 128, 10, 17,
                      TI8(T_ff_up), 40, 68);

    dump_tensor(T_ff_up, 128*40*68);

    RUN(TI8(T_ff_up),        40, 68, T_ff_dw,   40, 68, 33);
    RUN(TI8(T_ff_dw),        40, 68, T_ff_low,  40, 68, 34);
    RUN(TI8(T_l2d_ds2_pw),   40, 68, T_ff_high, 40, 68,  6);

    /* t=43: ff_add = relu(ff_high + ff_low) */
    add_i8_requant(TI8(T_ff_add),
                   TI8(T_ff_high),      TI8(T_ff_low),  128*40*68,
                   FASTSCNN_CONV[ 6].Y_scale, FASTSCNN_CONV[ 6].Y_zp,
                   FASTSCNN_CONV[34].Y_scale, FASTSCNN_CONV[34].Y_zp,
                   FASTSCNN_CONV[35].X_scale, FASTSCNN_CONV[35].X_zp, 1);

    dump_tensor(T_ff_add, 128*40*68);

    /* --- Classifier (t=44..48) --- */
    RUN(TI8(T_ff_add),       40, 68, T_cls_ds1_dw, 40, 68, 35);
    RUN(TI8(T_cls_ds1_dw),   40, 68, T_cls_ds1_pw, 40, 68, 36);
    RUN(TI8(T_cls_ds1_pw),   40, 68, T_cls_ds2_dw, 40, 68, 37);
    RUN(TI8(T_cls_ds2_dw),   40, 68, T_cls_ds2_pw, 40, 68, 38);
    RUN(TI8(T_cls_ds2_pw),   40, 68, T_cls_out,    40, 68, 39);

    /* --- Upsample final ×8 (t=49) --- */
    resize_nearest_i8(TI8(T_cls_out), 1, 40, 68, TI8(T_output), 320, 544);
    dump_tensor(T_output, 1*320*544);
}

#undef RUN

/* ============================================================
 *  API pública
 * ============================================================ */
const int8_t *fastscnn_output(void)      { return TI8(T_output); }
float         fastscnn_output_scale(void){ return FASTSCNN_CONV[39].Y_scale; }
int8_t        fastscnn_output_zp(void)   { return FASTSCNN_CONV[39].Y_zp; }

int fastscnn_init(const char *wpath, const char *ppath)
{
    FILE *f = fopen(wpath, "rb");
    if (!f) { perror(wpath); return -1; }
    g_weights = malloc(FASTSCNN_WEIGHTS_SIZE);
    if (!g_weights) { fclose(f); return -1; }
    if (fread(g_weights, 1, FASTSCNN_WEIGHTS_SIZE, f) != FASTSCNN_WEIGHTS_SIZE) {
        fclose(f); return -1;
    }
    fclose(f);

    f = fopen(ppath, "rb");
    if (!f) { perror(ppath); return -1; }
    g_ppm = malloc(FASTSCNN_PPM_BIN_SIZE);
    if (!g_ppm) { fclose(f); return -1; }
    if (fread(g_ppm, 1, FASTSCNN_PPM_BIN_SIZE, f) != FASTSCNN_PPM_BIN_SIZE) {
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}
