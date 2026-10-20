#include "kernels_int8.h"
#include <math.h>
#include <stddef.h>

void conv2d_int8(const int8_t *X, int Hin, int Win,
                 int8_t       *Y, int Hout, int Wout,
                 const FastSCNNConvDesc *d,
                 const uint8_t *base)
{
    const int8_t  *W      = (const int8_t  *)(base + d->off_W);
    const int32_t *bias_c = (const int32_t *)(base + d->off_bias_corr);
    const float   *M      = (const float   *)(base + d->off_M);

    const int Cout     = d->Cout;
    const int Cin      = d->Cin;
    const int k        = d->k;
    const int S        = d->stride;
    const int P        = d->pad;
    const int G        = d->groups;
    const int Cin_g    = Cin  / G;
    const int Cout_g   = Cout / G;
    const int Y_zp     = d->Y_zp;
    const int X_zp     = d->X_zp;
    const int has_relu = d->has_relu;

    /* =========================================================
     * Fast path 1: 1×1, s=1, p=0, g=1  (~85% dos MACs)
     * Sem boundary, sem ky/kx. Processa uma linha de saída por
     * vez com acumulador em registradores/stack (vetorizável).
     * ========================================================= */
    if (k == 1 && S == 1 && P == 0 && G == 1) {
        const int HW = Hout * Wout;

        /* buffer de acumulação (por linha) — Wout máx = 271 */
        int32_t acc_row[272];

        for (int co = 0; co < Cout; co++) {
            const float   m_co = M[co];
            const int32_t b_co = bias_c[co];
            const int8_t *wp   = W + co * Cin;
            int8_t       *yp   = Y + co * HW;

            for (int oy = 0; oy < Hout; oy++) {
                /* inicializa com bias */
                for (int ox = 0; ox < Wout; ox++) acc_row[ox] = b_co;

                /* acumula: acc_row[ox] += Σ_ci w[ci]·x[ci][oy][ox] */
                for (int ci = 0; ci < Cin; ci++) {
                    const int8_t *xp = X + ci * Hin * Win + oy * Wout;
                    const int32_t wv = (int32_t)wp[ci];
                    /* loop vetorizável: int32 += int32 * int8 */
                    for (int ox = 0; ox < Wout; ox++)
                        acc_row[ox] += wv * (int32_t)xp[ox];
                }

                /* requantização + clamp + escrita */
                for (int ox = 0; ox < Wout; ox++) {
                    int32_t q = (int32_t)lrintf((float)acc_row[ox] * m_co) + Y_zp;
                    if (has_relu && q < Y_zp) q = Y_zp;
                    if (q < -128) q = -128;
                    if (q >  127) q =  127;
                    yp[oy * Wout + ox] = (int8_t)q;
                }
            }
        }
        return;
    }

    /* =========================================================
     * Fast path 2: depthwise puro (Cin_g = Cout_g = 1)  (~7%)
     * Sem loop de ci; pré-calcula a correção de padding.
     * ========================================================= */
    if (Cin_g == 1 && Cout_g == 1) {
        for (int co = 0; co < Cout; co++) {
            const float   m_co = M[co];
            const int32_t b_co = bias_c[co];
            const int8_t *xp   = X + co * Hin * Win;
            const int8_t *wp   = W + co * k * k;
            int8_t       *yp   = Y + co * Hout * Wout;

            for (int oy = 0; oy < Hout; oy++) {
                for (int ox = 0; ox < Wout; ox++) {
                    int32_t acc = b_co;

                    for (int ky = 0; ky < k; ky++) {
                        const int iy = oy * S - P + ky;
                        if (iy < 0 || iy >= Hin) {
                            /* linha inteira em padding: soma X_zp · w */
                            for (int kx = 0; kx < k; kx++)
                                acc += X_zp * (int32_t)wp[ky*k+kx];
                            continue;
                        }
                        const int8_t *xr = xp + iy * Win;
                        for (int kx = 0; kx < k; kx++) {
                            const int ix = ox * S - P + kx;
                            int v = (ix >= 0 && ix < Win)
                                    ? (int)xr[ix]
                                    : X_zp;
                            acc += v * (int32_t)wp[ky*k+kx];
                        }
                    }

                    int32_t q = (int32_t)lrintf((float)acc * m_co) + Y_zp;
                    if (has_relu && q < Y_zp) q = Y_zp;
                    if (q < -128) q = -128;
                    if (q >  127) q =  127;
                    yp[oy * Wout + ox] = (int8_t)q;
                }
            }
        }
        return;
    }

    /* =========================================================
     * Fast path 3 (3×3, g=1, e o l2d_conv) — mantém o genérico
     * abaixo, mas com o boundary check hoisted por linha.
     * ========================================================= */
    for (int co = 0; co < Cout; co++) {
        const float   m_co = M[co];
        const int32_t b_co = bias_c[co];
        const int g        = co / Cout_g;
        const int ci_base  = g * Cin_g;

        for (int oy = 0; oy < Hout; oy++) {
            for (int ox = 0; ox < Wout; ox++) {
                int32_t acc = 0;
                for (int ci = 0; ci < Cin_g; ci++) {
                    const int8_t *xp = X + (ci_base + ci) * Hin * Win;
                    const int8_t *wp = W + (co * Cin_g + ci) * k * k;
                    for (int ky = 0; ky < k; ky++) {
                        const int iy = oy * S - P + ky;
                        const int row_valid = (iy >= 0 && iy < Hin);
                        const int8_t *xr = row_valid ? xp + iy * Win : NULL;
                        for (int kx = 0; kx < k; kx++) {
                            const int ix = ox * S - P + kx;
                            int v = (row_valid && ix >= 0 && ix < Win)
                                    ? (int)xr[ix]
                                    : X_zp;
                            acc += v * (int32_t)wp[ky*k+kx];
                        }
                    }
                }
                acc += b_co;
                int32_t q = (int32_t)lrintf((float)acc * m_co) + Y_zp;
                if (has_relu && q < Y_zp) q = Y_zp;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                Y[co * Hout * Wout + oy * Wout + ox] = (int8_t)q;
            }
        }
    }
}
