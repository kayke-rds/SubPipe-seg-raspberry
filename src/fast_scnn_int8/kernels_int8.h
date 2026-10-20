#ifndef KERNELS_INT8_H
#define KERNELS_INT8_H

#include <stdint.h>
#include "fastscnn_weights.h"

/*
 * Convolução INT8 genérica com bias pré-escalado e requantização.
 *
 *   X  : int8, layout NCHW contíguo  [Cin , Hin , Win ]
 *   Y  : int8, layout NCHW contíguo  [Cout, Hout, Wout]
 *   d  : descritor da conv (Cout, Cin, k, stride, pad, groups, has_relu,
 *                            X_zp, Y_zp, X_scale, Y_scale)
 *   base : ponteiro para o início do fastscnn_weights.bin carregado em RAM
 *
 * Fórmula executada:
 *   acc = Σ X_i8·W_i8 + bias_corr[q]
 *   y   = clamp( round(acc · M[q]) + Y_zp , lo , 127 )   com lo = Y_zp se has_relu
 */
void conv2d_int8(const int8_t *X, int Hin, int Win,
                 int8_t       *Y, int Hout, int Wout,
                 const FastSCNNConvDesc *d,
                 const uint8_t *base);

#endif
