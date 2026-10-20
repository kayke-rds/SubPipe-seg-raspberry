#ifndef FASTSCNN_FORWARD_H
#define FASTSCNN_FORWARD_H
#include <stdint.h>

int          fastscnn_init(const char *weights_bin, const char *ppm_bin);
void         fastscnn_forward(const float *input_nchw);
const int8_t *fastscnn_output(void);
float        fastscnn_output_scale(void);
int8_t       fastscnn_output_zp(void);

/* opcional: dump de cada tensor de saída de conv para debug */
void         fastscnn_enable_dump(void);

const int8_t *fastscnn_logits(void);   /* 1×40×68 int8 logits pré-upsample */

#endif
