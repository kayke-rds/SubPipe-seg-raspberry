#ifndef FASTSCNN_H
#define FASTSCNN_H
#include <stdint.h>
#include "fastscnn_mem.h"
#include "fastscnn_weights.h"

int  fastscnn_init(const char *weights_path);   /* carrega .bin na RAM */
void fastscnn_forward(const float *input_nchw); /* input [3,320,544] */
const int8_t *fastscnn_output(void);            /* [1,40,68] int8 */
float fastscnn_output_scale(void);              /* Y_scale da última conv */
int8_t fastscnn_output_zp(void);

#endif
