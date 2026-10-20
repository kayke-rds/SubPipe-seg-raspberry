/* fastscnn_ppm_fp32.h -- GERADO. NÃO EDITAR. */
#ifndef FASTSCNN_PPM_FP32_H
#define FASTSCNN_PPM_FP32_H
#include <stdint.h>

#define FASTSCNN_PPM_N_CONV    5u
#define FASTSCNN_PPM_BIN_SIZE  197632u
#define FASTSCNN_PPM_IN_SCALE  0.09730412811040878f
#define FASTSCNN_PPM_IN_ZP     0
#define FASTSCNN_PPM_OUT_SCALE 0.033021531999111176f
#define FASTSCNN_PPM_OUT_ZP    -128

typedef struct {
    uint32_t off_W, off_B;
    uint16_t Cin, Cout;
} FastSCNNPPMConvDesc;

static const FastSCNNPPMConvDesc FASTSCNN_PPM_CONV[FASTSCNN_PPM_N_CONV] = {
    {0, 16384, 128, 32},  /* conv1 */
    {16512, 32896, 128, 32},  /* conv2 */
    {33024, 49408, 128, 32},  /* conv3 */
    {49536, 65920, 128, 32},  /* conv4 */
    {66048, 197120, 256, 128},  /* out */
};

#endif
