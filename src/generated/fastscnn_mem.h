/* fastscnn_mem.h -- GERADO por plan_memory_fastscnn.py. NÃO EDITAR. */
#ifndef FASTSCNN_MEM_H
#define FASTSCNN_MEM_H
#include <stdint.h>

#define FASTSCNN_ARENA_BYTES   1727040u
#define FASTSCNN_ARENA_ALIGN   64u

typedef enum {
    T_l2d_conv = 0,
    T_l2d_ds1_dw = 1,
    T_l2d_ds1_pw = 2,
    T_l2d_ds2_dw = 3,
    T_l2d_ds2_pw = 4,
    T_b1_0_expand = 5,
    T_b1_0_dw = 6,
    T_b1_0_project = 7,
    T_b1_1_expand = 8,
    T_b1_1_dw = 9,
    T_b1_1_project = 10,
    T_b1_1_add = 11,
    T_b1_2_expand = 12,
    T_b1_2_dw = 13,
    T_b1_2_project = 14,
    T_b1_2_add = 15,
    T_b2_0_expand = 16,
    T_b2_0_dw = 17,
    T_b2_0_project = 18,
    T_b2_1_expand = 19,
    T_b2_1_dw = 20,
    T_b2_1_project = 21,
    T_b2_1_add = 22,
    T_b2_2_expand = 23,
    T_b2_2_dw = 24,
    T_b2_2_project = 25,
    T_b2_2_add = 26,
    T_b3_0_expand = 27,
    T_b3_0_dw = 28,
    T_b3_0_project = 29,
    T_b3_1_expand = 30,
    T_b3_1_dw = 31,
    T_b3_1_project = 32,
    T_b3_1_add = 33,
    T_b3_2_expand = 34,
    T_b3_2_dw = 35,
    T_b3_2_project = 36,
    T_b3_2_add = 37,
    T_ppm_x_fp32 = 38,
    T_ppm_pool1 = 39,
    T_ppm_pool2 = 40,
    T_ppm_pool3 = 41,
    T_ppm_pool6 = 42,
    T_ppm_conv1_out = 43,
    T_ppm_conv2_out = 44,
    T_ppm_conv3_out = 45,
    T_ppm_conv4_out = 46,
    T_ppm_up1 = 47,
    T_ppm_up2 = 48,
    T_ppm_up3 = 49,
    T_ppm_up6 = 50,
    T_ppm_cat = 51,
    T_ppm_out_fp32 = 52,
    T_ppm_out_i8 = 53,
    T_ff_up = 54,
    T_ff_dw = 55,
    T_ff_low = 56,
    T_ff_high = 57,
    T_ff_add = 58,
    T_cls_ds1_dw = 59,
    T_cls_ds1_pw = 60,
    T_cls_ds2_dw = 61,
    T_cls_ds2_pw = 62,
    T_cls_out = 63,
    T_output = 64,
    T_COUNT
} TensorId;

typedef enum { DT_F32=0, DT_I8=1, DT_U8=2, DT_I32=3, DT_I64=4 } TensorDType;

typedef struct {
    uint32_t off;
    uint32_t size;
    uint16_t c, h, w;
    uint8_t  dtype;
} TensorDesc;

static const TensorDesc FASTSCNN_TENSOR[T_COUNT] = {
    {0u, 1378880u, 32, 159, 271, DT_I8},  /* l2d_conv */
    {1378880u, 348160u, 32, 80, 136, DT_I8},  /* l2d_ds1_dw */
    {0u, 522240u, 48, 80, 136, DT_I8},  /* l2d_ds1_pw */
    {522240u, 130560u, 48, 40, 68, DT_I8},  /* l2d_ds2_dw */
    {0u, 174080u, 64, 40, 68, DT_I8},  /* l2d_ds2_pw */
    {174080u, 1044480u, 384, 40, 68, DT_I8},  /* b1_0_expand */
    {1218560u, 261120u, 384, 20, 34, DT_I8},  /* b1_0_dw */
    {174080u, 43520u, 64, 20, 34, DT_I8},  /* b1_0_project */
    {217600u, 261120u, 384, 20, 34, DT_I8},  /* b1_1_expand */
    {478720u, 261120u, 384, 20, 34, DT_I8},  /* b1_1_dw */
    {217600u, 43520u, 64, 20, 34, DT_I8},  /* b1_1_project */
    {261120u, 43520u, 64, 20, 34, DT_I8},  /* b1_1_add */
    {304640u, 261120u, 384, 20, 34, DT_I8},  /* b1_2_expand */
    {565760u, 261120u, 384, 20, 34, DT_I8},  /* b1_2_dw */
    {174080u, 43520u, 64, 20, 34, DT_I8},  /* b1_2_project */
    {217600u, 43520u, 64, 20, 34, DT_I8},  /* b1_2_add */
    {261120u, 261120u, 384, 20, 34, DT_I8},  /* b2_0_expand */
    {174080u, 65280u, 384, 10, 17, DT_I8},  /* b2_0_dw */
    {239360u, 16320u, 96, 10, 17, DT_I8},  /* b2_0_project */
    {255680u, 97920u, 576, 10, 17, DT_I8},  /* b2_1_expand */
    {353600u, 97920u, 576, 10, 17, DT_I8},  /* b2_1_dw */
    {174080u, 16320u, 96, 10, 17, DT_I8},  /* b2_1_project */
    {190400u, 16320u, 96, 10, 17, DT_I8},  /* b2_1_add */
    {206720u, 97920u, 576, 10, 17, DT_I8},  /* b2_2_expand */
    {304640u, 97920u, 576, 10, 17, DT_I8},  /* b2_2_dw */
    {174080u, 16320u, 96, 10, 17, DT_I8},  /* b2_2_project */
    {206720u, 16320u, 96, 10, 17, DT_I8},  /* b2_2_add */
    {223040u, 97920u, 576, 10, 17, DT_I8},  /* b3_0_expand */
    {320960u, 97920u, 576, 10, 17, DT_I8},  /* b3_0_dw */
    {174080u, 21760u, 128, 10, 17, DT_I8},  /* b3_0_project */
    {195840u, 130560u, 768, 10, 17, DT_I8},  /* b3_1_expand */
    {326400u, 130560u, 768, 10, 17, DT_I8},  /* b3_1_dw */
    {195840u, 21760u, 128, 10, 17, DT_I8},  /* b3_1_project */
    {217600u, 21760u, 128, 10, 17, DT_I8},  /* b3_1_add */
    {239360u, 130560u, 768, 10, 17, DT_I8},  /* b3_2_expand */
    {369920u, 130560u, 768, 10, 17, DT_I8},  /* b3_2_dw */
    {174080u, 21760u, 128, 10, 17, DT_I8},  /* b3_2_project */
    {195840u, 21760u, 128, 10, 17, DT_I8},  /* b3_2_add */
    {391680u, 87040u, 128, 10, 17, DT_F32},  /* ppm_x_fp32 */
    {683648u, 512u, 128, 1, 1, DT_F32},  /* ppm_pool1 */
    {680448u, 2048u, 128, 2, 2, DT_F32},  /* ppm_pool2 */
    {671232u, 4608u, 128, 3, 3, DT_F32},  /* ppm_pool3 */
    {652800u, 18432u, 128, 6, 6, DT_F32},  /* ppm_pool6 */
    {684672u, 128u, 32, 1, 1, DT_F32},  /* ppm_conv1_out */
    {684160u, 512u, 32, 2, 2, DT_F32},  /* ppm_conv2_out */
    {682496u, 1152u, 32, 3, 3, DT_F32},  /* ppm_conv3_out */
    {675840u, 4608u, 32, 6, 6, DT_F32},  /* ppm_conv4_out */
    {174080u, 21760u, 32, 10, 17, DT_F32},  /* ppm_up1 */
    {565760u, 21760u, 32, 10, 17, DT_F32},  /* ppm_up2 */
    {587520u, 21760u, 32, 10, 17, DT_F32},  /* ppm_up3 */
    {609280u, 21760u, 32, 10, 17, DT_F32},  /* ppm_up6 */
    {217600u, 174080u, 256, 10, 17, DT_F32},  /* ppm_cat */
    {478720u, 87040u, 128, 10, 17, DT_F32},  /* ppm_out_fp32 */
    {631040u, 21760u, 128, 10, 17, DT_I8},  /* ppm_out_i8 */
    {684800u, 348160u, 128, 40, 68, DT_I8},  /* ff_up */
    {174080u, 348160u, 128, 40, 68, DT_I8},  /* ff_dw */
    {522240u, 348160u, 128, 40, 68, DT_I8},  /* ff_low */
    {174080u, 348160u, 128, 40, 68, DT_I8},  /* ff_high */
    {870400u, 348160u, 128, 40, 68, DT_I8},  /* ff_add */
    {0u, 348160u, 128, 40, 68, DT_I8},  /* cls_ds1_dw */
    {348160u, 348160u, 128, 40, 68, DT_I8},  /* cls_ds1_pw */
    {0u, 348160u, 128, 40, 68, DT_I8},  /* cls_ds2_dw */
    {348160u, 348160u, 128, 40, 68, DT_I8},  /* cls_ds2_pw */
    {0u, 2752u, 1, 40, 68, DT_I8},  /* cls_out */
    {2752u, 174080u, 1, 320, 544, DT_I8},  /* output */
};

#define TENSOR_F32(a, id) ((float  *)((a) + FASTSCNN_TENSOR[id].off))
#define TENSOR_I8(a, id)  ((int8_t *)((a) + FASTSCNN_TENSOR[id].off))

#endif /* FASTSCNN_MEM_H */
