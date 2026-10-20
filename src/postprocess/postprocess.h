#ifndef POSTPROCESS_H
#define POSTPROCESS_H
#include <stdint.h>
#include "preprocess.h"

/* YOLO-specific — mantido, mas não usado pelo benchmark Fast-SCNN. */
void decode_mask_for_detection(const float *proto, const float *output0,
                               int det, int img_h, int img_w,
                               uint8_t *out_mask);

/* Métricas genéricas */
float mask_iou(const uint8_t *a, const uint8_t *b, int n);
float mask_dice(const uint8_t *a, const uint8_t *b, int n);
void  mask_precision_recall(const uint8_t *pred, const uint8_t *gt, int n,
                            float *precision, float *recall);

/* Visualização simples (overlay apenas) */
void visualize_comparison(const uint8_t *pred, const uint8_t *gt,
                          int w, int h, uint8_t *rgb_out);

/* Fast-SCNN: binariza logits INT8 em máscara 0/1.
 *   pred = (logit_q > y_zp) ? 1 : 0
 * O threshold em q é equivalente a logit > 0 em float. */
void fastscnn_binarize(const int8_t *logits, int n, int y_zp, uint8_t *out_mask);

/* Compõe imagem 1088x640 (WxH):
 *   [ GT mask 544x320 ] [ Pred mask 544x320 ]
 *   [      overlay 544x320 (centralizado)    ]
 * Overlay: TP=verde, FP=vermelho, FN=azul, TN=preto.
 * `out_rgb` deve ter 1088*640*3 bytes. */
void compose_output(const uint8_t *img_rgb, int sw, int sh,
                    const uint8_t *mask_gt, const uint8_t *mask_pred,
                    int w, int h, const LetterboxInfo *lb,
                    uint8_t *out_rgb);

void fastscnn_binarize_bilinear(const int8_t *logits, int h, int w, int y_zp,
                                uint8_t *mask, int oh, int ow);

void decode_mask_for_detection_multi(const float *proto, const float *output0,
                                     int num_classes, int num_anchors,
                                     int det, int img_h, int img_w,
                                     uint8_t *out_mask);

int  find_best_detection(const float *output0, int num_classes,
                         int num_anchors, int target_class);

void decode_fastscnn_mask_int8(const int8_t *output, int h, int w,
                               float scale, int zero_point, float threshold,
                               uint8_t *out_mask);

#endif
