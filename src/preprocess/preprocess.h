#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <stdint.h>

typedef struct {
    float r;
    int   dw, dh;
    int   new_h, new_w;
} LetterboxInfo;

/* Letterbox RGB → NCHW float.
 * Pipeline: bilinear resize + padding 114 + /255 + (v - mean[c]) / std[c].
 * `mean` e `std` são arrays RGB (3 elementos cada).
 */
void letterbox_image_rgb_to_nchw(const uint8_t *img, int src_h, int src_w,
                                 float *out_nchw, int dst_h, int dst_w,
                                 LetterboxInfo *info);

/* Letterbox da máscara GT → uint8 [dst_h, dst_w] com 0/1.
 * Usa o mesmo (r, dw, dh) da imagem para alinhamento pixel-perfect. */
void letterbox_mask_nearest(const uint8_t *mask, int src_h, int src_w,
                            uint8_t *out, int dst_h, int dst_w,
                            const LetterboxInfo *info);

void letterbox_image_rgb_to_nchw_norm(const uint8_t *img, int src_h, int src_w,
                                      float *out_nchw, int dst_h, int dst_w,
                                      const float mean[3], const float std[3],
                                      LetterboxInfo *info);

#endif
