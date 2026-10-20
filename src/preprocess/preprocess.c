/* preprocess.c -- Letterbox unificado (YOLO26 + Fast-SCNN)
 *
 * YOLO26   : letterbox_image_rgb_to_nchw()      -> [0..1], pad 114/255
 * FastSCNN : letterbox_image_rgb_to_nchw_norm() -> (v - mean)/std por canal,
 *                                                  pad normalizado
 *
 * A interpolação da imagem é BILINEAR em ambos; a máscara GT usa NEAREST
 * e padding 0 para preservar a binariedade.
 */
#include "preprocess.h"
#include <math.h>
#include <string.h>

/* ============================================================== */
/* YOLO26: letterbox sem normalização (mantém assinatura usada   */
/*         por main.c — 7 args)                                   */
/* ============================================================== */
void letterbox_image_rgb_to_nchw(const uint8_t *img, int src_h, int src_w,
                                 float *out_nchw, int dst_h, int dst_w,
                                 LetterboxInfo *info)
{
    float r = fminf((float)dst_h / (float)src_h,
                    (float)dst_w / (float)src_w);
    int new_h = (int)roundf((float)src_h * r);
    int new_w = (int)roundf((float)src_w * r);
    int dw = (dst_w - new_w) / 2;
    int dh = (dst_h - new_h) / 2;

    const float pad = 114.0f / 255.0f;
    for (int i = 0; i < 3 * dst_h * dst_w; i++)
        out_nchw[i] = pad;

    for (int y = 0; y < new_h; y++) {
        float sy = (y + 0.5f) * (float)src_h / (float)new_h - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) { y0 = 0; fy = 0.0f; y1 = 0; }
        if (y1 >= src_h) { y1 = src_h - 1; fy = 0.0f; }

        for (int x = 0; x < new_w; x++) {
            float sx = (x + 0.5f) * (float)src_w / (float)new_w - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) { x0 = 0; fx = 0.0f; x1 = 0; }
            if (x1 >= src_w) { x1 = src_w - 1; fx = 0.0f; }

            int dy = y + dh;
            int dx = x + dw;

            for (int c = 0; c < 3; c++) {
                float v00 = img[((size_t)y0 * src_w + x0) * 3 + c];
                float v01 = img[((size_t)y0 * src_w + x1) * 3 + c];
                float v10 = img[((size_t)y1 * src_w + x0) * 3 + c];
                float v11 = img[((size_t)y1 * src_w + x1) * 3 + c];
                float top = v00 * (1.0f - fx) + v01 * fx;
                float bot = v10 * (1.0f - fx) + v11 * fx;
                float v = (top * (1.0f - fy) + bot * fy) / 255.0f;
                out_nchw[(size_t)c * dst_h * dst_w + (size_t)dy * dst_w + dx] = v;
            }
        }
    }

    info->r = r;
    info->dw = dw;
    info->dh = dh;
    info->new_h = new_h;
    info->new_w = new_w;
}

/* ============================================================== */
/* Fast-SCNN: letterbox COM normalização MEAN/STD por canal       */
/* ============================================================== */
void letterbox_image_rgb_to_nchw_norm(const uint8_t *img, int src_h, int src_w,
                                      float *out_nchw, int dst_h, int dst_w,
                                      const float mean[3], const float std[3],
                                      LetterboxInfo *info)
{
    float r = fminf((float)dst_h / (float)src_h,
                    (float)dst_w / (float)src_w);
    int new_h = (int)roundf((float)src_h * r);
    int new_w = (int)roundf((float)src_w * r);
    int dw = (dst_w - new_w) / 2;
    int dh = (dst_h - new_h) / 2;

    const float pad_v = 114.0f / 255.0f;
    for (int c = 0; c < 3; c++) {
        float pad_norm = (pad_v - mean[c]) / std[c];
        float *p = out_nchw + (size_t)c * dst_h * dst_w;
        for (int i = 0; i < dst_h * dst_w; i++) p[i] = pad_norm;
    }

    for (int y = 0; y < new_h; y++) {
        float sy = (y + 0.5f) * (float)src_h / (float)new_h - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) { y0 = 0; fy = 0.0f; y1 = 0; }
        if (y1 >= src_h) { y1 = src_h - 1; fy = 0.0f; }

        for (int x = 0; x < new_w; x++) {
            float sx = (x + 0.5f) * (float)src_w / (float)new_w - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) { x0 = 0; fx = 0.0f; x1 = 0; }
            if (x1 >= src_w) { x1 = src_w - 1; fx = 0.0f; }

            int dy = y + dh;
            int dx = x + dw;

            for (int c = 0; c < 3; c++) {
                float v00 = img[((size_t)y0 * src_w + x0) * 3 + c];
                float v01 = img[((size_t)y0 * src_w + x1) * 3 + c];
                float v10 = img[((size_t)y1 * src_w + x0) * 3 + c];
                float v11 = img[((size_t)y1 * src_w + x1) * 3 + c];
                float top = v00 * (1.0f - fx) + v01 * fx;
                float bot = v10 * (1.0f - fx) + v11 * fx;
                float v = (top * (1.0f - fy) + bot * fy) / 255.0f;
                v = (v - mean[c]) / std[c];
                out_nchw[(size_t)c * dst_h * dst_w + (size_t)dy * dst_w + dx] = v;
            }
        }
    }

    info->r = r;
    info->dw = dw;
    info->dh = dh;
    info->new_h = new_h;
    info->new_w = new_w;
}

/* ============================================================== */
/* Letterbox de máscara GT — compartilhado                        */
/* ============================================================== */
void letterbox_mask_nearest(const uint8_t *mask, int src_h, int src_w,
                            uint8_t *out, int dst_h, int dst_w,
                            const LetterboxInfo *info)
{
    memset(out, 0, (size_t)dst_h * dst_w);

    for (int y = 0; y < info->new_h; y++) {
        int sy = (int)((float)y / info->r);
        if (sy >= src_h) sy = src_h - 1;
        if (sy < 0) sy = 0;

        for (int x = 0; x < info->new_w; x++) {
            int sx = (int)((float)x / info->r);
            if (sx >= src_w) sx = src_w - 1;
            if (sx < 0) sx = 0;

            uint8_t v = (mask[(size_t)sy * src_w + sx] > 127) ? 1 : 0;
            out[(size_t)(y + info->dh) * dst_w + (x + info->dw)] = v;
        }
    }
}

/* ============================================================== */
/* Utilitários adicionais (preservados, fora do hot path)         */
/* ============================================================== */

void letterbox_image_rgb_to_nhwc_int8(const uint8_t *img, int src_h, int src_w,
                                      int8_t *out_nhwc, int dst_h, int dst_w,
                                      float scale, int zero_point,
                                      LetterboxInfo *info)
{
    float r = fminf((float)dst_h / (float)src_h,
                    (float)dst_w / (float)src_w);
    int new_h = (int)roundf((float)src_h * r);
    int new_w = (int)roundf((float)src_w * r);
    int dw = (dst_w - new_w) / 2;
    int dh = (dst_h - new_h) / 2;

    float pad_val = 114.0f / 255.0f;
    int8_t pad_q = (int8_t)(roundf(pad_val / scale) + zero_point);
    if (pad_q < -128) pad_q = -128;
    if (pad_q >  127) pad_q =  127;

    for (int i = 0; i < dst_h * dst_w * 3; i++)
        out_nhwc[i] = pad_q;

    for (int y = 0; y < new_h; y++) {
        float sy = (y + 0.5f) * (float)src_h / (float)new_h - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) { y0 = 0; fy = 0.0f; y1 = 0; }
        if (y1 >= src_h) { y1 = src_h - 1; fy = 0.0f; }

        for (int x = 0; x < new_w; x++) {
            float sx = (x + 0.5f) * (float)src_w / (float)new_w - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) { x0 = 0; fx = 0.0f; x1 = 0; }
            if (x1 >= src_w) { x1 = src_w - 1; fx = 0.0f; }

            int dy = y + dh;
            int dx = x + dw;

            for (int c = 0; c < 3; c++) {
                float v00 = img[((size_t)y0 * src_w + x0) * 3 + c];
                float v01 = img[((size_t)y0 * src_w + x1) * 3 + c];
                float v10 = img[((size_t)y1 * src_w + x0) * 3 + c];
                float v11 = img[((size_t)y1 * src_w + x1) * 3 + c];
                float top = v00 * (1.0f - fx) + v01 * fx;
                float bot = v10 * (1.0f - fx) + v11 * fx;
                float v = (top * (1.0f - fy) + bot * fy) / 255.0f;

                int q = (int)(roundf(v / scale) + zero_point);
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                out_nhwc[(dy * dst_w + dx) * 3 + c] = (int8_t)q;
            }
        }
    }

    info->r = r;
    info->dw = dw;
    info->dh = dh;
    info->new_h = new_h;
    info->new_w = new_w;
}

void resize_image_rgb_to_nhwc_int8_fscnn(
    const uint8_t *img, int src_h, int src_w,
    int8_t *out_nhwc, int dst_h, int dst_w,
    float scale, int zero_point,
    const float mean[3], const float std[3])
{
    for (int y = 0; y < dst_h; y++) {
        float sy = (y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0)      { y0 = 0; fy = 0.0f; y1 = 0; }
        if (y1 >= src_h) { y1 = src_h - 1; fy = 0.0f; }

        for (int x = 0; x < dst_w; x++) {
            float sx = (x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0)      { x0 = 0; fx = 0.0f; x1 = 0; }
            if (x1 >= src_w) { x1 = src_w - 1; fx = 0.0f; }

            for (int c = 0; c < 3; c++) {
                float v00 = img[((size_t)y0 * src_w + x0) * 3 + c];
                float v01 = img[((size_t)y0 * src_w + x1) * 3 + c];
                float v10 = img[((size_t)y1 * src_w + x0) * 3 + c];
                float v11 = img[((size_t)y1 * src_w + x1) * 3 + c];
                float top = v00 * (1.0f - fx) + v01 * fx;
                float bot = v10 * (1.0f - fx) + v11 * fx;
                float v   = (top * (1.0f - fy) + bot * fy) / 255.0f;

                float n = (v - mean[c]) / std[c];

                int q = (int)roundf(n / scale) + zero_point;
                if (q < -128) q = -128;
                if (q >  127) q =  127;

                out_nhwc[(y * dst_w + x) * 3 + c] = (int8_t)q;
            }
        }
    }
}

void resize_mask_nearest_direct(
    const uint8_t *mask, int src_h, int src_w,
    uint8_t *out, int dst_h, int dst_w)
{
    for (int y = 0; y < dst_h; y++) {
        int sy = (int)((float)y * (float)src_h / (float)dst_h);
        if (sy >= src_h) sy = src_h - 1;
        for (int x = 0; x < dst_w; x++) {
            int sx = (int)((float)x * (float)src_w / (float)dst_w);
            if (sx >= src_w) sx = src_w - 1;
            out[(size_t)y * dst_w + x] =
                (mask[(size_t)sy * src_w + sx] > 127) ? 1 : 0;
        }
    }
}
