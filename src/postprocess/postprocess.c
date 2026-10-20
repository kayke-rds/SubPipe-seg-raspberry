/* postprocess.c -- Decode e comparação unificado (YOLO26 + Fast-SCNN)
 *
 * YOLO26   : decode_mask_for_detection()       (6-arg, layout row-major
 *                                                 por detecção; é a versão
 *                                                 usada por main.c)
 *            decode_mask_for_detection_multi() (8-arg, layout channel-first
 *                                                 legado, preservado)
 * FastSCNN : fastscnn_binarize / fastscnn_binarize_bilinear
 *            decode_fastscnn_mask_int8 (variante simples)
 *
 * Compartilhado: mask_iou, mask_dice, mask_precision_recall,
 *                visualize_comparison, compose_output
 */
#include "postprocess.h"
#include <math.h>
#include <string.h>

/* ============================================================== */
/* YOLO26 — decode principal (layout per-detection: 38 floats)    */
/* ============================================================== */
void decode_mask_for_detection(const float *proto, const float *output0,
                               int det, int img_h, int img_w,
                               uint8_t *out_mask)
{
    const int PROTO_C = 32, PROTO_H = 80, PROTO_W = 80;

    /* layout: [x1, y1, x2, y2, score_pipe, score_bg, coeff_0..31] */
    const float *row    = output0 + (size_t)det * 38;
    const float *coeffs = row + 6;

    static float mask80[80 * 80];
    for (int i = 0; i < PROTO_H * PROTO_W; i++)
        mask80[i] = 0.0f;

    for (int k = 0; k < PROTO_C; k++) {
        float c = coeffs[k];
        if (c == 0.0f) continue;
        const float *p = proto + (size_t)k * PROTO_H * PROTO_W;
        for (int i = 0; i < PROTO_H * PROTO_W; i++)
            mask80[i] += c * p[i];
    }
    for (int i = 0; i < PROTO_H * PROTO_W; i++)
        mask80[i] = 1.0f / (1.0f + expf(-mask80[i]));

    float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
    float scale_y = (float)PROTO_H / (float)img_h;
    float scale_x = (float)PROTO_W / (float)img_w;

    for (int y = 0; y < img_h; y++) {
        float src_y = ((float)y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(src_y);
        float dy = src_y - (float)y0;
        int y1i = y0 + 1;
        if (y0 < 0) { y0 = 0; dy = 0.0f; y1i = 0; }
        if (y1i >= PROTO_H) { y1i = PROTO_H - 1; dy = 0.0f; }

        for (int x = 0; x < img_w; x++) {
            float src_x = ((float)x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(src_x);
            float dx = src_x - (float)x0;
            int x1i = x0 + 1;
            if (x0 < 0) { x0 = 0; dx = 0.0f; x1i = 0; }
            if (x1i >= PROTO_W) { x1i = PROTO_W - 1; dx = 0.0f; }

            float v00 = mask80[(size_t)y0  * PROTO_W + x0 ];
            float v01 = mask80[(size_t)y0  * PROTO_W + x1i];
            float v10 = mask80[(size_t)y1i * PROTO_W + x0 ];
            float v11 = mask80[(size_t)y1i * PROTO_W + x1i];
            float top = v00 * (1.0f - dx) + v01 * dx;
            float bot = v10 * (1.0f - dx) + v11 * dx;
            float val = top * (1.0f - dy) + bot * dy;

            if (x < x1 || x > x2 || y < y1 || y > y2)
                val = 0.0f;

            out_mask[(size_t)y * img_w + x] = (val > 0.5f) ? 1 : 0;
        }
    }
}

/* ============================================================== */
/* YOLO26 — decode alternativo (channel-first: 4+classes+32)      */
/* Preservado de versões anteriores; NÃO é usado pelo bench       */
/* unificado porque yolo_output0() entrega layout per-detection.  */
/* ============================================================== */
void decode_mask_for_detection_multi(const float *proto, const float *output0,
                                     int num_classes, int num_anchors,
                                     int det, int img_h, int img_w,
                                     uint8_t *out_mask)
{
    const int PROTO_C = 32, PROTO_H = 80, PROTO_W = 80;

    const float *coeffs = output0 + (4 + num_classes) * num_anchors + det;

    static float mask80[80 * 80];
    for (int i = 0; i < PROTO_H * PROTO_W; i++)
        mask80[i] = 0.0f;

    for (int k = 0; k < PROTO_C; k++) {
        float c = coeffs[k * num_anchors];
        if (c == 0.0f) continue;
        const float *p = proto + (size_t)k * PROTO_H * PROTO_W;
        for (int i = 0; i < PROTO_H * PROTO_W; i++)
            mask80[i] += c * p[i];
    }
    for (int i = 0; i < PROTO_H * PROTO_W; i++)
        mask80[i] = 1.0f / (1.0f + expf(-mask80[i]));

    float cx = output0[0 * num_anchors + det] * (float)img_w;
    float cy = output0[1 * num_anchors + det] * (float)img_h;
    float bw = output0[2 * num_anchors + det] * (float)img_w;
    float bh = output0[3 * num_anchors + det] * (float)img_h;

    float x1 = cx - bw * 0.5f;
    float y1 = cy - bh * 0.5f;
    float x2 = cx + bw * 0.5f;
    float y2 = cy + bh * 0.5f;

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > img_w - 1) x2 = img_w - 1;
    if (y2 > img_h - 1) y2 = img_h - 1;

    float scale_y = (float)PROTO_H / (float)img_h;
    float scale_x = (float)PROTO_W / (float)img_w;

    for (int y = 0; y < img_h; y++) {
        if ((float)y < y1 || (float)y > y2) {
            memset(out_mask + (size_t)y * img_w, 0, img_w);
            continue;
        }
        float src_y = ((float)y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(src_y);
        float dy = src_y - (float)y0;
        int y1i = y0 + 1;
        if (y0 < 0) { y0 = 0; dy = 0.0f; y1i = 0; }
        if (y1i >= PROTO_H) { y1i = PROTO_H - 1; dy = 0.0f; }

        for (int x = 0; x < img_w; x++) {
            if ((float)x < x1 || (float)x > x2) {
                out_mask[(size_t)y * img_w + x] = 0;
                continue;
            }
            float src_x = ((float)x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(src_x);
            float dx = src_x - (float)x0;
            int x1i = x0 + 1;
            if (x0 < 0) { x0 = 0; dx = 0.0f; x1i = 0; }
            if (x1i >= PROTO_W) { x1i = PROTO_W - 1; dx = 0.0f; }

            float v00 = mask80[(size_t)y0  * PROTO_W + x0 ];
            float v01 = mask80[(size_t)y0  * PROTO_W + x1i];
            float v10 = mask80[(size_t)y1i * PROTO_W + x0 ];
            float v11 = mask80[(size_t)y1i * PROTO_W + x1i];
            float top = v00 * (1.0f - dx) + v01 * dx;
            float bot = v10 * (1.0f - dx) + v11 * dx;
            float val = top * (1.0f - dy) + bot * dy;

            out_mask[(size_t)y * img_w + x] = (val > 0.5f) ? 1 : 0;
        }
    }
}

int find_best_detection(const float *output0, int num_classes, int num_anchors,
                        int target_class)
{
    int best_det = -1;
    float best_score = -1.0f;

    for (int a = 0; a < num_anchors; a++) {
        float score;
        if (target_class >= 0) {
            score = output0[(4 + target_class) * num_anchors + a];
        } else {
            score = -1.0f;
            for (int c = 0; c < num_classes; c++) {
                float s = output0[(4 + c) * num_anchors + a];
                if (s > score) score = s;
            }
        }
        if (score > best_score) {
            best_score = score;
            best_det = a;
        }
    }
    return best_det;
}

/* ============================================================== */
/* Métricas compartilhadas                                        */
/* ============================================================== */
float mask_iou(const uint8_t *a, const uint8_t *b, int n) {
    int inter = 0, uni = 0;
    for (int i = 0; i < n; i++) {
        int va = a[i] != 0, vb = b[i] != 0;
        inter += (va && vb);
        uni   += (va || vb);
    }
    return uni == 0 ? 0.0f : (float)inter / (float)uni;
}

float mask_dice(const uint8_t *a, const uint8_t *b, int n) {
    int inter = 0, sum = 0;
    for (int i = 0; i < n; i++) {
        int va = a[i] != 0, vb = b[i] != 0;
        inter += (va && vb);
        sum   += va + vb;
    }
    return sum == 0 ? 1.0f : 2.0f * (float)inter / (float)sum;
}

void mask_precision_recall(const uint8_t *pred, const uint8_t *gt, int n,
                           float *precision, float *recall)
{
    int tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < n; i++) {
        int p = pred[i] != 0, g = gt[i] != 0;
        if (p && g) tp++;
        else if (p && !g) fp++;
        else if (!p && g) fn++;
    }
    *precision = (tp + fp) > 0 ? (float)tp / (float)(tp + fp) : 0.0f;
    *recall    = (tp + fn) > 0 ? (float)tp / (float)(tp + fn) : 0.0f;
}

void visualize_comparison(const uint8_t *pred, const uint8_t *gt,
                          int w, int h, uint8_t *rgb_out)
{
    for (int i = 0; i < w * h; i++) {
        int p = pred[i] != 0, g = gt[i] != 0;
        uint8_t *px = rgb_out + (size_t)i * 3;
        if (p && g)      { px[0] =   0; px[1] = 255; px[2] =   0; }
        else if (p && !g){ px[0] = 255; px[1] =   0; px[2] =   0; }
        else if (!p && g){ px[0] =   0; px[1] =   0; px[2] = 255; }
        else             { px[0] =   0; px[1] =   0; px[2] =   0; }
    }
}

/* ============================================================== */
/* Fast-SCNN — binarização por zero-point                         */
/* ============================================================== */
void fastscnn_binarize(const int8_t *logits, int n, int y_zp, uint8_t *out_mask)
{
    for (int i = 0; i < n; i++)
        out_mask[i] = ((int)logits[i] > y_zp) ? 1 : 0;
}

void fastscnn_binarize_bilinear(const int8_t *logits, int h, int w, int y_zp,
                                uint8_t *mask, int oh, int ow)
{
    for (int oy = 0; oy < oh; oy++) {
        float sy = (oy + 0.5f) * (float)h / (float)oh - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) { y0 = 0; fy = 0; y1 = 0; }
        if (y1 >= h) { y1 = h - 1; fy = 0; }

        for (int ox = 0; ox < ow; ox++) {
            float sx = (ox + 0.5f) * (float)w / (float)ow - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) { x0 = 0; fx = 0; x1 = 0; }
            if (x1 >= w) { x1 = w - 1; fx = 0; }

            float v = ((float)logits[y0*w+x0] * (1-fx)
                     + (float)logits[y0*w+x1] * fx) * (1-fy)
                    + ((float)logits[y1*w+x0] * (1-fx)
                     + (float)logits[y1*w+x1] * fx) * fy;

            int q = (int)lrintf(v);
            mask[oy*ow+ox] = (q > y_zp) ? 1 : 0;
        }
    }
}

/* Variante "simples" (por pixel, sem bilinear) preservada */
void decode_fastscnn_mask_int8(
    const int8_t *output, int h, int w,
    float scale, int zero_point,
    float threshold,
    uint8_t *out_mask)
{
    const int n = h * w;
    for (int i = 0; i < n; i++) {
        float logit = ((float)output[i] - (float)zero_point) * scale;
        out_mask[i] = (logit > threshold) ? 1 : 0;
    }
}

/* ============================================================== */
/* Composição visual 2W x 2H                          */
/* ============================================================== */
void compose_output(const uint8_t *img_rgb, int sw, int sh,
                    const uint8_t *mask_gt, const uint8_t *mask_pred,
                    int w, int h, const LetterboxInfo *lb,
                    uint8_t *out_rgb)
{
    const int tw = 2 * w;
    const int th = 2 * h;

    /* Limpa a tela inteira (fundo preto) */
    memset(out_rgb, 0, (size_t)tw * th * 3);

    /* --- Topo-esquerda: GT mask --- */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t v = mask_gt[(size_t)y * w + x] ? 255 : 0;
            uint8_t *p = out_rgb + ((size_t)y * tw + x) * 3;
            p[0] = v; p[1] = v; p[2] = v;
        }
    }

    /* --- Topo-direita: Pred mask --- */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t v = mask_pred[(size_t)y * w + x] ? 255 : 0;
            uint8_t *p = out_rgb + ((size_t)y * tw + (w + x)) * 3;
            p[0] = v; p[1] = v; p[2] = v;
        }
    }

    /* --- Base-esquerda: Imagem Original Redimensionada --- */
    if (img_rgb && lb) {
        /* Preenche com cinza escuro (padrão do letterbox) */
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint8_t *p = out_rgb + ((size_t)(h + y) * tw + x) * 3;
                p[0] = 40; p[1] = 40; p[2] = 40;
            }
        }

        /* Desenha a imagem usando as mesmas proporções (letterbox) */
        for (int y = 0; y < lb->new_h; y++) {
            int sy = (int)((float)y / lb->r);
            if (sy >= sh) sy = sh - 1;
            if (sy < 0) sy = 0;

            for (int x = 0; x < lb->new_w; x++) {
                int sx = (int)((float)x / lb->r);
                if (sx >= sw) sx = sw - 1;
                if (sx < 0) sx = 0;

                int dy = y + lb->dh;
                int dx = x + lb->dw;
                if (dy >= 0 && dy < h && dx >= 0 && dx < w) {
                    uint8_t *p = out_rgb + ((size_t)(h + dy) * tw + dx) * 3;
                    p[0] = img_rgb[((size_t)sy * sw + sx) * 3 + 0];
                    p[1] = img_rgb[((size_t)sy * sw + sx) * 3 + 1];
                    p[2] = img_rgb[((size_t)sy * sw + sx) * 3 + 2];
                }
            }
        }
    }

    /* --- Base-direita: Overlay colorido --- */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int g = mask_gt  [(size_t)y * w + x] != 0;
            int p = mask_pred[(size_t)y * w + x] != 0;
            uint8_t r, gg, b;
            if      ( p &&  g) { r=  0; gg=255; b=  0; }  /* TP verde   */
            else if ( p && !g) { r=255; gg=  0; b=  0; }  /* FP vermelho */
            else if (!p &&  g) { r=  0; gg=  0; b=255; }  /* FN azul    */
            else               { r=  0; gg=  0; b=  0; }  /* TN preto   */

            uint8_t *px = out_rgb +
                ((size_t)(h + y) * tw + (w + x)) * 3;
            px[0] = r; px[1] = gg; px[2] = b;
        }
    }
}
