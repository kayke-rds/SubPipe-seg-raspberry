#!/usr/bin/env python3
"""
compare_with_onnxruntime.py

Diagnóstico completo do forward em C contra o ONNX Runtime.

Compara, no MESMO input (o letterbox que o C realmente produziu), as saídas do
binário yolo_seg contra as do ORT, e reporta:

  1. Sanidade do letterbox: input.bin do C vs letterbox Python lado a lado.
  2. Divergência global em output0 e output1 (norma máx/mean, relativa).
  3. Estatística da cabeça: distribuição do score e cluster do top-300.
  4. Veredito localizado: backbone/neck, proto, ou cabeça.

Uso:
    mkdir -p dump
    ./yolo_seg img_test.jpg mask_test.png yolo26n_seg_weights.bin
    python3 compare_with_onnxruntime.py best.onnx dump/ [--img img_test.jpg]

Requer: numpy, onnxruntime, pillow (opcional, só para checar letterbox).
"""
import sys, os, argparse
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("pip install onnxruntime")
try:
    from PIL import Image
except ImportError:
    Image = None

# ---------- shapes (do yolo26n_seg_layout.h) ----------
IN_C, IN_H, IN_W           = 3, 320, 320
OUT0_ROWS, OUT0_COLS       = 300, 38
PROTO_C, PROTO_H, PROTO_W  = 32, 80, 80

# limiar global de "OK". float32 acumula erro ~1e-5; ONNX e C usam a mesma
# ordem de operações (a menos de reassociação em convs), então 1e-3 é folgado.
TOL = 1e-3


# ================== helpers ==================

def load_bin(path, shape):
    a = np.fromfile(path, dtype="<f4")
    n = int(np.prod(shape))
    if a.size != n:
        sys.exit(f"{path}: esperado {n} floats, achei {a.size}")
    return a.reshape(shape)


def stats(name, x):
    x = np.asarray(x, dtype=np.float64)
    print(f"  {name:<24} min={x.min(): .5f}  max={x.max(): .5f}  "
          f"mean={x.mean(): .5f}  std={x.std(): .5f}  n={x.size}")


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) -
                               np.asarray(b, np.float64))))


def compare_arrays(c, o, tag):
    """c, o: arrays com mesma shape. Reporta |Δ| e |Δ| relativo."""
    diff = np.abs(c - o)
    rel  = diff / (np.maximum(np.abs(c), np.abs(o)) + 1e-12)
    print(f"  {tag}:  max|Δ|={diff.max():.4e}  mean|Δ|={diff.mean():.4e}  "
          f"max_rel={rel.max():.3e}  p99|Δ|={np.percentile(diff, 99):.4e}")
    return float(diff.max())


# ---- réplica exata de preprocess.c:letterbox_image_rgb_to_nchw ----
def letterbox_rgb_to_nchw(img_rgb, dst_h=320, dst_w=320):
    """
    img_rgb: (src_h, src_w, 3) uint8, canal 0 = R.
    Retorna (3, dst_h, dst_w) float32 normalizado por 255, com padding 114/255.
    """
    src_h, src_w, _ = img_rgb.shape
    r      = min(dst_h / src_h, dst_w / src_w)
    new_h  = int(round(src_h * r))
    new_w  = int(round(src_w * r))
    dw     = (dst_w - new_w) // 2
    dh     = (dst_h - new_h) // 2

    out = np.full((3, dst_h, dst_w), 114.0 / 255.0, dtype=np.float32)

    ys = (np.arange(new_h) + 0.5) * src_h / new_h - 0.5
    xs = (np.arange(new_w) + 0.5) * src_w / new_w - 0.5

    y0 = np.floor(ys).astype(int); fy = ys - y0; y1 = y0 + 1
    x0 = np.floor(xs).astype(int); fx = xs - x0; x1 = x0 + 1

    y0c = np.clip(y0, 0, src_h - 1); y1c = np.clip(y1, 0, src_h - 1)
    x0c = np.clip(x0, 0, src_w - 1); x1c = np.clip(x1, 0, src_w - 1)
    fy[y0 < 0] = 0.0;  fy[y1 >= src_h] = 0.0
    fx[x0 < 0] = 0.0;  fx[x1 >= src_w] = 0.0

    img = img_rgb.astype(np.float32)
    v00 = img[np.ix_(y0c, x0c)]   # (new_h, new_w, 3)
    v01 = img[np.ix_(y0c, x1c)]
    v10 = img[np.ix_(y1c, x0c)]
    v11 = img[np.ix_(y1c, x1c)]
    fy3 = fy[:, None, None]
    fx3 = fx[None, :, None]
    top = v00 * (1 - fx3) + v01 * fx3
    bot = v10 * (1 - fx3) + v11 * fx3
    v   = (top * (1 - fy3) + bot * fy3) / 255.0
    v   = np.transpose(v, (2, 0, 1))                 # (3, new_h, new_w)
    out[:, dh:dh + new_h, dw:dw + new_w] = v
    return out, (r, dw, dh, new_h, new_w)


# ---- análise da cabeça ----
def analyze_head(out0, tag):
    """out0: (300, 38). Colunas: xyxy(4) | score(1) | cls(1) | coef(32)."""
    print(f"--- {tag}: output0 (top-300) ---")
    score = out0[:, 4]
    stats(f"{tag} score", score)
    print(f"  top-5 scores:  {np.array2string(score[:5], precision=6)}")
    print(f"  bottom-5:      {np.array2string(score[-5:], precision=6)}")
    print(f"  det[0] box:    {np.array2string(out0[0, :4], precision=3)}")
    print(f"  det[0] cls:    {out0[0, 5]:.3f}")

    # sinal típico de cls4 plano: os 300 primeiros scores praticamente idênticos
    spread = float(score[:10].max() - score[:10].min())
    if spread < 1e-3:
        print(f"  !! AVISO: os 10 primeiros scores estão colados "
              f"(spread={spread:.2e}). Sinal de nível com cls plano.")


def rank_corr(a, b):
    """Correlação de postos entre a e b (útil para TopK)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ================== main ==================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="caminho do best.onnx")
    ap.add_argument("dump", help="pasta com input.bin, out0.bin, out1.bin")
    ap.add_argument("--img", default=None,
                    help="imagem original (opcional; checa o letterbox)")
    args = ap.parse_args()

    d = args.dump

    print(f"== 1) dumps do C em {d}/ ==")
    c_in   = load_bin(os.path.join(d, "input.bin"), (IN_C, IN_H, IN_W))
    c_out0 = load_bin(os.path.join(d, "out0.bin"),  (OUT0_ROWS, OUT0_COLS))
    c_out1 = load_bin(os.path.join(d, "out1.bin"),  (PROTO_C, PROTO_H, PROTO_W))
    stats("C input", c_in)
    stats("C out1 (proto)", c_out1)

    # -------- 2) sanidade do letterbox --------
    if args.img and Image is not None:
        print(f"\n== 2) letterbox: Python vs C (imagem {args.img}) ==")
        img = np.asarray(Image.open(args.img).convert("RGB"))
        py_in, info = letterbox_rgb_to_nchw(img)
        print(f"  r={info[0]:.6f}  dw={info[1]}  dh={info[2]}  "
              f"new={info[3]}x{info[4]}")
        m = max_abs_diff(py_in, c_in)
        print(f"  max|Δ| letterbox = {m:.3e}   "
              f"{'OK' if m < 1e-6 else 'DIVERGE!'}")
        if m >= 1e-6:
            # localiza a primeira divergência
            bad = np.argwhere(np.abs(py_in - c_in) > 1e-5)
            if bad.size:
                ch, yy, xx = bad[0]
                print(f"  primeira divergência em (c={ch}, y={yy}, x={xx}): "
                      f"C={c_in[ch,yy,xx]:.6f}  Py={py_in[ch,yy,xx]:.6f}")
    else:
        print("\n== 2) letterbox: pulado (sem --img ou sem PIL) ==")

    # -------- 3) ONNX Runtime no MESMO input --------
    print(f"\n== 3) ONNX Runtime em {args.onnx} ==")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    feeds = {in_name: c_in[None, ...]}
    o_out0, o_out1 = sess.run(None, feeds)
    o_out0 = o_out0[0]
    o_out1 = o_out1[0]
    stats("ORT input", c_in)
    stats("ORT out1 (proto)", o_out1)

    # -------- 4) comparação C vs ORT --------
    print(f"\n== 4) C vs ORT ==")
    d0 = compare_arrays(c_out0, o_out0, "out0 (cabeça)")
    d1 = compare_arrays(c_out1, o_out1, "out1 (proto) ")
    print(f"  tolerância global: {TOL:.0e}")
    print(f"  out0: {'OK' if d0 < TOL else 'DIVERGE'}")
    print(f"  out1: {'OK' if d1 < TOL else 'DIVERGE'}")

    # se out0 divergir, ver se é só reordenação de empate
    if d0 >= TOL:
        cc = rank_corr(c_out0[:, 4], o_out0[:, 4])
        print(f"  rank corr(score C, score ORT) = {cc:.6f}")
        # diferença nas coordenadas
        dd_box = np.abs(c_out0[:, :4] - o_out0[:, :4]).max()
        dd_coef = np.abs(c_out0[:, 6:] - o_out0[:, 6:]).max()
        print(f"  max|Δ| box = {dd_box:.4e}   max|Δ| coef = {dd_coef:.4e}")

    # -------- 5) distribuição da cabeça --------
    print()
    analyze_head(c_out0, "C")
    print()
    analyze_head(o_out0, "ORT")

    # -------- 6) veredito localizado --------
    print(f"\n== 5) veredito ==")
    if d1 < TOL and d0 < TOL:
        print("  C é FIEL ao ONNX no mesmo input.")
        print("  → O forward não tem bug de implementação. Se a predição é ruim,")
        print("    o problema está no input (imagem fora do domínio de treino,")
        print("    letterbox errado, ordem de canais) ou o modelo realmente não")
        print("    detecta nada aqui. Compare com um output de referência do")
        print("    Ultralytics no MESMO arquivo (yolo segment predict ...).")
    elif d1 >= TOL and d0 >= TOL:
        print("  C DIVERGE em out0 e out1.")
        print("  → Suspeitar de backbone/neck/SPPF/C2PSA.")
        print("    Como o Proto lê P3/P4/P5 do neck, um erro pequeno no neck")
        print("    pode contaminar out1. Instrumente print_stats a cada bloco")
        print("    (m0..m22) e compare com este script rodando ORT em modo")
        print("    de saídas intermediárias.")
    elif d1 >= TOL:
        print("  C DIVERGE só em out1 (proto).")
        print("  → Suspeitar de Proto26: feat_refine, resize ×2/×4, feat_fuse,")
        print("    cv1, ConvT2x2s2, cv2, cv3.")
    else:
        print("  C DIVERGE só em out0 (cabeça).")
        print("  → Suspeitar de head_branch (box/cls/coef por nível), decode")
        print("    (Slice/Sub/Add/Mul por stride) ou TopK.")
        print("    Como os 3 níveis compartilham a mesma implementação de")
        print("    head_branch, olhe cls3/cls4/cls5 individualmente: se só")
        print("    um nível está errado, é o cfg daquele nível, não a função.")

    return 0 if (d0 < TOL and d1 < TOL) else 1


if __name__ == "__main__":
    sys.exit(main())
