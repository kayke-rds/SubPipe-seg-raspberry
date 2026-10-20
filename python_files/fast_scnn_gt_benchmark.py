
import numpy as np, onnxruntime as ort, cv2, glob, os
from pathlib import Path

H, W = 320, 544
MEAN = np.array([0.3551499, 0.5187109, 0.49404788], dtype=np.float32)
STD  = np.array([0.11498363, 0.07658384, 0.14597918], dtype=np.float32)

sess_fp32 = ort.InferenceSession('fastscnn_fp32.onnx',              providers=['CPUExecutionProvider'])
sess_int8 = ort.InferenceSession('fastscnn_int8_qdq_ppmfp32.onnx', providers=['CPUExecutionProvider'])

def letterbox(img, fill=114):
    sh, sw = img.shape[:2]
    r = min(H/sh, W/sw)
    nh, nw = int(round(sh*r)), int(round(sw*r))
    dh, dw = (H-nh)//2, (W-nw)//2
    canvas = np.full((H, W, 3), fill, dtype=np.uint8)
    canvas[dh:dh+nh, dw:dw+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, (r, dw, dh, nh, nw)

def preprocess(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    lb, info = letterbox(img)
    lb = lb.astype(np.float32)/255.0
    lb = (lb - MEAN)/STD
    return np.transpose(lb, (2,0,1))[None].astype(np.float32), info

def gt_to_binary(mask_path, info, thr=127):
    """Carrega GT preservando a convenção do main.c: RGB com tubo vermelho."""
    r, dw, dh, nh, nw = info
    m = cv2.imread(mask_path, cv2.IMREAD_COLOR)   # BGR, como o main.c
    if m is None:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None: return None
        m_bin = (m > thr).astype(np.uint8)
    else:
        b, g, r_ch = m[:,:,0], m[:,:,1], m[:,:,2]
        # regra do main.c: pixel de tubo é "vermelho dominante"
        m_bin = ((r_ch > 60) & (r_ch > g + 30) & (r_ch > b + 30)).astype(np.uint8)
        # fallback: se nada detectado, tenta grayscale
        if m_bin.sum() == 0:
            mg = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            m_bin = (mg > thr).astype(np.uint8)

    # letterbox do GT: nearest + padding 0
    m_r = cv2.resize(m_bin, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((H, W), dtype=np.uint8)
    canvas[dh:dh+nh, dw:dw+nw] = m_r
    return canvas

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni   = np.logical_or(a, b).sum()
    return inter / uni if uni > 0 else float('nan')   # NaN quando ambos vazios

def dice(a, b):
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 2*inter / s if s > 0 else float('nan')

IMGS  = '/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision/train/images/'
MASKS = '/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision/train/masks/'

files = sorted(glob.glob(os.path.join(IMGS, '*.jpg')))[50:75] or \
        sorted(glob.glob(os.path.join(IMGS, '*.png')))[50:75]

rows = []
for f in files:
    mask_path = os.path.join(MASKS, Path(f).stem + '.png')
    if not os.path.exists(mask_path): continue

    x, info = preprocess(f)
    gt = gt_to_binary(mask_path, info)
    if gt is None: continue

    yf_logit = sess_fp32.run(None, {'input': x})[0][0,0]
    yq_logit = sess_int8.run(None, {'input': x})[0][0,0]

    yf = (1/(1+np.exp(-yf_logit)) > 0.5).astype(np.uint8)
    yq = (1/(1+np.exp(-yq_logit)) > 0.5).astype(np.uint8)

    rows.append({
        'stem': Path(f).stem,
        'gt_px': int(gt.sum()),
        'fp32_px': int(yf.sum()),
        'int8_px': int(yq.sum()),
        'iou_fp32': iou(yf, gt), 'dice_fp32': dice(yf, gt),
        'iou_int8': iou(yq, gt), 'dice_int8': dice(yq, gt),
    })

import pandas as pd
df = pd.DataFrame(rows)
print(df[['stem','gt_px','fp32_px','int8_px','iou_fp32','iou_int8']].to_string(index=False))

valid = df.dropna(subset=['iou_fp32','iou_int8'])
empty_gt = (df['gt_px'] == 0).sum()

print(f"\n=== resumo (n={len(df)}, GT vazio em {empty_gt}) ===")
print(f"logit MAE FP32↔INT8 : (fora do loop, rode separado)")
print(f"FP32 IoU vs GT      : {valid['iou_fp32'].mean():.4f}")
print(f"INT8 IoU vs GT      : {valid['iou_int8'].mean():.4f}")
print(f"Δ IoU (FP32-INT8)   : {(valid['iou_fp32']-valid['iou_int8']).mean():+.4f}")
