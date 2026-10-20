# check_int8.py
import numpy as np, onnxruntime as ort, cv2, os, glob

H, W = 320, 544
MEAN = np.array([0.3551499061927887, 0.5187109166720711, 0.4940478866148916], dtype=np.float32)   # RGB
STD  = np.array([0.11498362917744831, 0.0765838378600218, 0.14597918539958313], dtype=np.float32)

sess_fp32 = ort.InferenceSession('fastscnn_fp32.onnx',     providers=['CPUExecutionProvider'])
sess_int8 = ort.InferenceSession('fastscnn_int8_qdq_ppmfp32.onnx', providers=['CPUExecutionProvider'])

def letterbox(img):
    sh, sw = img.shape[:2]
    r = min(H / sh, W / sw)
    nh, nw = int(round(sh * r)), int(round(sw * r))
    dh, dw = (H - nh) // 2, (W - nw) // 2
    canvas = np.full((H, W, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[dh:dh+nh, dw:dw+nw] = resized
    return canvas

def load(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = letterbox(img).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))[None].astype(np.float32)

imgs = sorted(glob.glob('/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision/train/images/*.jpg'))[50:70]
if not imgs:
    imgs = sorted(glob.glob('/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision/train/images/*.png'))[50:70]

diffs_logit = []
agreements  = []
for p in imgs:
    x = load(p)
    yf = sess_fp32.run(None, {'input': x})[0]
    yq = sess_int8.run(None, {'input': x})[0]
    diffs_logit.append(np.max(np.abs(yf - yq)))
    bf = (1/(1+np.exp(-yf)) > 0.5)
    bq = (1/(1+np.exp(-yq)) > 0.5)
    agreements.append((bf == bq).mean())

print(f"n imagens       : {len(imgs)}")
print(f"max logit diff  : {np.mean(diffs_logit):.4f}  (pior: {np.max(diffs_logit):.4f})")
print(f"acordo binário  : {np.mean(agreements)*100:.3f}%  (pior: {np.min(agreements)*100:.3f}%)")

# IoU direto entre FP32 e INT8 como proxy (o que importa na segmentação)
def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni   = np.logical_or(a, b).sum()
    return inter / (uni + 1e-7)

ious = []
for p in imgs:
    x = load(p)
    yf = (1/(1+np.exp(-sess_fp32.run(None, {'input': x})[0])) > 0.5)
    yq = (1/(1+np.exp(-sess_int8.run(None, {'input': x})[0])) > 0.5)
    ious.append(iou(yf, yq))

print(f"IoU FP32 vs INT8: {np.mean(ious):.4f}  (pior: {np.min(ious):.4f})")
