# compare_real.py
import numpy as np, cv2, glob, os, subprocess
from pathlib import Path

H, W = 320, 544
MEAN = np.array([0.3551499, 0.5187109, 0.49404788], dtype=np.float32)
STD  = np.array([0.11498363, 0.07658384, 0.14597918], dtype=np.float32)

import onnxruntime as ort
sess = ort.InferenceSession('fastscnn_int8_qdq_ppmfp32.onnx',
                            providers=['CPUExecutionProvider'])

Y_ZP = 24   # cls_out
Y_SCALE = 0.09853581

def letterbox(img):
    sh, sw = img.shape[:2]
    r = min(H/sh, W/sw)
    nh, nw = int(round(sh*r)), int(round(sw*r))
    dh, dw = (H-nh)//2, (W-nw)//2
    canvas = np.full((H, W, 3), 114, dtype=np.uint8)
    canvas[dh:dh+nh, dw:dw+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas

IMGS = sorted(glob.glob('/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision/train/images/*.jpg'))[50:75]

ious, agrees = [], []
for i, path in enumerate(IMGS):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    lb = letterbox(img).astype(np.float32) / 255.0
    x = ((lb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    x.tofile('test_forward_input.bin')

    # ONNX
    y_onnx = sess.run(None, {'input': x})[0][0, 0, ::8, ::8]   # subsample 40x68
    # C
    # Sanity: o arquivo que o C vai ler deve ter o mesmo hash do que salvamos
    import hashlib
    h_py = hashlib.md5(x.tobytes()).hexdigest()
    h_file = hashlib.md5(open('test_forward_input.bin', 'rb').read()).hexdigest()
    assert h_py == h_file, f"input não foi salvo: py={h_py[:8]} file={h_file[:8]}"
    subprocess.run(['./test_forward_debug'], check=True, capture_output=True)
    y_c = np.fromfile('dump_T063.bin', dtype=np.int8).reshape(40, 68)

    # binariza
    b_onnx = (np.rint(y_onnx / Y_SCALE).astype(np.int32) + Y_ZP > Y_ZP).astype(np.uint8)
    b_c    = (y_c.astype(np.int32) > Y_ZP).astype(np.uint8)

    inter = np.logical_and(b_onnx, b_c).sum()
    uni   = np.logical_or(b_onnx, b_c).sum()
    iou   = inter / uni if uni > 0 else 1.0
    agree = 100.0 * (b_onnx == b_c).sum() / b_onnx.size

    ious.append(iou)
    agrees.append(agree)
    print(f"{Path(path).name:20s} iou={iou:.4f}  agree={agree:6.3f}%  "
          f"pos_onnx={b_onnx.sum():4d} pos_c={b_c.sum():4d}")

print(f"\nMÉDIA IoU  = {np.mean(ious):.4f}")
print(f"MÉDIA agree= {np.mean(agrees):.3f}%")
print(f"MIN   IoU  = {np.min(ious):.4f}")
