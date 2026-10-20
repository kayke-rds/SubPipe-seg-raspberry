#!/usr/bin/env python3
"""Gera input sintético e referência ONNX para comparação."""
import numpy as np
import onnxruntime as ort

H, W = 320, 544
MEAN = np.array([0.3551499, 0.5187109, 0.49404788], dtype=np.float32)
STD  = np.array([0.11498363, 0.07658384, 0.14597918], dtype=np.float32)

# input sintético normalizado
rng = np.random.default_rng(7)
img = rng.random((H, W, 3), dtype=np.float32) * 0.6 + 0.2
img = (img - MEAN) / STD
x = np.transpose(img, (2, 0, 1))[None].astype(np.float32)

x.tofile('test_forward_input.bin')

# ONNX FP32
sess = ort.InferenceSession('fastscnn_int8_qdq_ppmfp32.onnx',
                            providers=['CPUExecutionProvider'])
y = sess.run(None, {'input': x})[0][0, 0]   # [H, W] FP32

# quantiza com a escala da última conv (cls_out)
FINAL_SCALE = 0.09853580594062805    # ajustar com o Y_scale de FASTSCNN_CONV[39]
FINAL_ZP    = 24             # ajustar com o Y_zp

y_q = np.clip(np.rint(y / FINAL_SCALE).astype(np.int32) + FINAL_ZP, -128, 127).astype(np.int8)
y_q.tofile('test_forward_ref.bin')

print(f"input  : {x.shape}  range=[{x.min():.3f}, {x.max():.3f}]")
print(f"ref i8 : {y_q.shape}  range=[{y_q.min()}, {y_q.max()}]  mean={y_q.mean():.2f}")
