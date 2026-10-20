#!/usr/bin/env python3
"""Compara cls_out (INT8 raw) e a binarização final."""
import numpy as np

Y_ZP = 24          # cls_out (CONV[39].Y_zp no fastscnn_weights.h)

ref = np.fromfile('ref/T063.bin',   dtype=np.int8).reshape(40, 68)
c   = np.fromfile('dump_T063.bin',  dtype=np.int8).reshape(40, 68)

# 1) Logit bruto
d = np.abs(ref.astype(np.int32) - c.astype(np.int32))
print(f"=== cls_out (logits int8) ===")
print(f"  max|d| = {d.max()}")
print(f"  mean|d| = {d.mean():.4f}")
print(f"  exatos = {100.0*(d==0).sum()/d.size:.2f}%")
print(f"  |d|≤1 = {100.0*(d<=1).sum()/d.size:.2f}%")
print(f"  |d|≤2 = {100.0*(d<=2).sum()/d.size:.2f}%")

# 2) Binarização (sigmoid(logit) > 0.5 ⟺ logit > 0 ⟺ q > Y_zp)
b_ref = (ref.astype(np.int32) > Y_ZP).astype(np.uint8)
b_c   = (c.astype(np.int32)   > Y_ZP).astype(np.uint8)

inter = np.logical_and(b_ref, b_c).sum()
uni   = np.logical_or(b_ref, b_c).sum()
iou   = inter / uni if uni > 0 else 1.0
dice  = 2*inter / (b_ref.sum() + b_c.sum() + 1e-7) if (b_ref.sum()+b_c.sum()) > 0 else 1.0
agree = 100.0 * (b_ref == b_c).sum() / b_ref.size

print(f"\n=== binarização (q > Y_zp={Y_ZP}) ===")
print(f"  pixels positivos: ref={b_ref.sum()}  c={b_c.sum()}")
print(f"  acordo de pixel : {agree:.4f}%")
print(f"  IoU (ref vs c)  : {iou:.4f}")
print(f"  Dice (ref vs c) : {dice:.4f}")

# 3) Aviso se algum pixel mudou de classe
flips = np.argwhere(b_ref != b_c)
if len(flips) > 0:
    print(f"\n  {len(flips)} pixels com flip de classe (max|diff| nesses = "
          f"{d[tuple(flips.T)].max()}):")
    for y, x in flips[:10]:
        print(f"    [{y},{x}] ref={ref[y,x]:+4d}  c={c[y,x]:+4d}  "
              f"diff={ref[y,x]-c[y,x]:+d}")
else:
    print(f"\n  NENHUM pixel mudou de classe na binarização")
