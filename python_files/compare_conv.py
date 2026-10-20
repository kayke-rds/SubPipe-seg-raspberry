#!/usr/bin/env python3
"""Compara test_output_ref.bin vs test_output_c.bin."""
import numpy as np

# lê shapes do meta
meta = {}
with open('test_meta.txt') as f:
    for line in f:
        k, v = line.split()
        meta[k] = int(v)

Cout, Hout, Wout = meta['Cout'], meta['Hout'], meta['Wout']
print(f"conv_idx={meta['conv_idx']}  "
      f"Cin={meta['Cin']} Cout={Cout}  in={meta['Hin']}x{meta['Win']}  "
      f"out={Hout}x{Wout}")

Y_ref = np.fromfile('test_output_ref.bin', dtype=np.int8).reshape(Cout, Hout, Wout)
Y_c   = np.fromfile('test_output_c.bin',   dtype=np.int8).reshape(Cout, Hout, Wout)

diff     = Y_ref.astype(np.int32) - Y_c.astype(np.int32)
abs_diff = np.abs(diff)

print(f"Y_ref: min={Y_ref.min()} max={Y_ref.max()} mean={Y_ref.mean():.4f}")
print(f"Y_c  : min={Y_c.min()}  max={Y_c.max()}  mean={Y_c.mean():.4f}")
print()
print(f"max |diff| : {abs_diff.max()}")
print(f"mean |diff|: {abs_diff.mean():.6f}")
print(f"exactos    : {(diff==0).sum()}/{diff.size} "
      f"= {100.0*(diff==0).sum()/diff.size:.3f}%")
print(f"|diff| ≤ 1 : {(abs_diff<=1).sum()}/{diff.size} "
      f"= {100.0*(abs_diff<=1).sum()/diff.size:.3f}%")

if abs_diff.max() > 0:
    idx = np.argwhere(abs_diff > 0)[:15]
    print(f"\nprimeiras {len(idx)} posições com diff != 0:")
    for c, y, x in idx:
        print(f"  [{c},{y},{x}] ref={Y_ref[c,y,x]:4d}  "
              f"c={Y_c[c,y,x]:4d}  diff={diff[c,y,x]:+d}")
