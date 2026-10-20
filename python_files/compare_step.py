#!/usr/bin/env python3
import os, re
import numpy as np

def parse_mem_h(path):
    txt = open(path).read()
    m_arr = re.search(r'FASTSCNN_TENSOR\[T_COUNT\]\s*=\s*\{(.*?)\};', txt, re.S)
    body = m_arr.group(1)
    out = {}
    idx = 0
    for m in re.finditer(
        r'\{\s*(\d+)u\s*,\s*(\d+)u\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
        r'\s*DT_(\w+)\s*\}\s*,\s*/\*\s*(\S+)\s*\*/',
        body):
        off, size, c, h, w, dt, name = m.groups()
        out[name] = dict(tid=idx, c=int(c), h=int(h), w=int(w), dtype=dt)
        idx += 1
    return out

tensors = parse_mem_h('fastscnn_mem.h')

ORDER = [
    'l2d_conv','l2d_ds1_dw','l2d_ds1_pw','l2d_ds2_dw','l2d_ds2_pw',
    'b1_0_expand','b1_0_dw','b1_0_project',
    'b1_1_expand','b1_1_dw','b1_1_project','b1_1_add',
    'b1_2_expand','b1_2_dw','b1_2_project','b1_2_add',
    'b2_0_expand','b2_0_dw','b2_0_project',
    'b2_1_expand','b2_1_dw','b2_1_project','b2_1_add',
    'b2_2_expand','b2_2_dw','b2_2_project','b2_2_add',
    'b3_0_expand','b3_0_dw','b3_0_project',
    'b3_1_expand','b3_1_dw','b3_1_project','b3_1_add',
    'b3_2_expand','b3_2_dw','b3_2_project','b3_2_add',
    'ff_up',
    'ff_dw','ff_low','ff_high','ff_add',
    'cls_ds1_dw','cls_ds1_pw','cls_ds2_dw','cls_ds2_pw','cls_out',
]

print(f"{'tensor':16s} {'tid':>4s} {'shape':>14s} {'max|d|':>7s} {'exatos':>9s} {'≤1':>9s}")
print('-' * 66)

first_bad = None
for name in ORDER:
    t = tensors.get(name)
    if t is None:
        print(f"  {name}: ausente no mem.h")
        continue
    tid = t['tid']
    ref = f'ref/T{tid:03d}.bin'
    cc  = f'dump_T{tid:03d}.bin'
    if not os.path.exists(ref):
        print(f"  {name:16s} {tid:4d}  ref AUSENTE ({ref})")
        continue
    if not os.path.exists(cc):
        print(f"  {name:16s} {tid:4d}  dump AUSENTE ({cc})")
        continue
    a = np.fromfile(ref, dtype=np.int8)
    b = np.fromfile(cc,  dtype=np.int8)
    if a.size != b.size:
        print(f"  {name:16s} {tid:4d}  SIZE {a.size} vs {b.size}")
        if first_bad is None: first_bad = (name, tid, 'size')
        continue
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    exact = 100.0 * (d == 0).sum() / d.size
    le1   = 100.0 * (d <= 1).sum() / d.size
    flag  = '' if d.max() <= 1 else ' <<<'
    print(f"  {name:16s} {tid:4d} {t['c']:>3d}x{t['h']:>3d}x{t['w']:>3d} "
          f"{d.max():7d} {exact:8.2f}% {le1:8.2f}%{flag}")
    if d.max() > 1 and first_bad is None:
        first_bad = (name, tid, f'max diff {d.max()}')

if first_bad:
    print(f"\n>>> PRIMEIRA DIVERGÊNCIA: {first_bad[0]} (tid={first_bad[1]}): {first_bad[2]}")
else:
    print("\n>>> Tudo bateu (diff ≤ 1)")
