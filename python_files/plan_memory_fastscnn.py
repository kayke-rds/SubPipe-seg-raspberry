#!/usr/bin/env python3
"""
plan_memory_fastscnn.py -- Planeja a arena de ativações da Fast-SCNN.

Lê 'tensors_c.csv' (canônico, escrito à mão) e produz 'fastscnn_mem.h'.
Não depende do ONNX.
"""
import sys, os, csv, time
from bisect import insort

ALIGN_BYTES = 64

def bytes_of(c, h, w, dtype):
    elems = c * h * w
    if dtype == 'INT8' or dtype == 'UINT8': return elems
    if dtype in ('FLOAT', 'INT32'):         return elems * 4
    if dtype == 'INT64':                    return elems * 8
    raise ValueError(f"dtype não suportado: {dtype}")

def align_up(n, a): return ((n + a - 1) // a) * a

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else src

    tensors = []
    with open(os.path.join(src, "tensors_c.csv")) as f:
        for row in csv.DictReader(f):
            tensors.append({
                'name':  row['name'],
                'c':     int(row['c']),
                'h':     int(row['h']),
                'w':     int(row['w']),
                'dtype': row['dtype'].strip(),
                'birth': int(row['birth']),
                'death': int(row['death']),
            })

    order = sorted(tensors, key=lambda t: (t['birth'], -bytes_of(t['c'], t['h'], t['w'], t['dtype'])))

    allocated = []  # (off, size, death)

    def find_offset(size, birth):
        live = sorted((o, o + s) for (o, s, d) in allocated if d >= birth)
        pos = 0
        for (s, e) in live:
            if pos + size <= s: return pos
            if e > pos:         pos = e
        return pos

    total = 0
    t0 = time.time()
    for k, t in enumerate(order):
        size = align_up(bytes_of(t['c'], t['h'], t['w'], t['dtype']), ALIGN_BYTES)
        off  = find_offset(size, t['birth'])
        t['offset']     = off
        t['size_bytes'] = size
        insort(allocated, (off, size, t['death']))
        total = max(total, off + size)
        print(f"[{k:3d}/{len(order)}] {t['name']:16s} "
              f"{t['dtype']:5s} {t['c']:>4d}x{t['h']:>4d}x{t['w']:>4d} "
              f"b={t['birth']:>3d} d={t['death']:>3d} "
              f"off={off:>9d} size={size:>9d} t={time.time()-t0:.2f}s")

    # ---- header ----
    hdr = []
    A = hdr.append
    A("/* fastscnn_mem.h -- GERADO por plan_memory_fastscnn.py. NÃO EDITAR. */")
    A("#ifndef FASTSCNN_MEM_H")
    A("#define FASTSCNN_MEM_H")
    A("#include <stdint.h>")
    A("")
    A(f"#define FASTSCNN_ARENA_BYTES   {total}u")
    A(f"#define FASTSCNN_ARENA_ALIGN   {ALIGN_BYTES}u")
    A("")
    A("typedef enum {")
    for t in tensors:
        A(f"    T_{t['name']} = {tensors.index(t)},")
    A("    T_COUNT")
    A("} TensorId;")
    A("")
    A("typedef enum { DT_F32=0, DT_I8=1, DT_U8=2, DT_I32=3, DT_I64=4 } TensorDType;")
    A("")
    A("typedef struct {")
    A("    uint32_t off;")
    A("    uint32_t size;")
    A("    uint16_t c, h, w;")
    A("    uint8_t  dtype;")
    A("} TensorDesc;")
    A("")
    A("static const TensorDesc FASTSCNN_TENSOR[T_COUNT] = {")
    dt_map = {'FLOAT':'DT_F32','INT8':'DT_I8','UINT8':'DT_U8',
              'INT32':'DT_I32','INT64':'DT_I64'}
    for t in tensors:
        A(f"    {{{t['offset']}u, {t['size_bytes']}u, "
          f"{t['c']}, {t['h']}, {t['w']}, {dt_map[t['dtype']]}}},  "
          f"/* {t['name']} */")
    A("};")
    A("")
    A("#define TENSOR_F32(a, id) ((float  *)((a) + FASTSCNN_TENSOR[id].off))")
    A("#define TENSOR_I8(a, id)  ((int8_t *)((a) + FASTSCNN_TENSOR[id].off))")
    A("")
    A("#endif /* FASTSCNN_MEM_H */")

    with open(os.path.join(out, "fastscnn_mem.h"), "w") as f:
        f.write("\n".join(hdr) + "\n")

    print(f"\nok: arena={total} bytes ({total/1024:.1f} KB)")
    print(f"    {len(tensors)} tensores na arena")

if __name__ == "__main__":
    main()
