#!/usr/bin/env python3
"""
extract_ppm_fp32.py

Extrai as 5 convs FP32 do PPM (pesos + bias) e as escalas de
input/output do PPM (INT8 nas bordas).

Gera:
  fastscnn_ppm_fp32.bin
  fastscnn_ppm_fp32.h
"""
import sys, os
from collections import deque
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


# ---------- index ----------
def build_index(g):
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producers = {out: n for n in g.node for out in n.output}
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    return init, producers, consumers


# ---------- helpers ----------
def find_conv_by_weight(graph, name_contains):
    for n in graph.node:
        if n.op_type == 'Conv' and name_contains in n.input[1]:
            return n
    raise ValueError(f"conv com peso '{name_contains}' não encontrada")


def read_conv_fp32(conv, init):
    W = init[conv.input[1]].astype(np.float32)
    B = (init[conv.input[2]].astype(np.float32)
         if len(conv.input) > 2 else np.zeros(W.shape[0], dtype=np.float32))
    return W, B


def find_q_after(conv, consumers, init):
    """Primeiro QuantizeLinear alcançável a partir da saída da conv,
    atravessando qualquer op intermediário."""
    q = deque([conv.output[0]])
    visited = set()
    while q:
        t = q.popleft()
        if t in visited: continue
        visited.add(t)
        for c in consumers.get(t, []):
            if c.op_type == 'QuantizeLinear':
                s  = float(init[c.input[1]])
                zp = int(init[c.input[2]]) if len(c.input) > 2 else 0
                return s, zp, c.output[0]
            for o in c.output:
                q.append(o)
    raise ValueError(f"não achei Q após {conv.name}")


def channels_of(graph, name):
    """Retorna o número de canais (dim 1) de um tensor, ou -1."""
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        if v.name == name:
            dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
            return dims[1] if len(dims) >= 4 else -1
    return -1


def resolve_q_dq_for_tensor(graph, producers, init, tensor_name):
    """
    Dado um tensor X (fp32 no grafo QDQ), retorna (scale, zp) do Q que
    originou esse X, seguindo a cadeia X ← DequantizeLinear ← int8 ← QuantizeLinear.
    Se não achar DQ imediato, sobe por ops intermediárias (Relu/Clip).
    """
    cur_name = tensor_name
    visited = set()
    for _ in range(15):
        if cur_name in visited: break
        visited.add(cur_name)

        prod = producers.get(cur_name)
        if prod is None:
            break
        if prod.op_type == 'DequantizeLinear':
            s  = float(init[prod.input[1]])
            zp = int(init[prod.input[2]]) if len(prod.input) > 2 else 0
            return s, zp, prod.input[0]   # scale, zp, int8 tensor
        if prod.op_type in ('Relu', 'Clip', 'Resize', 'Transpose'):
            cur_name = prod.input[0]
            continue
        cur_name = prod.input[0]
    raise ValueError(f"não consegui resolver Q/DQ para {tensor_name}")


# ---------- main ----------
def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'fastscnn_int8_qdq_ppmfp32.onnx'
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"lendo: {onnx_path}")

    m = onnx.load(onnx_path)
    m = shape_inference.infer_shapes(m)
    g = m.graph
    init, producers, consumers = build_index(g)

    # ===== 1. As 5 convs do PPM =====
    ppm_conv_names = {
        'conv1': 'ppm.conv1.conv.0.weight',
        'conv2': 'ppm.conv2.conv.0.weight',
        'conv3': 'ppm.conv3.conv.0.weight',
        'conv4': 'ppm.conv4.conv.0.weight',
        'out'  : 'ppm.out.conv.0.weight',
    }
    blobs = []
    for key, wname in ppm_conv_names.items():
        conv = find_conv_by_weight(g, wname)
        W, B = read_conv_fp32(conv, init)
        Cout, Cin, kH, kW = W.shape
        assert kH == 1 and kW == 1, f"{wname}: só 1x1 suportado"
        print(f"  {key}: Cin={Cin} Cout={Cout} k=1x1  ({conv.name})")
        blobs.append({'key': key, 'name': conv.name,
                      'W': W, 'B': B, 'Cin': Cin, 'Cout': Cout})

    # ===== 2. PPM input (o `x` original do concat, 128 canais) =====
    # Sobe da ppm.out.conv até o Concat do PPM
    ppm_out_conv = find_conv_by_weight(g, 'ppm.out.conv.0.weight')
    cur_name = ppm_out_conv.input[0]

    concat_node = None
    for _ in range(15):
        prod = producers.get(cur_name)
        if prod is None:
            break
        if prod.op_type == 'Concat':
            concat_node = prod
            break
        cur_name = prod.input[0]

    if concat_node is None:
        raise ValueError("não achei Concat do PPM")

    print(f"  Concat do PPM: {concat_node.name}  inputs={len(concat_node.input)}")

    # O x tem 128 canais; os 4 upsamples têm 32. Pega o de 128.
    x_fp32_name = None
    for inp in concat_node.input:
        ch = channels_of(g, inp)
        if ch == 128:
            x_fp32_name = inp
            break

    if x_fp32_name is None:
        # fallback: primeiro input do concat é sempre o x
        x_fp32_name = concat_node.input[0]
        print(f"  AVISO: canal desconhecido, usando primeiro input")

    X_scale, X_zp, x_int8_src = resolve_q_dq_for_tensor(
        g, producers, init, x_fp32_name)

    print(f"  PPM input : x_fp32_tensor = {x_fp32_name}")
    print(f"              int8 tensor    = {x_int8_src}")
    print(f"              scale = {X_scale:.6e}  zp = {X_zp}")

    # ===== 3. PPM output =====
    Y_scale, Y_zp, y_int8_dst = find_q_after(ppm_out_conv, consumers, init)
    print(f"  PPM output: scale = {Y_scale:.6e}  zp = {Y_zp}")
    print(f"              int8 tensor = {y_int8_dst}")

    # ===== 4. Escreve .bin =====
    bin_path = os.path.join(out_dir, 'fastscnn_ppm_fp32.bin')
    offsets = []
    with open(bin_path, 'wb') as f:
        for b in blobs:
            cur = {}
            for key, arr in (('W', b['W']), ('B', b['B'])):
                pos = f.tell()
                pad = (-pos) % 64
                if pad: f.write(b'\x00' * pad)
                cur[key] = f.tell()
                f.write(arr.tobytes())
            offsets.append(cur)
        total = f.tell()

    # ===== 5. Escreve .h =====
    hdr = []
    A = hdr.append
    A("/* fastscnn_ppm_fp32.h -- GERADO. NÃO EDITAR. */")
    A("#ifndef FASTSCNN_PPM_FP32_H")
    A("#define FASTSCNN_PPM_FP32_H")
    A("#include <stdint.h>")
    A("")
    A(f"#define FASTSCNN_PPM_N_CONV    {len(blobs)}u")
    A(f"#define FASTSCNN_PPM_BIN_SIZE  {total}u")
    A(f"#define FASTSCNN_PPM_IN_SCALE  {X_scale!r}f")
    A(f"#define FASTSCNN_PPM_IN_ZP     {X_zp}")
    A(f"#define FASTSCNN_PPM_OUT_SCALE {Y_scale!r}f")
    A(f"#define FASTSCNN_PPM_OUT_ZP    {Y_zp}")
    A("")
    A("typedef struct {")
    A("    uint32_t off_W, off_B;")
    A("    uint16_t Cin, Cout;")
    A("} FastSCNNPPMConvDesc;")
    A("")
    A("static const FastSCNNPPMConvDesc FASTSCNN_PPM_CONV[FASTSCNN_PPM_N_CONV] = {")
    for b, off in zip(blobs, offsets):
        A(f"    {{{off['W']}, {off['B']}, {b['Cin']}, {b['Cout']}}},  /* {b['key']} */")
    A("};")
    A("")
    A("#endif")
    with open(os.path.join(out_dir, 'fastscnn_ppm_fp32.h'), 'w') as f:
        f.write("\n".join(hdr) + "\n")

    print(f"\nok: {bin_path} = {total} bytes")
    print(f"    {out_dir}/fastscnn_ppm_fp32.h")


if __name__ == '__main__':
    main()
