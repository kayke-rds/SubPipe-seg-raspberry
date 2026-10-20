#!/usr/bin/env python3
"""
extract_weights_fastscnn.py  (v2)

Lê fastscnn_int8_qdq.onnx e produz:
  - fastscnn_weights.bin : W_i8, W_scale, bias_corr_i32, M por conv
  - fastscnn_weights.h   : offsets e metadados
"""
import sys, os
import numpy as np
import onnx
from onnx import numpy_helper
from collections import deque


def build_index(graph):
    producers, consumers = {}, {}
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        for out in node.output:
            producers[out] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return producers, consumers, initializers


def get_initializer(name, initializers):
    if name not in initializers:
        raise KeyError(f"initializer ausente: {name}")
    return initializers[name]


def extract_weight(conv, producers, initializers):
    """W_i8, W_scale, W_zp do peso da conv (via DequantizeLinear)."""
    node = producers.get(conv.input[1])
    if node is None or node.op_type != 'DequantizeLinear':
        raise ValueError(f"peso de {conv.name}: producer é "
                         f"{node.op_type if node else 'None'}, não DQ")
    W_i8    = get_initializer(node.input[0], initializers)
    W_scale = get_initializer(node.input[1], initializers).astype(np.float32)
    W_zp    = (get_initializer(node.input[2], initializers).astype(np.int32)
               if len(node.input) > 2 else np.zeros_like(W_scale, dtype=np.int32))
    return W_i8, W_scale, W_zp


def extract_bias(conv, producers, initializers):
    """
    Bias quantizado (INT32 per-channel) + scale + zp.
    Retorna bias_quant (int32[Cout]), bias_scale (float[Cout]), bias_zp (int32[Cout]).
    """
    name = conv.input[2]                    # ex: 'xxx.weight_bias'
    node = producers.get(name)

    if node is None or node.op_type != 'DequantizeLinear':
        # alguns exportadores colocam direto: <name>_quantized
        if (name + '_quantized') in initializers:
            bq = get_initializer(name + '_quantized', initializers).astype(np.int32)
            bs = get_initializer(name + '_quantized_scale', initializers).astype(np.float32)
            bz = get_initializer(name + '_quantized_zero_point', initializers).astype(np.int32)
            return bq, bs, bz
        raise ValueError(f"bias de {conv.name}: não achei nem DQ nem sufixo _quantized")

    # Caso normal: bias é saída de DQ
    bq = get_initializer(node.input[0], initializers).astype(np.int32)
    bs = get_initializer(node.input[1], initializers).astype(np.float32)
    bz = (get_initializer(node.input[2], initializers).astype(np.int32)
          if len(node.input) > 2 else np.zeros_like(bs, dtype=np.int32))
    return bq, bs, bz


def find_input_quant(conv, producers, initializers):
    """(X_scale, X_zp) da entrada da conv."""
    node = producers.get(conv.input[0])
    if node is None or node.op_type != 'DequantizeLinear':
        # rede: entrada da rede é inicializador
        s  = float(get_initializer('input_scale', initializers))
        zp = int(get_initializer('input_zero_point', initializers))
        return s, zp
    s  = float(get_initializer(node.input[1], initializers))
    zp = int(get_initializer(node.input[2], initializers)) if len(node.input) > 2 else 0
    return s, zp


def find_output_quant(conv, consumers, initializers):
    """(Y_scale, Y_zp, has_relu) do tensor após a conv.
    has_relu detectado pelo nome do tensor (Dynamo fundiu Conv+Relu no Q).
    """
    # heurística: se o output da conv se chama 'relu*', tem ReLU fundido
    start_has_relu = any(t.lower().startswith('relu') for t in conv.output)

    visited = set()
    q = deque((t, start_has_relu) for t in conv.output)
    while q:
        t, has_relu = q.popleft()
        if t in visited: continue
        visited.add(t)
        for c in consumers.get(t, []):
            if c.op_type == 'QuantizeLinear':
                s  = float(get_initializer(c.input[1], initializers))
                zp = int(get_initializer(c.input[2], initializers)) if len(c.input) > 2 else 0
                return s, zp, has_relu
            if c.op_type in ('Relu', 'Clip'):
                for out in c.output:
                    q.append((out, has_relu or c.op_type == 'Relu'))
    raise ValueError(f"não achei Q após {conv.name}")

def extract_attrs(conv):
    stride, pads, groups = 1, (0, 0), 1
    for a in conv.attribute:
        if a.name == 'strides':   stride = a.ints[0]
        elif a.name == 'pads':    pads = (a.ints[0], a.ints[1])
        elif a.name == 'group':   groups = a.i
    return stride, pads, groups


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else src

    model = onnx.load(os.path.join(src, 'fastscnn_int8_qdq_ppmfp32.onnx'))
    graph = model.graph
    producers, consumers, initializers = build_index(graph)

    convs = [n for n in graph.node if n.op_type == 'Conv']
    print(f"Encontrei {len(convs)} Convs\n")

    blobs = []
    for i, conv in enumerate(convs):
        try:
            W_i8, W_scale, W_zp    = extract_weight(conv, producers, initializers)
            bq,   b_scale, b_zp    = extract_bias(conv, producers, initializers)
            X_scale, X_zp          = find_input_quant(conv, producers, initializers)
            Y_scale, Y_zp, has_relu = find_output_quant(conv, consumers, initializers)
            stride, pads, groups   = extract_attrs(conv)

            Cout, Cin_g, kH, kW = W_i8.shape
            Cin = Cin_g * groups

            # Sanidade: W_zp deve ser 0; X_scale * W_scale deve casar com b_scale
            assert np.all(W_zp == 0), f"{conv.name}: W_zp != 0"
            expected_b_scale = X_scale * W_scale
            if not np.allclose(b_scale, expected_b_scale, rtol=1e-3, atol=1e-8):
                print(f"[{i:2d}] AVISO {conv.name}: b_scale != X_scale*W_scale  "
                      f"({b_scale[:3]} vs {expected_b_scale[:3]})")

            # W_sum
            W_sum = W_i8.astype(np.int64).sum(axis=(1, 2, 3)).astype(np.int32)

            # bias_corr_i32[q] = (bias_quant - bias_zp) - X_zp * W_sum[q]
            bias_corr_i32 = (bq - b_zp - X_zp * W_sum).astype(np.int32)

            # M[q] = (X_scale * W_scale[q]) / Y_scale
            M = (X_scale * W_scale / Y_scale).astype(np.float32)

            blobs.append({
                'name': conv.name,
                'W_i8': W_i8.astype(np.int8),
                'W_scale': W_scale,
                'bias_corr_i32': bias_corr_i32,
                'M': M,
                'Cout': Cout, 'Cin': Cin, 'k': kH,
                'stride': stride, 'pad': pads[0], 'groups': groups,
                'X_scale': X_scale, 'X_zp': X_zp,
                'Y_scale': Y_scale, 'Y_zp': Y_zp,
                'has_relu': has_relu,
            })
            print(f"[{i:2d}/{len(convs)}] {conv.name:26s} "
                  f"Cin={Cin:4d} Cout={Cout:4d} k={kH} s={stride} p={pads[0]} g={groups} "
                  f"relu={int(has_relu)}  "
                  f"W_scale=[{W_scale.min():.2e},{W_scale.max():.2e}]  "
                  f"M=[{M.min():.2e},{M.max():.2e}]  "
                  f"bias_corr=[{bias_corr_i32.min():.0f},{bias_corr_i32.max():.0f}]")
        except Exception as e:
            print(f"[{i:2d}] {conv.name} ERRO: {e}")

    if not blobs:
        print("nada a escrever")
        return

    # ---- escrever .bin ----
    bin_path = os.path.join(out, 'fastscnn_weights.bin')
    hdr_path = os.path.join(out, 'fastscnn_weights.h')

    offsets = []
    total = 0
    with open(bin_path, 'wb') as f:
        for b in blobs:
            cur = {}
            for key in ('W_i8', 'W_scale', 'bias_corr_i32', 'M'):
                arr = b[key]
                pos = f.tell()
                pad = (-pos) % 64
                if pad: f.write(b'\x00' * pad)
                cur[key] = f.tell()
                f.write(arr.tobytes())
            offsets.append(cur)
        total = f.tell()

    # ---- escrever .h ----
    h = []
    A = h.append
    A("/* fastscnn_weights.h -- GERADO por extract_weights_fastscnn.py. */")
    A("#ifndef FASTSCNN_WEIGHTS_H")
    A("#define FASTSCNN_WEIGHTS_H")
    A("#include <stdint.h>")
    A("")
    A(f"#define FASTSCNN_N_CONVS       {len(blobs)}u")
    A(f"#define FASTSCNN_WEIGHTS_SIZE  {total}u")
    A("")
    A("typedef struct {")
    A("    uint32_t off_W;")
    A("    uint32_t off_scale_w;")
    A("    uint32_t off_bias_corr;")
    A("    uint32_t off_M;")
    A("    uint16_t Cout, Cin;")
    A("    uint8_t  k, stride, pad;")
    A("    uint16_t groups;              /* uint16_t: depthwise chega a 768 */")
    A("    uint8_t  has_relu;")
    A("    float    X_scale, Y_scale;")
    A("    int8_t   X_zp, Y_zp;")
    A("} FastSCNNConvDesc;")
    A("")
    A("static const FastSCNNConvDesc FASTSCNN_CONV[FASTSCNN_N_CONVS] = {")
    for b, off in zip(blobs, offsets):
        A(f"    {{  /* {b['name']} */")
        A(f"        {off['W_i8']}, {off['W_scale']}, "
          f"{off['bias_corr_i32']}, {off['M']},")
        A(f"        {b['Cout']}, {b['Cin']}, "
          f"{b['k']}, {b['stride']}, {b['pad']}, {b['groups']}, {int(b['has_relu'])},")
        A(f"        {b['X_scale']!r}f, {b['Y_scale']!r}f, "
          f"{b['X_zp']}, {b['Y_zp']},")
        A(f"    }},")
    A("};")
    A("")
    A("#endif /* FASTSCNN_WEIGHTS_H */")

    with open(hdr_path, 'w') as f:
        f.write("\n".join(h) + "\n")

    print(f"\nok:")
    print(f"  {bin_path}  = {total} bytes ({total/1024:.1f} KB)")
    print(f"  {hdr_path}  ({len(blobs)} convs)")


if __name__ == '__main__':
    main()
