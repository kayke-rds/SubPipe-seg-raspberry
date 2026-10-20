#!/usr/bin/env python3
"""
gen_test_conv.py <conv_idx>

Gera para a conv indicada:
  test_input.bin       int8  [Cin, Hin, Win]
  test_output_ref.bin  int8  [Cout, Hout, Wout]   (referência numpy)
  test_meta.txt        metadados para o C

A referência usa padding = X_zp (mesma convenção do ONNX/PyTorch para INT8),
que é também o que o kernel em C faz após a correção.
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
from collections import deque


def shape_of(graph, name):
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        if v.name == name:
            return [d.dim_value for d in v.type.tensor_type.shape.dim]
    return None


def find_initializer(init, name):
    if name not in init:
        raise KeyError(f"initializer ausente: {name}")
    return init[name]


def extract_weight(conv, producers, init):
    """W_i8, W_scale, W_zp (zero-point do peso, tipicamente 0)."""
    wn = producers.get(conv.input[1])
    if wn is None or wn.op_type != 'DequantizeLinear':
        raise ValueError(f"peso de {conv.name} não vem de DQ")
    W_i8    = find_initializer(init, wn.input[0]).astype(np.int8)
    W_scale = find_initializer(init, wn.input[1]).astype(np.float32)
    W_zp    = (find_initializer(init, wn.input[2]).astype(np.int32)
               if len(wn.input) > 2 else np.zeros(W_scale.shape, dtype=np.int32))
    return W_i8, W_scale, W_zp


def extract_bias(conv, producers, init):
    """bias_q (int32), bias_scale (float), bias_zp (int32)."""
    name = conv.input[2]
    node = producers.get(name)
    if node is not None and node.op_type == 'DequantizeLinear':
        bq = find_initializer(init, node.input[0]).astype(np.int32)
        bs = find_initializer(init, node.input[1]).astype(np.float32)
        bz = (find_initializer(init, node.input[2]).astype(np.int32)
              if len(node.input) > 2 else np.zeros(bs.shape, dtype=np.int32))
        return bq, bs, bz

    # fallback: initializer direto com sufixo _quantized
    if (name + '_quantized') in init:
        bq = find_initializer(init, name + '_quantized').astype(np.int32)
        bs = find_initializer(init, name + '_quantized_scale').astype(np.float32)
        bz = find_initializer(init, name + '_quantized_zero_point').astype(np.int32)
        return bq, bs, bz
    raise ValueError(f"bias de {conv.name} não encontrado ({name})")


def find_input_quant(conv, producers, init):
    """(X_scale, X_zp) da entrada da conv."""
    node = producers.get(conv.input[0])
    if node is None:
        # entrada da rede
        return (float(find_initializer(init, 'input_scale')),
                int(find_initializer(init, 'input_zero_point')))
    if node.op_type != 'DequantizeLinear':
        raise ValueError(f"input de {conv.name} não vem de DQ: {node.op_type}")
    s  = float(find_initializer(init, node.input[1]))
    zp = int(find_initializer(init, node.input[2])) if len(node.input) > 2 else 0
    return s, zp


def find_output_quant(conv, consumers, init):
    """(Y_scale, Y_zp, has_relu) do tensor após a conv."""
    has_relu = any(t.lower().startswith('relu') for t in conv.output)
    q = deque((t, has_relu) for t in conv.output)
    visited = set()
    while q:
        t, relu = q.popleft()
        if t in visited: continue
        visited.add(t)
        for c in consumers.get(t, []):
            if c.op_type == 'QuantizeLinear':
                s  = float(find_initializer(init, c.input[1]))
                zp = int(find_initializer(init, c.input[2])) if len(c.input) > 2 else 0
                return s, zp, relu
            if c.op_type in ('Relu', 'Clip'):
                for o in c.output:
                    q.append((o, relu or c.op_type == 'Relu'))
    raise ValueError(f"não achei Q após {conv.name}")


def extract_attrs(conv):
    S = P = None; G = 1
    for a in conv.attribute:
        if a.name == 'strides': S = a.ints[0]
        if a.name == 'pads':    P = a.ints[0]
        if a.name == 'group':   G = a.i
    return (S or 1), (P or 0), G


def reference_forward(X_i8, W_i8, bias_corr, M, X_zp, Y_zp, has_relu,
                      Cin, Cout, kH, S, P, G, Hout, Wout):
    """
    Referência numpy 1:1 com o kernel em C.
    IMPORTANTE: padding = X_zp (não zero), como ONNX/PyTorch.
    """
    Cin_g  = Cin  // G
    Cout_g = Cout // G
    Y = np.empty((Cout, Hout, Wout), dtype=np.int8)

    for co in range(Cout):
        g_idx   = co // Cout_g
        ci_base = g_idx * Cin_g
        X_g     = X_i8[ci_base : ci_base + Cin_g].astype(np.int32)

        # Padding quantizado correto
        X_g_pad = np.pad(X_g, ((0,0), (P, P), (P, P)),
                         mode='constant', constant_values=int(X_zp))

        acc = np.zeros((Hout, Wout), dtype=np.int64)
        for ky in range(kH):
            for kx in range(kH):
                sl = X_g_pad[:, ky : ky + (Hout-1)*S + 1 : S,
                                kx : kx + (Wout-1)*S + 1 : S]
                w_col = W_i8[co, :, ky, kx].astype(np.int64)
                acc += np.tensordot(w_col, sl, axes=([0],[0]))

        acc = acc.astype(np.int64) + int(bias_corr[co])

        q_float = acc.astype(np.float32) * np.float32(M[co])
        q_int   = np.rint(q_float).astype(np.int32) + int(Y_zp)
        if has_relu:
            q_int = np.maximum(q_int, Y_zp)
        q_int = np.clip(q_int, -128, 127)

        Y[co] = q_int.astype(np.int8)
    return Y


def main():
    conv_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 44

    m = onnx.load('fastscnn_int8_qdq_ppmfp32.onnx')
    m = shape_inference.infer_shapes(m)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producers = {out: n for n in g.node for out in n.output}
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    convs = [n for n in g.node if n.op_type == 'Conv']
    if conv_idx >= len(convs):
        raise ValueError(f"conv_idx={conv_idx} mas só há {len(convs)} convs")
    conv = convs[conv_idx]
    print(f"conv_idx={conv_idx}  name={conv.name}")

    in_shape  = shape_of(g, conv.input[0])
    out_shape = shape_of(g, conv.output[0])
    _, Cin, Hin, Win    = in_shape
    _, Cout, Hout, Wout = out_shape
    print(f"  in : {in_shape}")
    print(f"  out: {out_shape}")

    W_i8, W_scale, W_zp     = extract_weight(conv, producers, init)
    bias_q, bias_s, bias_z  = extract_bias(conv, producers, init)
    X_scale, X_zp           = find_input_quant(conv, producers, init)
    Y_scale, Y_zp, has_relu = find_output_quant(conv, consumers, init)
    S, P, G                 = extract_attrs(conv)

    Cout_w, Cin_g, kH, kW = W_i8.shape
    Cin_eff = Cin_g * G
    assert Cout_w == Cout
    assert Cin_eff == Cin
    assert kH == kW, "kernel não quadrado não suportado"
    assert np.all(W_zp == 0), f"W_zp != 0: {W_zp[:5]}"

    print(f"  Cin={Cin} Cout={Cout} k={kH} s={S} p={P} g={G} relu={int(has_relu)}")
    print(f"  X_scale={X_scale:.6e}  X_zp={X_zp}")
    print(f"  Y_scale={Y_scale:.6e}  Y_zp={Y_zp}")

    # pré-cálculos
    W_sum         = W_i8.astype(np.int64).sum(axis=(1, 2, 3)).astype(np.int32)
    bias_corr_i32 = (bias_q - bias_z - X_zp * W_sum).astype(np.int32)
    M             = (X_scale * W_scale / Y_scale).astype(np.float32)

    # input sintético em distribuição pós-ReLU
    rng   = np.random.default_rng(42)
    noise = rng.exponential(scale=25.0, size=(Cin, Hin, Win))
    noise = np.clip(noise, 0, 255).astype(np.int32)
    X_i8  = (noise + X_zp).astype(np.int8)

    X_i8.tofile('test_input.bin')

    Y_ref = reference_forward(
        X_i8, W_i8, bias_corr_i32, M, X_zp, Y_zp, has_relu,
        Cin, Cout, kH, S, P, G, Hout, Wout)

    Y_ref.tofile('test_output_ref.bin')

    with open('test_meta.txt', 'w') as f:
        f.write(f"conv_idx {conv_idx}\n")
        f.write(f"Cin  {Cin}\nHin  {Hin}\nWin  {Win}\n")
        f.write(f"Cout {Cout}\nHout {Hout}\nWout {Wout}\n")

    print(f"\ntest_input.bin      = {X_i8.size} bytes  "
          f"(min={X_i8.min()} max={X_i8.max()} mean={X_i8.mean():.2f})")
    print(f"test_output_ref.bin = {Y_ref.size} bytes  "
          f"(min={Y_ref.min()} max={Y_ref.max()} mean={Y_ref.mean():.2f})")

    # Mapeia índice ONNX para índice em FASTSCNN_CONV (0..39)
    if conv_idx <= 32:
        h_idx = conv_idx
    elif conv_idx <= 37:
        h_idx = None    # conv do PPM: fora do .h
    else:
        h_idx = conv_idx - 5

    with open('test_meta.txt', 'w') as f:
        f.write(f"conv_idx {conv_idx}\n")
        f.write(f"h_idx    {h_idx if h_idx is not None else -1}\n")
        f.write(f"Cin  {Cin}\nHin  {Hin}\nWin  {Win}\n")
        f.write(f"Cout {Cout}\nHout {Hout}\nWout {Wout}\n")


if __name__ == '__main__':
    main()
