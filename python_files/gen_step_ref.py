#!/usr/bin/env python3
"""
gen_step_ref.py -- referência ONNX para cada tensor intermediário.
"""
import os, re, sys
import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort
from collections import deque

TARGETS = [
    ('l2d_conv',      'conv_out', 'learning_to_downsample.conv.conv.0.weight'),
    ('l2d_ds1_dw',    'conv_out', 'learning_to_downsample.dsconv1.conv.0.weight'),
    ('l2d_ds1_pw',    'conv_out', 'learning_to_downsample.dsconv1.conv.3.weight'),
    ('l2d_ds2_dw',    'conv_out', 'learning_to_downsample.dsconv2.conv.0.weight'),
    ('l2d_ds2_pw',    'conv_out', 'learning_to_downsample.dsconv2.conv.3.weight'),

    ('b1_0_expand',   'conv_out', 'global_feature_extractor.bottleneck1.0.block.0.conv.0.weight'),
    ('b1_0_dw',       'conv_out', 'global_feature_extractor.bottleneck1.0.block.1.conv.0.weight'),
    ('b1_0_project',  'conv_out', 'global_feature_extractor.bottleneck1.0.block.2.weight'),
    ('b1_1_expand',   'conv_out', 'global_feature_extractor.bottleneck1.1.block.0.conv.0.weight'),
    ('b1_1_dw',       'conv_out', 'global_feature_extractor.bottleneck1.1.block.1.conv.0.weight'),
    ('b1_1_project',  'conv_out', 'global_feature_extractor.bottleneck1.1.block.2.weight'),
    ('b1_1_add',      'input_q',  'global_feature_extractor.bottleneck1.2.block.0.conv.0.weight'),
    ('b1_2_expand',   'conv_out', 'global_feature_extractor.bottleneck1.2.block.0.conv.0.weight'),
    ('b1_2_dw',       'conv_out', 'global_feature_extractor.bottleneck1.2.block.1.conv.0.weight'),
    ('b1_2_project',  'conv_out', 'global_feature_extractor.bottleneck1.2.block.2.weight'),
    ('b1_2_add',      'input_q',  'global_feature_extractor.bottleneck2.0.block.0.conv.0.weight'),

    ('b2_0_expand',   'conv_out', 'global_feature_extractor.bottleneck2.0.block.0.conv.0.weight'),
    ('b2_0_dw',       'conv_out', 'global_feature_extractor.bottleneck2.0.block.1.conv.0.weight'),
    ('b2_0_project',  'conv_out', 'global_feature_extractor.bottleneck2.0.block.2.weight'),
    ('b2_1_expand',   'conv_out', 'global_feature_extractor.bottleneck2.1.block.0.conv.0.weight'),
    ('b2_1_dw',       'conv_out', 'global_feature_extractor.bottleneck2.1.block.1.conv.0.weight'),
    ('b2_1_project',  'conv_out', 'global_feature_extractor.bottleneck2.1.block.2.weight'),
    ('b2_1_add',      'input_q',  'global_feature_extractor.bottleneck2.2.block.0.conv.0.weight'),
    ('b2_2_expand',   'conv_out', 'global_feature_extractor.bottleneck2.2.block.0.conv.0.weight'),
    ('b2_2_dw',       'conv_out', 'global_feature_extractor.bottleneck2.2.block.1.conv.0.weight'),
    ('b2_2_project',  'conv_out', 'global_feature_extractor.bottleneck2.2.block.2.weight'),
    ('b2_2_add',      'input_q',  'global_feature_extractor.bottleneck3.0.block.0.conv.0.weight'),

    ('b3_0_expand',   'conv_out', 'global_feature_extractor.bottleneck3.0.block.0.conv.0.weight'),
    ('b3_0_dw',       'conv_out', 'global_feature_extractor.bottleneck3.0.block.1.conv.0.weight'),
    ('b3_0_project',  'conv_out', 'global_feature_extractor.bottleneck3.0.block.2.weight'),
    ('b3_1_expand',   'conv_out', 'global_feature_extractor.bottleneck3.1.block.0.conv.0.weight'),
    ('b3_1_dw',       'conv_out', 'global_feature_extractor.bottleneck3.1.block.1.conv.0.weight'),
    ('b3_1_project',  'conv_out', 'global_feature_extractor.bottleneck3.1.block.2.weight'),
    ('b3_1_add',      'input_q',  'global_feature_extractor.bottleneck3.2.block.0.conv.0.weight'),
    ('b3_2_expand',   'conv_out', 'global_feature_extractor.bottleneck3.2.block.0.conv.0.weight'),
    ('b3_2_dw',       'conv_out', 'global_feature_extractor.bottleneck3.2.block.1.conv.0.weight'),
    ('b3_2_project',  'conv_out', 'global_feature_extractor.bottleneck3.2.block.2.weight'),
    ('b3_2_add',      'ppm_in',   None),          # NOVO — resolve pelo concat do PPM

    ('ff_up',         'input_q',  'feature_fusion.dwconv.conv.0.weight'),  # substitui ppm_out_i8
    ('ff_dw',         'conv_out', 'feature_fusion.dwconv.conv.0.weight'),
    ('ff_low',        'conv_out', 'feature_fusion.conv_lower_res.0.weight'),
    ('ff_high',       'conv_out', 'feature_fusion.conv_higher_res.0.weight'),
    ('ff_add',        'input_q',  'classifier.dsconv1.conv.0.weight'),

    ('cls_ds1_dw',    'conv_out', 'classifier.dsconv1.conv.0.weight'),
    ('cls_ds1_pw',    'conv_out', 'classifier.dsconv1.conv.3.weight'),
    ('cls_ds2_dw',    'conv_out', 'classifier.dsconv2.conv.0.weight'),
    ('cls_ds2_pw',    'conv_out', 'classifier.dsconv2.conv.3.weight'),
    ('cls_out',       'conv_out', 'classifier.conv.1.weight'),
]


def parse_mem_h(path):
    """
    Extrai {name: {tid, c, h, w, dtype}} do fastscnn_mem.h.
    A ordem das linhas do array casa com a ordem do enum (ambos são
    `sorted(tensors.keys())` no gerador). Formato esperado:
        {0u, 1378880u, 32, 159, 271, DT_I8},  /* l2d_conv */
    """
    txt = open(path).read()
    m_arr = re.search(r'FASTSCNN_TENSOR\[T_COUNT\]\s*=\s*\{(.*?)\};', txt, re.S)
    if not m_arr:
        # formato alternativo: enum + array separados, mas linha única
        m_arr = re.search(r'FASTSCNN_TENSOR\[T_COUNT\]\s*=\s*\{(.*?)\n\};', txt, re.S)
    if not m_arr:
        raise RuntimeError("não achei FASTSCNN_TENSOR no mem.h")
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


def find_conv_by_weight(graph, name_contains):
    for n in graph.node:
        if n.op_type == 'Conv' and name_contains in n.input[1]:
            return n
    return None


def q_after_conv(conv, consumers):
    q = deque([conv.output[0]]); visited = set()
    while q:
        t = q.popleft()
        if t in visited: continue
        visited.add(t)
        for c in consumers.get(t, []):
            if c.op_type == 'QuantizeLinear':
                return c.output[0]
            for o in c.output: q.append(o)
    return None


def q_before_conv(conv, producers):
    in_fp32 = conv.input[0]
    dq = producers.get(in_fp32)
    if dq is None:
        return None
    if dq.op_type == 'DequantizeLinear':
        return dq.input[0]
    if dq.op_type == 'QuantizeLinear':
        return dq.output[0]
    return None


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'fastscnn_int8_qdq_ppmfp32.onnx'
    inp_path  = sys.argv[2] if len(sys.argv) > 2 else 'test_forward_input.bin'

    tensors = parse_mem_h('fastscnn_mem.h')
    print(f"fastscnn_mem.h: {len(tensors)} tensores")

    m = onnx.load(onnx_path)
    m = shape_inference.infer_shapes(m)
    g = m.graph
    producers = {out: n for n in g.node for out in n.output}
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    resolved = []
    missing = []
    for canonical, kind, key in TARGETS:
        if canonical not in tensors:
            missing.append((canonical, 'não está em fastscnn_mem.h'))
            continue
        t = tensors[canonical]
        if kind == 'conv_out':
            conv = find_conv_by_weight(g, key)
            if conv is None:
                missing.append((canonical, f'conv: {key}')); continue
            name = q_after_conv(conv, consumers)

        elif kind == 'ppm_in':
            ppm_out_conv = find_conv_by_weight(g, 'ppm.out.conv.0.weight')
            if ppm_out_conv is None:
                missing.append((canonical, 'ppm.out.conv não achada')); continue
            cur = ppm_out_conv.input[0]
            concat_node = None
            for _ in range(15):
                prod = producers.get(cur)
                if prod is None: break
                if prod.op_type == 'Concat': concat_node = prod; break
                cur = prod.input[0]
            if concat_node is None:
                missing.append((canonical, 'concat do PPM não achado')); continue
            # acha o input de 128 canais (o x)
            x_name = None
            for inp in concat_node.input:
                v = next((v for v in g.value_info if v.name == inp), None)
                if v is None:
                    v = next((v for v in g.input if v.name == inp), None)
                if v is not None:
                    dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
                    if len(dims) >= 4 and dims[1] == 128:
                        x_name = inp; break
            if x_name is None:
                x_name = concat_node.input[0]
            dq = producers.get(x_name)
            if dq is None or dq.op_type != 'DequantizeLinear':
                missing.append((canonical, f'x não é DQ: {dq.op_type if dq else None}'))
                continue
            name = dq.input[0]   # tensor int8

        else:   # input_q
            conv = find_conv_by_weight(g, key)
            if conv is None:
                missing.append((canonical, f'conv: {key}')); continue
            name = q_before_conv(conv, producers)
        if name is None:
            missing.append((canonical, 'sem Q')); continue
        resolved.append((canonical, t['tid'], t['c'], t['h'], t['w'], name))

    if missing:
        print(f"\n{len(missing)} falhas:")
        for c, e in missing: print(f"  {c}: {e}")

    print(f"\n{len(resolved)} tensores resolvidos")
    for canonical, tid, c, h, w, name in resolved[:5]:
        print(f"  {canonical:18s} tid={tid:3d}  <- {name[:55]}")

    # Adiciona como outputs extras
    existing = {o.name for o in g.output}
    for _, _, _, _, _, name in resolved:
        if name not in existing:
            m.graph.output.append(onnx.ValueInfoProto(name=name))

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), so,
                                providers=['CPUExecutionProvider'])

    x = np.fromfile(inp_path, dtype=np.float32).reshape(1, 3, 320, 544)
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {'input': x})
    by_name = dict(zip(out_names, outs))

    os.makedirs('ref', exist_ok=True)
    for canonical, tid, c, h, w, name in resolved:
        arr = by_name[name]
        if arr.dtype != np.int8:
            print(f"  AVISO {canonical}: dtype {arr.dtype} → convertendo pra int8")
        arr = arr.reshape(c, h, w).astype(np.int8)
        arr.tofile(f'ref/T{tid:03d}.bin')
    print(f"\nok: {len(resolved)} refs em ref/")


if __name__ == '__main__':
    main()
