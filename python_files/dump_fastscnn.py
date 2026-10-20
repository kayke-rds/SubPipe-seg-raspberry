# dump_fastscnn.py — corrigido
import onnx, json, numpy as np
from onnx import numpy_helper

model = onnx.load('fastscnn_int8_qdq_ppmfp32.onnx')
graph = model.graph
init = {i.name: i for i in graph.initializer}

qmap = {}
for node in graph.node:
    if node.op_type not in ('QuantizeLinear', 'DequantizeLinear'):
        continue
    x_name = node.input[0]
    s_name = node.input[1]
    zp_name = node.input[2] if len(node.input) > 2 else None

    scale = numpy_helper.to_array(init[s_name]).tolist() if s_name in init else None
    zp    = numpy_helper.to_array(init[zp_name]).tolist() if zp_name in init else None

    is_weight = x_name in init and init[x_name].data_type == onnx.TensorProto.INT8

    qmap.setdefault(x_name, {})
    qmap[x_name].update({
        'scale': scale,
        'zero_point': zp,
        'role': 'weight' if is_weight else 'activation',
        'op': node.op_type,
    })

pesos_q = [n for n in qmap if qmap[n]['role'] == 'weight']
ativ_q  = [n for n in qmap if qmap[n]['role'] == 'activation']

print(f"pesos quantizados : {len(pesos_q)}")
print(f"ativas quantizadas: {len(ativ_q)}")
print(f"dtypes de pesos   : {set(onnx.TensorProto.DataType.Name(init[n].data_type) for n in pesos_q)}")

# ---- dump correto: só usa init[] para pesos ----
entries = []
for n, info in qmap.items():
    if info['role'] == 'weight':
        t = init[n]
        entries.append({
            'name':    t.name,
            'role':    'weight',
            'dims':    list(t.dims),
            'dtype':   onnx.TensorProto.DataType.Name(t.data_type),
            'scale':   info['scale'],
            'zero_point': info['zero_point'],
            'scale_len': len(info['scale']) if isinstance(info['scale'], list) else 1,
        })
    else:
        entries.append({
            'name':    n,
            'role':    'activation',
            'scale':   info['scale'],
            'zero_point': info['zero_point'],
            'op':      info['op'],
        })

with open('model_map.json', 'w') as f:
    json.dump({'tensors': entries}, f, indent=2)

# ---- sumário dos pesos ----
print("\nprimeiros 8 pesos:")
for e in [x for x in entries if x['role']=='weight'][:8]:
    s = e['scale'] if isinstance(e['scale'], list) else [e['scale']]
    print(f"  {e['name'][:60]:60s} dims={e['dims']} "
          f"scale_len={e['scale_len']} scale_range=[{min(s):.2e},{max(s):.2e}] "
          f"zp={e['zero_point']}")

print("\nprimeiras 8 ativações:")
for e in [x for x in entries if x['role']=='activation'][20:50]:
    s = e['scale'] if isinstance(e['scale'], list) else [e['scale']]
    print(f"  {e['name'][:60]:60s} op={e['op']:18s} "
          f"scale={s if not isinstance(s, list) or len(s)>3 else [f'{v:.2e}' for v in s]} "
          f"zp={e['zero_point']}")
