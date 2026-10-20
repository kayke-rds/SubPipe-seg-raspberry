# gen_fastscnn_csv.py
import onnx, csv
from onnx import shape_inference

def gen_tensors_csv(onnx_path, csv_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model, strict_mode=True)
    graph = model.graph

    # producers/consumers por nome de tensor
    producer = {}
    consumers = {}
    for i, node in enumerate(graph.node):
        for out in node.output:
            if out:
                producer[out] = i        # op_id
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(i)

    init_names = {init.name for init in graph.initializer}
    output_names = {o.name for o in graph.output}
    input_names  = {i.name for i in graph.input}

    # shapes: value_info + inputs + outputs
    vi = {v.name: v for v in graph.value_info}
    for v in list(graph.input) + list(graph.output):
        vi[v.name] = v
    # initializers também têm shape
    for init in graph.initializer:
        vi[init.name] = None  # tratado à parte

    rows = []
    tid = 0
    for init in graph.initializer:
        elems = 1
        for d in init.dims: elems *= d
        rows.append({
            'tid': tid,
            'onnx_name': init.name,
            'shape': 'x'.join(str(d) for d in init.dims),
            'elems': elems,
            'dtype': onnx.TensorProto.DataType.Name(init.data_type),
            'kind': 'initializer',
            'producer_op': '',
            'consumer_ops': ' '.join(str(c) for c in consumers.get(init.name, [])),
            'io': '',
        })
        tid += 1

    for name, v in vi.items():
        if v is None: continue
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in v.type.tensor_type.shape.dim]
        elems = 1
        elem_type = v.type.tensor_type.elem_type
        dtype = onnx.TensorProto.DataType.Name(elem_type)
        for d in dims: elems *= d
        rows.append({
            'tid': tid,
            'onnx_name': name,
            'shape': 'x'.join(str(d) for d in dims),
            'elems': elems,
            'dtype': dtype,
            'kind': 'activation',
            'producer_op': str(producer.get(name, -1)),
            'consumer_ops': ' '.join(str(c) for c in consumers.get(name, [])),
            'io': 'output' if name in output_names else ('input' if name in input_names else ''),
        })
        tid += 1

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tid','onnx_name','shape','elems','dtype','kind',
                                          'producer_op','consumer_ops','io'])
        w.writeheader(); w.writerows(rows)
    print(f"{csv_path}: {len(rows)} tensores")

def gen_ops_csv(onnx_path, csv_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    # mapear nome → tid para reusar o mesmo id
    name_to_tid = {}
    tid = 0
    for init in graph.initializer:
        name_to_tid[init.name] = tid; tid += 1
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        if v.name not in name_to_tid:
            name_to_tid[v.name] = tid; tid += 1

    rows = []
    for i, node in enumerate(graph.node):
        ins  = [f"t{name_to_tid[x]}" for x in node.input  if x and x in name_to_tid]
        outs = [f"t{name_to_tid[x]}" for x in node.output if x and x in name_to_tid]
        rows.append({
            'op_id': i,
            'op': node.op_type,
            'in_tensors':  ' '.join(ins),
            'out_tensors': ' '.join(outs),
        })

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['op_id','op','in_tensors','out_tensors'])
        w.writeheader(); w.writerows(rows)
    print(f"{csv_path}: {len(rows)} ops")

if __name__ == '__main__':
    gen_tensors_csv('fastscnn_int8_qdq_ppmfp32.onnx', 'fastscnn_tensors_ppmfp32.csv')
    gen_ops_csv   ('fastscnn_int8_qdq_ppmfp32.onnx', 'fastscnn_ops_ppmfp32.csv')
