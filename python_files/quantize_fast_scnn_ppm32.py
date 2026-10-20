# quantize_fastscnn_exclude_ppm.py
import onnx
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
from quantize_fast_scnn import FastSCNNCalibrationReader

m = onnx.load('fastscnn_fp32.onnx')

# PPM: nós 61 (primeiro ReduceMean) a 174 (último Relu do self.out)
PPM_START, PPM_END = 61, 175   # range semiaberto
nodes_to_exclude = [n.name for n in m.graph.node[PPM_START:PPM_END]]
print(f"excluindo {len(nodes_to_exclude)} nós do PPM")
print("primeiros:", nodes_to_exclude[:5])
print("últimos :", nodes_to_exclude[-5:])

quantize_static(
    model_input='fastscnn_fp32.onnx',
    model_output='fastscnn_int8_qdq_ppmfp32.onnx',
    calibration_data_reader=FastSCNNCalibrationReader(
        '/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision/train/images/'
    ),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=True,
    nodes_to_exclude=nodes_to_exclude,
)
