# export_onnx.py
import torch
from fast_scnn import FastSCNN

def export_fastscnn_onnx(checkpoint_path, onnx_path, H=320, W=544):
    model = FastSCNN(num_classes=1, aux=False)
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    sd = ckpt['model_state_dict']
    # remove tudo que pertence ao ramo auxiliar
    sd = {k: v for k, v in sd.items() if not k.startswith('auxlayer.')}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    assert not missing and not unexpected, "state_dict não bateu"

    model.eval()

    dummy = torch.randn(1, 3, H, W)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=18,
        input_names=['input'], output_names=['output'],
        dynamic_axes=None,
    )
    print(f"ONNX FP32 exportado: {onnx_path}")

if __name__ == '__main__':
    export_fastscnn_onnx(
        checkpoint_path='./torch_checkpoints/fast-scnn.pt',
        onnx_path='fastscnn_fp32.onnx',
    )
