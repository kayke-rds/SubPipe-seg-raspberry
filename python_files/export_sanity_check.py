# check_onnx.py
import numpy as np
import onnxruntime as ort
import torch
from fast_scnn import FastSCNN

CKPT = './torch_checkpoints/fast-scnn.pt'
ONNX = 'fastscnn_fp32.onnx'
H, W = 320, 544

# --- PyTorch ---
model = FastSCNN(num_classes=1, aux=False)
sd = torch.load(CKPT, map_location='cpu')['model_state_dict']
sd = {k: v for k, v in sd.items() if not k.startswith('auxlayer.')}
model.load_state_dict(sd, strict=True)
model.eval()

x = torch.randn(1, 3, H, W)
with torch.no_grad():
    y_pt = model(x)[0].numpy()

# --- ONNX ---
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
y_onnx = sess.run(None, {'input': x.numpy()})[0]

print("shapes:", y_pt.shape, y_onnx.shape)
print("max abs diff :", np.max(np.abs(y_pt - y_onnx)))
print("mean abs diff:", np.mean(np.abs(y_pt - y_onnx)))
print("max |pt|     :", np.max(np.abs(y_pt)))
