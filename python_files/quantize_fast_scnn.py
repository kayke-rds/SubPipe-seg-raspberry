# quantize_fastscnn.py
import os, cv2, numpy as np
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantFormat, QuantType,
)

H, W = 320, 544

MEAN = np.array([0.3551499061927887, 0.5187109166720711, 0.4940478866148916], dtype=np.float32)   # RGB
STD  = np.array([0.11498362917744831, 0.0765838378600218, 0.14597918539958313], dtype=np.float32)

class FastSCNNCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir, num_samples=100):
        self.paths = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.paths.append(os.path.join(image_dir, f))
        self.paths = self.paths[:num_samples]
        self.idx = 0
        print(f"[calib] {len(self.paths)} imagens em {image_dir}")
        print(f"[calib] mean={MEAN}, std={STD}")

    def _letterbox(self, img):
        """Reproduz o letterbox do preprocess.c (bilinear + padding 114 centralizado)."""
        src_h, src_w = img.shape[:2]
        r = min(H / src_h, W / src_w)
        new_h, new_w = int(round(src_h * r)), int(round(src_w * r))
        dh, dw = (H - new_h) // 2, (W - new_w) // 2

        # padding 114 (cinza) igual ao preprocess.c
        canvas = np.full((H, W, 3), 114, dtype=np.uint8)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas[dh:dh + new_h, dw:dw + new_w] = resized
        return canvas

    def get_next(self):
        if self.idx >= len(self.paths):
            return None
        img = cv2.imread(self.paths[self.idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._letterbox(img)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD                    # <-- normalização do treino
        img = np.transpose(img, (2, 0, 1))[None]    # 1,C,H,W
        self.idx += 1
        return {'input': img.astype(np.float32)}

def main():
    quantize_static(
        model_input='fastscnn_fp32.onnx',
        model_output='fastscnn_int8_qdq.onnx',
        calibration_data_reader=FastSCNNCalibrationReader('/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision/train/images/'),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,     # moderno: sem reduce_range, mais precisão
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric':     True,
            'CalibMethod':         'MinMax',
            # PPM: AdaptiveAvgPool + Resize costuma degradar se quantizado
            'OpTypesToExcludeOutputQuantization': ['GlobalAveragePool'],
        },
    )
    print("ok: fastscnn_int8_qdq.onnx")

if __name__ == '__main__':
    main()
