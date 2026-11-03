from ultralytics import YOLO

model = YOLO("/home/Cake/treino_redes_tcc_exp2/Comparative-study-of-cnn-based-architectures-on-subpipe/runs/segment/runs/yolo26_31_10/fold4/weights/best.pt")

model.export(
    format="onnx",
    imgsz=(320,320),
    batch=1,
    nms=False,
    opset=12,
    data="/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision+Marina/test-data.yaml"
)
