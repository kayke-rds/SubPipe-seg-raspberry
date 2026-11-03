from ultralytics import YOLO

model = YOLO("/home/Cake/treino_redes_tcc_exp2/runs/segment/runs/yolo26-backbone-freeze-square-image/fold3/weights/best.pt")

model.export(
    format="onnx",
    imgsz=(320,320),
    batch=1,
    nms=False,
    opset=12,
    data="/home/Cake/Documentos/UFRB/TCC/SubPipeOriginalDivision+Marina/test-data.yaml"
)
