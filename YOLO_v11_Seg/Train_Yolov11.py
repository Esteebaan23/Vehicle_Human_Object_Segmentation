from ultralytics import YOLO

# Cargar el modelo YOLO
model = YOLO("yolo11n-seg.pt")

# Entrenar el modelo con CPU
model.train(data="dataset.yaml", imgsz=640, batch=16, epochs=100, workers=0)
