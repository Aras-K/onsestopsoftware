from ultralytics import YOLO

model = YOLO("yolo11n.pt")

result = model.train(data="/Users/aras.koplanto/Documents/OneStop/trainingset/data.yaml", epochs=100, lr0=0.01)

