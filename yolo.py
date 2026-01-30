from ultralytics import YOLO

model = YOLO("yolo11n.pt") # type: ignore

results = model("Wo humsafar se baate .jpg",save=True)
results[0].show()