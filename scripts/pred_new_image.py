import sys
from ultralytics import YOLO
import scripts.register_cardd_blocks 

weights = "/content/car-damage-yolo/runs/A_resnetlike_lite_150epochs/weights/best.pt"

img_path = "data/miata.jpg"
model = YOLO(weights)

results = model.predict(
    source=img_path,
    imgsz=512,
    conf=0.25,
    iou=0.5,
    save=True,
    show_labels=True,
    show_conf=True
)
print("pred ok in", results[0].save_dir)
