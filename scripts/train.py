import scripts.register_cardd_blocks
from ultralytics import YOLO

cfg = dict(
    data="configs/data_cardd.yaml",
    imgsz=512,
    epochs=150,
    batch=16,
    device=0,
    workers=8,
    seed=42,
    deterministic=True,
    amp=True,
    project="runs/cardd_backbone_compare_fast",
)

runs = [
    ("models/yolo_cardd_resnetlike_lite.yaml",   "A_resnetlike_lite_150epochs"),
    ("models/yolo_cardd_convnextlike_lite.yaml", "B_convnextlike_lite_150epochs"),
    ("models/yolo_cardd_mbv3like_lite.yaml",     "C_mbv3like_lite_150epochs"),
]

for model_path, name in runs:
    print("\n" + "="*70)
    print("TRAIN:", name, "| model:", model_path)
    YOLO(model_path).train(name=name, **cfg)

