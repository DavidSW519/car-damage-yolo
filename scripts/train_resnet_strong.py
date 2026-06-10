from ultralytics import YOLO
import scripts.register_cardd_blocks


DATA = "configs/data_cardd.yaml"
MODEL = "models/yolo_cardd_resnetlike_strong.yaml"


def main():
    model = YOLO(MODEL)

    model.train(
        data=DATA,
        task="detect",
        imgsz=640,
        epochs=200,
        batch=16,
        patience=40,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        seed=42,
        deterministic=True,
        amp=True,
        project="runs/cardd_resnet_strong",
        name="resnetlike_strong",

        # Online augmentation de YOLO
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=8.0,
        translate=0.12,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.10,
        close_mosaic=20,
        erasing=0.25,
    )


if __name__ == "__main__":
    main()