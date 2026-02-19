# 🚗 Car Damage YOLO  
## Vehicle Damage Detection and Severity Estimation using Deep Learning

This project implements deep learning models for vehicle damage detection and severity estimation using YOLO-based architectures. The repository is designed to compare different backbone networks under a unified experimental protocol.

---

## 📌 Project Overview

The main objectives of this project are:

- Detect vehicle damage
- Compare different backbone architectures
- Evaluate computational cost
- Analyze model performance and convergence

All models are trained and evaluated under identical experimental conditions to ensure fair comparison.
---

## 📂 Project Structure

```
car-damage-yolo/
│
├── models/                  # Custom backbones and model definitions
│
├── scripts/
│   ├── train.py             # Main training script
│   ├── download_dataset.py  # Dataset download script
│   ├── pred_new_image.py    # Inference script for new images
│
├── data/                    # Dataset directory
│
└── runs/                    # Training results and logs
```
---

## Run

### 1 Download data

python scripts/download_dataset.py

### 2 Train Models
python scripts/train.py

### 2 Pred using new images Models
python scripts/pred_new_image.py
