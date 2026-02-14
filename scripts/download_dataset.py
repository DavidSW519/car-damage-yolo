import shutil
from pathlib import Path
import kagglehub

DATASET = "gabrielfcarvalho/cardd-with-yolo-annotations-images-labels"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TARGET_DIR = DATA_DIR / "cardd"

def main():
    print("Descargando dataset desde KaggleHub...")
    dataset_path = kagglehub.dataset_download(DATASET)

    dataset_path = Path(dataset_path)
    print(f"Dataset descargado en cache: {dataset_path}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("Copiando dataset a carpeta del proyecto...")
    for item in dataset_path.iterdir():
        dest = TARGET_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"\n Dataset listo en: {TARGET_DIR}")

    # Verificacion
    images = list(TARGET_DIR.rglob("images"))
    labels = list(TARGET_DIR.rglob("labels"))

    print(f"Encontradas carpetas images: {len(images)}")
    print(f"Encontradas carpetas labels: {len(labels)}")

if __name__ == "__main__":
    main()
