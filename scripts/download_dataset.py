import os
import subprocess
from pathlib import Path

DATASET = "gabrielfcarvalho/cardd-with-yolo-annotations-images-labels"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "cardd"
ZIP_PATH = DATA_DIR / "cardd.zip"

def run(cmd: str):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1) Check Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"No se encontró {kaggle_json}.\n"
            "En Colab: sube kaggle.json y muévelo a ~/.kaggle/kaggle.json con permisos 600."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 2) Download (zip)
    if not ZIP_PATH.exists():
        run(f'kaggle datasets download -d {DATASET} -p "{DATA_DIR}" -f "*.zip"')
        # Kaggle guarda con nombre del dataset; renombramos al fijo
        zips = sorted(DATA_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zips:
            raise RuntimeError("No se encontró zip descargado en data/.")
        zips[0].rename(ZIP_PATH)
    else:
        print(f"Zip ya existe: {ZIP_PATH}")

    # 3) Unzip
    marker = OUT_DIR / ".unzipped"
    if not marker.exists():
        run(f'unzip -q "{ZIP_PATH}" -d "{OUT_DIR}"')
        marker.write_text("ok")
    else:
        print("Ya se descomprimió antes, saltando unzip.")

    # 4) Basic structure checks
    # Algunas versiones quedan en subcarpeta; detectamos donde estén images/labels
    candidates = [OUT_DIR] + [p for p in OUT_DIR.iterdir() if p.is_dir()]
    found = None
    for c in candidates:
        if (c / "images").exists() and (c / "labels").exists():
            found = c
            break

    if not found:
        raise RuntimeError(
            f"No encontré estructura images/ y labels/ dentro de {OUT_DIR}.\n"
            "Revisa el contenido de data/cardd/."
        )

    print(f"\n✅ Dataset listo en: {found}")
    print(f"   - images/: {len(list((found/'images').rglob('*.*')))} archivos")
    print(f"   - labels/: {len(list((found/'labels').rglob('*.txt')))} txt")

if __name__ == "__main__":
    main()
