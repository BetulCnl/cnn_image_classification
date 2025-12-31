from pathlib import Path
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#print("prepare_data.py çalış tı")

RAW_DIR = Path("data/raw/animals10")
OUT_DIR = Path("data/processed")

IMG_SIZE = (224, 224)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def list_images(cls_dir: Path):
    """Bir sınıf klasöründeki tüm görselleri (jpg/jpeg/png) topla (case-insensitive)."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images += list(cls_dir.glob(ext))
    return images

def prepare_data():
    # Ham veri klasörü kontrol
    print("RAW_DIR:", RAW_DIR.resolve())
    print("RAW_DIR var mı?", RAW_DIR.exists())

    # Çıkış klasörünü garanti oluştur
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sınıf klasörlerini bul
    classes = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    print("Bulunan sınıf sayısı:", len(classes))

    # train/val/test klasörlerini oluştur
    for split in ["train", "val", "test"]:
        for cls in classes:
            (OUT_DIR / split / cls.name).mkdir(parents=True, exist_ok=True)

    # Her sınıfı işle
    for cls in classes:
        images = list_images(cls)

        if len(images) == 0:
            print(f"Uyarı: {cls.name} içinde jpg/jpeg/png bulunamadı!")
            continue

        random.shuffle(images)

        # 1) train ve kalan (val+test)
        train_imgs, temp_imgs = train_test_split(
            images, test_size=(1 - TRAIN_RATIO), random_state=42
        )

        # 2) kalanları val ve test diye ayır
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            random_state=42
        )

        splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

        # Resize + kaydet
        for split, imgs in splits.items():
            for img_path in tqdm(imgs, desc=f"{cls.name} - {split}"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMG_SIZE)

                    save_path = OUT_DIR / split / cls.name / img_path.name
                    img.save(save_path)
                except Exception as e:
                    print(f"Hata: {img_path} okunamadı/kaydedilemedi -> {e}")

    print("Veri hazırlama (split + resize) tamamlandı.")

if __name__ == "__main__":
    prepare_data()
