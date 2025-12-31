# train.py  (CUSTOM CNN - ÖDEV İÇİN, RESNET YOK)

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "best_custom_cnn.pth"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12

LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

NUM_WORKERS = 0   
PRINT_EVERY = 10

# Hızlı deneme için YORUM STIRINA ALDIM
# MAX_TRAIN_BATCHES = 50
# MAX_VAL_BATCHES = 20
MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#TRANSFORMLAR

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# DATADTAADDATDATTADA
#3 DATASET / DATALOADER

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Train klasörü yok: {TRAIN_DIR}")
if not VAL_DIR.exists():
    raise FileNotFoundError(f"Val klasörü yok: {VAL_DIR}")

train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
val_ds   = datasets.ImageFolder(str(VAL_DIR), transform=eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=False)

test_loader = None
if TEST_DIR.exists():
    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)
    print("Test set bulundu. Test size:", len(test_ds))
else:
    print("Uyarı: data/processed/test yok -> test aşaması atlanacak.")

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)
print("Num classes:", num_classes)
print("Train size:", len(train_ds), "| Val size:", len(val_ds))

# BEST CUSTOM CNN 
# BU MİMARİMDE ÖNCEKİ DİĞER 2 SİNDEN OPTİMİZASYON ALG. LARI VE POOLING ÇOK FARKLI

class CustomCNN(nn.Module):
    """
    Ödev için: tamamen custom mimari.
    Conv-BN-ReLU-Conv-BN-ReLU-Pool blokları + GAP + MLP.
    """
    def __init__(self, num_classes: int):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),     # 224 -> 112
            block(32, 64),    # 112 -> 56
            block(64, 128),   # 56 -> 28
            nn.Dropout2d(0.10),
            block(128, 128),  # 28 -> 14  (bir blok daha: kapasite artar)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (B,128,1,1)

        self.classifier = nn.Sequential(
            nn.Flatten(),            # (B,128)
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes=num_classes).to(device)


#loss optimizasyon scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

#------------------------------------------------------------------------------
#YARDIMCI FONKSİYONLAR

def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def run_train_epoch(epoch_idx: int):
    model.train()
    loss_sum, acc_sum, batches = 0.0, 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(logits, labels)
        loss_sum += loss.item()
        acc_sum += acc
        batches += 1

        if batches % PRINT_EVERY == 0:
            print(f"train batch {batches} | avg_loss={loss_sum/batches:.4f} avg_acc={acc_sum/batches:.4f}")

        if MAX_TRAIN_BATCHES is not None and batches >= MAX_TRAIN_BATCHES:
            print(f"(Hızlı deneme) Train {MAX_TRAIN_BATCHES} batch ile sınırlandı.")
            break

    return loss_sum / batches, acc_sum / batches

@torch.no_grad()
def run_eval(split_name: str, loader: DataLoader):
    model.eval()
    loss_sum, acc_sum, batches = 0.0, 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        acc = accuracy_from_logits(logits, labels)

        loss_sum += loss.item()
        acc_sum += acc
        batches += 1

        if batches % PRINT_EVERY == 0:
            print(f"{split_name} batch {batches} | avg_loss={loss_sum/batches:.4f} avg_acc={acc_sum/batches:.4f}")

        if split_name == "val" and MAX_VAL_BATCHES is not None and batches >= MAX_VAL_BATCHES:
            print(f"(Hızlı deneme) Val {MAX_VAL_BATCHES} batch ile sınırlandı.")
            break

    return loss_sum / batches, acc_sum / batches


#TRAIN VE HANGİ MODELİ SAKLAYACAĞIMA KARAR (VALL LOSS EN DÜŞÜK

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Epoch {epoch}/{EPOCHS} başladı ---")

    train_loss, train_acc = run_train_epoch(epoch)
    val_loss, val_acc = run_eval("val", val_loader)

    print(f"Epoch {epoch} SONUCU | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # En iyi modeli val_loss'a göre sakla
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ En iyi model güncellendi -> {BEST_MODEL_PATH} (best_val_loss={best_val_loss:.4f})")

    scheduler.step()

print("\nEğitim bitti. En iyi model:", BEST_MODEL_PATH)

#-----------------------------
#TEST

if test_loader is not None:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    test_loss, test_acc = run_eval("test", test_loader)
    print(f"\nTEST SONUCU | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

print("\nBitti.")

