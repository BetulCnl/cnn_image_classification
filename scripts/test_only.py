import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =========================
# PATH AYARLARI (OTOMATİK)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # cnn_odev/scripts
BASE_DIR = os.path.dirname(SCRIPT_DIR)                    # cnn_odev

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet18.pth")

# =========================
# AYARLAR
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows güvenli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# TRANSFORM (EĞİTİMLE AYNI)
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================
# KONTROLLER
# =========================
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Test klasörü bulunamadı: {TEST_DIR}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

# =========================
# DATASET & LOADER
# =========================
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("Test size:", len(test_dataset))
print("Classes:", test_dataset.classes)

# =========================
# MODEL (RESNET18)
# =========================
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))
model = model.to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

criterion = nn.CrossEntropyLoss()

# =========================
# TEST LOOP
# =========================
total_loss = 0.0
total_correct = 0
total = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader, 1):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)

        if i % 10 == 0:
            print(
                f"test batch {i} | "
                f"avg_loss={total_loss/total:.4f} "
                f"avg_acc={total_correct/total:.4f}"
            )

print("\n=== TEST SONUCU ===")
print(f"loss={total_loss/total:.4f}")
print(f"acc ={total_correct/total:.4f}")
