import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ...............................................................
# 1
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_custom_cnn.pth"

IMG_SIZE = 224
CLASSES = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 
           'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# ---------------------------------------------------------------
# 2 cnn bloklar 

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
            nn.Linear(256, len(CLASSES)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# 3 UTILS

@st.cache_resource
def load_model():
    """Loads the model once and caches it to avoid reloading on every interaction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(num_classes=len(CLASSES)).to(device)
    
    if not BEST_MODEL_PATH.exists():
        st.error(f"Model file not found at: {BEST_MODEL_PATH}")
        return None, None
    
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def process_image(image):
    """Preprocesses the PIL image for the model."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
#4 STREAMLIT APP
def main():
    st.set_page_config(page_title="Hayvan Sınıflandırma", page_icon="🐾")

    st.title("🐾 Hayvan Sınıflandırma Modeli")
    st.markdown("""
    Bu uygulama, yüklediğiniz görüntüyü analiz ederek aşağıdaki 10 sınıftan hangisine ait olduğunu tahmin eder:
    
    `cane` (köpek), `cavallo` (at), `elefante` (fil), `farfalla` (kelebek), `gallina` (tavuk),
    `gatto` (kedi), `mucca` (inek), `pecora` (koyun), `ragno` (örümcek), `scoiattolo` (sincap)
    """)

    # Load model
    model, device = load_model()
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Bir resim yükleyin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Yüklenen Resim', use_column_width=True)
            
            # Predict button
            if st.button('Tahmin Et'):
                with st.spinner('Analiz ediliyor...'):
                    # Preprocess
                    img_tensor = process_image(image).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        logits = model(img_tensor)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        
                        # Get top prediction
                        conf, pred_idx = torch.max(probs, 1)
                        predicted_class = CLASSES[pred_idx.item()]
                        confidence = conf.item() * 100

                    # Display Result
                    st.success(f"Tahmin: **{predicted_class.upper()}**")
                    st.progress(int(confidence))
                    st.info(f"Doğruluk Oranı: %{confidence:.2f}")

                    # Show details for top 3
                    st.subheader("Diğer Olasılıklar:")
                    top3_conf, top3_idx = torch.topk(probs, 3)
                    for i in range(3):
                        cls_name = CLASSES[top3_idx[0][i].item()]
                        cls_conf = top3_conf[0][i].item() * 100
                        st.write(f"- {cls_name}: %{cls_conf:.2f}")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
