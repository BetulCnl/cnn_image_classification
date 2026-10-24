# CNN ile Hayvan Sınıflandırma

Sıfırdan yazılmış (hazır model kullanılmayan) bir CNN ile 10 hayvan sınıfını ayırt eden PyTorch projesi. Eğitilen modeli denemek için Streamlit arayüzü de var.

Veri seti: [Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

## Sınıflar

Klasör adları veri setinden geldiği için İtalyanca:

`cane` köpek · `cavallo` at · `elefante` fil · `farfalla` kelebek · `gallina` tavuk · `gatto` kedi · `mucca` inek · `pecora` koyun · `ragno` örümcek · `scoiattolo` sincap

## Model

- Girdi: 224×224 RGB, değerler [0, 1] aralığında (ek normalizasyon yok)
- 4 blok `Conv-BN-ReLU-Conv-BN-ReLU-MaxPool` (32 → 64 → 128 → 128 kanal), arada `Dropout2d(0.1)`
- Global average pooling, ardından `Linear(128→256) → ReLU → Dropout(0.35) → Linear(256→10)`
- Kayıp: CrossEntropy (label smoothing 0.1)
- Optimizer: AdamW (lr 3e-4, weight decay 1e-4), her 4 epoch'ta lr yarıya iner
- 12 epoch, batch 32. En düşük validation loss'u veren ağırlıklar saklanır.
- Eğitimde augmentasyon: RandomResizedCrop, yatay çevirme, ColorJitter

Eğitilmiş ağırlıklar `models/best_custom_cnn.pth` dosyasında (yaklaşık 2.5 MB).

## Kurulum

`requirements.txt` yok, paketleri elle kurmak gerekiyor:

```bash
pip install torch torchvision streamlit pillow scikit-learn tqdm
```

## Sadece arayüzü çalıştırmak

```bash
streamlit run app.py
```

`http://localhost:8501` adresinde açılır. Bir jpg/png yükleyip "Tahmin Et"e basınca tahmin edilen sınıf, güven oranı ve en olası 3 sınıf gösterilir.

## Modeli yeniden eğitmek

1. Animals-10'u indir, sınıf klasörlerini `data/raw/animals10/` altına koy.
2. Veriyi böl ve 224×224'e küçült (%70 train, %15 val, %15 test):
   ```bash
   python scripts/prepare_data.py
   ```
   Sonuç `data/processed/{train,val,test}/` altına yazılır.
3. Eğit:
   ```bash
   python train.py
   ```
   Test klasörü varsa eğitim sonunda en iyi modelle test doğruluğu da yazdırılır.

`data/` klasörü boyutu yüzünden repoya eklenmedi.

## Dosyalar

```
app.py                  Streamlit arayüzü
train.py                model tanımı + eğitim döngüsü
scripts/prepare_data.py veri bölme ve yeniden boyutlandırma
scripts/test_only.py    önceki ResNet18 denemesinden kalan test betiği
models/                 eğitilmiş ağırlıklar
```

`scripts/test_only.py` repoda olmayan `models/best_resnet18.pth` dosyasını arıyor, mevcut özel CNN ile çalışmaz.
