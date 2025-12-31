Yapay Zeka Destekli Görüntü Sınıflandırıcı

Bu proje, derin öğrenme (Deep Learning) teknikleri kullanılarak eğitilmiş bir Convolutional Neural Network (CNN) modeli ile görüntü sınıflandırması yapmaktadır.
Eğitilen model, Streamlit tabanlı kullanıcı dostu bir web arayüzü üzerinden görsel alarak tahmin sonucunu kullanıcıya sunar.

 Proje Amacı

Görsel verileri ön işleme adımlarından geçirerek, eğitilmiş bir CNN modeli yardımıyla doğru şekilde sınıflandırmak ve bu süreci kullanıcı dostu, etkileşimli bir arayüz ile sunmak.

- Öne Çıkan Özellikler

Gelişmiş CNN Mimarisi
Özel olarak tasarlanmış ve eğitilmiş Custom CNN modeli kullanılmıştır.

Görüntü Ön İşleme
Yeniden boyutlandırma, normalize etme ve veri uyumluluğu kontrolleri.

Streamlit Web Arayüzü
Gerçek zamanlı tahmin, sade tasarım ve kolay kullanım.

Detaylı Çıktılar
En yüksek olasılıklı sınıf ve tahmin sonucunun kullanıcıya açık şekilde gösterimi.

Modüler Kod Yapısı
Eğitim, model ve arayüz bileşenleri ayrı dosyalar halinde düzenlenmiştir.

 Desteklenen Hayvan Sınıfları

Model aşağıdaki 10 hayvan sınıfını tanıyabilmektedir:

🐶 Köpek (cane)

🐱 Kedi (gatto)

🐮 İnek (mucca)

🐴 At (cavallo)

🐑 Koyun (pecora)

🐔 Tavuk (gallina)

🐘 Fil (elefante)

🦋 Kelebek (farfalla)

🕷️ Örümcek (ragno)

🐿️ Sincap (scoiattolo)

⚙️ Teknik Detaylar

Girdi Boyutu: 224 × 224 × 3

Normalizasyon: [0, 1] aralığında

Kayıp Fonksiyonu: Cross Entropy Loss

Optimizasyon: Adam Optimizer

Değerlendirme Metrikleri: Accuracy, Validation Loss

📂 Proje Yapısı

app.py → Streamlit tabanlı web arayüzü

train.py → CNN modelinin eğitim kodu

scripts/ → Yardımcı fonksiyonlar

data/ → train / val / test veri setleri (GitHub’a eklenmedi)

models/ → Eğitilmiş model dosyalarından yalnızca en son kullanılan best custom model yüklendi

🚀 Kurulum ve Çalıştırma
1️⃣ Gerekli Kütüphaneleri Yükleyin

pip install -r requirements.txt

2️⃣ (Opsiyonel) Modeli Eğitin

python train.py

3️⃣ Uygulamayı Başlatın

python -m streamlit run app.py

Uygulama varsayılan olarak şu adreste açılır:
http://localhost:8501

🧪 Örnek Kullanım Akışı

Web arayüzünü açın

Bir görüntü yükleyin (jpg / png)

Tahmin butonuna basın

Modelin sınıflandırma sonucunu ekranda görüntüleyin

📌 Notlar

Veri seti ve büyük boyutlu dosyalar GitHub’a eklenmemiştir.

Projede, örnek çalıştırma amacıyla eğitilmiş best_custom_cnn.pth modeli paylaşılmıştır.

Model, farklı veri setleriyle yeniden eğitilebilir şekilde tasarlanmıştır.

Model [Animals-10 Dataset] (https://www.kaggle.com/datasets/alessiocorrado99/animals10) kullanılarak eğitilmiştir.
