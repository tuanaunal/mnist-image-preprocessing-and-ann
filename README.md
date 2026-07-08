# MNIST ile Görüntü Ön İşleme ve Yapay Sinir Ağları (ANN) Projesi

Bu proje, derin öğrenme dünyasının klasik başlangıç taşı olan **MNIST el yazısı rakam veri setini** kullanarak, ham görüntülerin yapay sinir ağlarına verilmeden önce hangi ön işleme aşamalarından geçtiğini ve bir Yapay Sinir Ağı (ANN) modelinin nasıl inşa edilip eğitildiğini adım adım uygulamalı olarak göstermektedir.

---

## Proje İçeriği ve Yol Haritası

- **Görüntü Ön İşleme (Image Preprocessing):** Ham piksellerin normalizasyonu ve matris formuna getirilmesi.
- **Mimarinin Kurulması:** `Sequential` ve tamamen bağlı `Dense` katmanları ile yapay sinir ağı tasarımı.
- **Ezberlemeyi Önleme:** `Dropout` katmanları ile modelin kararlılığının artırılması.
- **Optimizasyon:** `Adam` optimizer motoru ile hata oranının (loss) minimize edilmesi.

---

## Kullanılan Teknolojiler ve Araçlar

- **Dil:** Python 3.12
- **Geliştirme Ortamı:** `venv` (Virtual Environment) ile izole laboratuvar odası
- **Kütüphaneler:**
  - `TensorFlow / Keras` (Derin Öğrenme Motoru)
  - `OpenCV` (Görüntü İşleme)
  - `NumPy` (Yüksek Performanslı Matris İşlemleri)
  - `Matplotlib` (Eğitim Grafikleri ve Görselleştirme)

---

## Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için:

1. Depoyu klonlayın veya indirin.
2. Proje dizininde terminali açıp sanal ortamı aktif edin:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
