# Bulanık Mantık ile ALE Tahmini

Bu proje, **Kablosuz Sensör Ağlarında (WSN)** ortalama lokalizasyon hatasını (**ALE**) tahmin etmek amacıyla bir **Mamdani Bulanık Çıkarım Sistemi** (FIS) geliştirmektedir.  
Tahmin sürecinde **üçgen** ve **gauss** üyelik fonksiyonları ile birlikte **iki farklı berraklaştırma yöntemi** (**centroid** ve **ağırlıklı ortalama**) kullanılmıştır.

## 📌 Proje Açıklaması

Kablosuz sensör ağlarında düğümlerin konumlarının doğru bir şekilde belirlenmesi kritik öneme sahiptir.  
Bu projede aşağıdaki dört girdi bilgisine göre ALE tahmini yapılmıştır:

- **Anchor Ratio:** Konumu bilinen düğüm oranı
- **Transmission Range:** Sensörün kapsama alanı
- **Node Density:** Düğüm/sensör yoğunluğu
- **Iteration Count:** Yineleme sayısı

Veriler normalize edilmiş, 10 adet "Eğer ... ise ..." kuralı tanımlanmış ve farklı kombinasyonlar denenerek modelin başarımı MAE ve RMSE ile ölçülmüştür.

## 🧠 Özellikler

- Üçgen ve Gauss üyelik fonksiyonları
- Centroid ve Ağırlıklı Ortalama berraklaştırma yöntemleri
- 10 kurallı bulanık mantık sistemi
- MAE ve RMSE performans ölçümü
- Grafikle performans görselleştirme

## 📊 Sonuçlar

| Kombinasyon                 | MAE   | RMSE  |
|----------------------------|-------|-------|
| Üçgen + Centroid           | 0.674 | 0.766 |
| Üçgen + Ağırlıklı Ortalama | 0.679 | 0.770 |
| Gauss + Centroid           | 0.712 | 0.814 |
| Gauss + Ağırlıklı Ortalama | 0.711 | 0.814 |

📌 En iyi sonuç **Üçgen + Centroid** kombinasyonuyla elde edilmiştir.

## 📂 Proje Dosyaları

- `main.py` : Python kodu
- `*.csv` : Kullanılan veri seti
- `kombinasyon_karsilastirma.png` : Sonuç grafiği
- `rapor.docx` : Akademik proje raporu
- `README.md` : Proje açıklaması

## 🔧 Gereksinimler

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-fuzzy`
- `scikit-learn`
