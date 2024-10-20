# Balık Türü Sınıflandırma Projesi

Bu proje, çeşitli balık türlerini sınıflandırmak için derin öğrenme modelleri kullanmayı amaçlamaktadır. Başlangıçta modelin **aşırı öğrenmeye (overfitting)** eğilimli olduğu ve eğitimin çok uzun sürdüğü gözlemlenmiştir. Modelin mimarisinde ve hiperparametrelerinde çeşitli değişiklikler yapılarak en iyi sonuçlar elde edilmiştir. **BatchNormalization** ve **Dropout** gibi tekniklerle modelin dengeli hale getirilmesi sağlanmıştır.

---

## İçindekiler
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Veri Seti](#veri-seti)
- [Model Yapısı ve Karşılaşılan Problemler](#model-yapısı-ve-karşılaşılan-problemler)
- [Performans ve Sonuçlar](#performans-ve-sonuçlar)



---

## Kullanılan Teknolojiler

Bu proje, Python diliyle geliştirilmiştir ve aşağıdaki kütüphaneler kullanılmıştır:
- **TensorFlow / Keras**: Derin öğrenme modelleri için.
- **NumPy**: Sayısal işlemler için.
- **Pandas**: Veri manipülasyonu için.
- **Matplotlib**: Görselleştirme için.
- **PIL (Python Imaging Library)**: Görselleri işlemek için.
- **tqdm**: Döngülerin ilerlemesini takip etmek için.

---

## Veri Seti

Proje, balık türlerine göre organize edilmiş görsellerden oluşan bir veri seti kullanmaktadır. Veriler, her tür için ayrı klasörlerde tutulur ve aşağıdaki işlemlerden geçirilir:

1. **Boyutlandırma**: Görseller 64x64 piksel boyutuna küçültülmüştür.
2. **RGB Dönüşümü**: Siyah-beyaz görselleri desteklemek için.
3. **Normalizasyon**: Piksel değerleri 0-255 aralığından 0-1 aralığına çekilmiştir.
4. **Veri Bölme**: Eğitim ve test seti olarak %80-%20 oranında ayrılmıştır.

---

## Model Yapısı ve Karşılaşılan Problemler

### İlk Denemeler: Aşırı Öğrenme Problemi

İlk başta Dropout katmanı kullanılmadan yapılan eğitimde model, eğitim verisine aşırı uyum sağladı fakat doğrulama setinde düşük performans gösterdi. Bu durum, modelin aşırı öğrenmeye girdiğini ortaya koydu. **Dropout** eklenerek her katmandaki nöronların %30’u rastgele devre dışı bırakıldı ve modelin genelleme yeteneği artırıldı.

### BatchNormalization’ın Etkisi

Başlangıçta BatchNormalization kullanılmadığında modelin eğitim süresi uzun sürdü. **BatchNormalization** ile veriler her katmandan sonra normalize edilerek eğitim süresi kısaldı ve model daha kararlı hale geldi.

---

### Model Mimarisi

Son olarak kullanılan model mimarisi aşağıdaki gibidir:

```python
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(12288,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(9, activation='softmax'))
```

ReLU Aktivasyonu: Doğrusal olmayan ilişkilerin öğrenilmesini sağlar.
BatchNormalization: Eğitim hızını artırır ve daha kararlı bir model sunar.
Dropout: Aşırı öğrenmeyi önler.
Softmax Aktivasyonu: 9 sınıf için olasılık hesaplar.

## Performans ve Sonuçlar

Aşağıdaki tablo, modelin farklı sürümlerdeki performansını göstermektedir:

| **Model**                                  | **Eğitim Doğruluğu** | **Doğrulama Doğruluğu** | **Eğitim Kaybı** | **Doğrulama Kaybı** |
|--------------------------------------------|----------------------|-------------------------|------------------|--------------------|
| **İlk Model (Dropout olmadan)**             | %98.0                | %72.5                   | 0.10             | 1.15               |
| **BatchNormalization olmadan**              | %96.0                | %80.2                   | 0.15             | 0.95               |
| **Son Model (BatchNormalization ve Dropout ile)** | %95.4           | %93.1                   | 0.156            | 0.193              |

Bu sonuçlar, **BatchNormalization** ve **Dropout** kullanarak daha dengeli bir model elde ettiğimizi ve aşırı öğrenme probleminin çözüldüğünü göstermektedir. 

Son modelin **Test Kayıp Değeri**: `0.1928` ve **Test Doğruluğu**: `%93.1` olarak kaydedilmiştir, bu da modelin yüksek doğrulama doğruluğuna ek olarak test verilerinde de başarılı olduğunu göstermektedir.

