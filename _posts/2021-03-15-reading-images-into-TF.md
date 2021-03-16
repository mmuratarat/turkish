---
layout: post
title:  "Derin Öğrenme Modelleri için TensorFlow'a Görüntü Veri Kümelerini Yükleme"
author: "MMA"
comments: true
---

Evrişimsel sinir ağı (Convolutional Neural Network), derin öğrenme sinir ağlarının bir sınıfıdır. CNN'ler, görüntü tanımada büyük yenilikler getirmektedir. Genellikle görsel görüntüleri analiz etmek üzere çeşitli bilgisayarlı görü görevleri için kullanılırlar, örneğin, görüntü sınıflandırma, nesne tanıma, görüntü bölütleme v.b. Facebook'un fotoğraf etiketlemesinden tutun da sürücüsüz arabalara kadar her şeyin merkezinde bulunabilir. Sağlık hizmetlerinden internet güvenliğine kadar her konuda perde arkasında yer alırlar. Hızlıdırlar ve verimlidirler. Ancak, bu alana yeni giren birinin ilk sorduğu soru, bir bilgisayarlı görü görevi için bir CNN modelini nasıl eğitiriz?

Evrişimsel bir sinir ağını eğitirken görüntü verilerinin en iyi şekilde nasıl hazırlanacağını bilmek zordur. Bu, modelin hem eğitimi hem de değerlendirilmesi sırasında hem piksel değerlerinin ölçeklendirilmesini hem de görüntü verisi çeşitlendirme (data augmentation) tekniklerinin kullanılmasını içerir.

Modelinizi eğitmek için bir görüntüyü bir CNN'e nasıl besleyeceğinizi öğrenmek istiyorsunuz. Bunu yapmak için, eğitim setinizdeki görüntüleri vektörleştirilmiş bir forma dönüştürmeniz (bir dizi veya matrise çevirmeniz) gerekir. Bunu yapma yönteminiz kullandığınız dile ve/veya yazılım iskeletine bağlıdır (örn. Numpy, Tensorflow, Scikit Learn, vb.). En önemlisi, bunu yapmaya ne karar verirseniz verin, söz konusu yöntemin hem eğitim hem de test boyunca tutarlı olmasıdır. İşte bu nedenle, iki yazılık bu seride önce görüntü verilerini TensorFlow ortamına kolaylıkla nasıl okutulacağından daha sonra okutulan bu görüntüleri modelin performansını arttırması açısından nasıl çeşitlendirilebileceğinden bir uygulama ile bahsedeceğim.

İlk olarak gerekli paketleri ve modülleri içe aktaralım:

```python
import tensorflow as tf
print(tf.__version__)
#2.4.1
import pathlib
import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
``` 
Yapılan işlemleri göstermek için, Google tarafından yayınlanan `flowers` veri setini kullanacağız. Bu veri kümesini indirmek, aşağıdaki kod satırını çalıştırmak kadar kolaydır.

Aşağıdaki kodu çalıştırdıktan sonra oluşacak `flower_photos` klasöründe `daisy`, `roses`, `sunflowers`, `dandelion`, `tulips` isimli 5 alt klasör ve bir `License.txt` dosyası olacaktır.

`flowers`, veri kümesinin indirildiği yolu (benim durumumda - `/Users/mustafamuratarat/.keras/datasets`) içerir. Veri kümesinin yapısı aşağıdaki gibi olacaktır:

```
flower_photos
      ├── daisy [633 görüntü]
      ├── dandelion [898 görüntü]
      ├── roses [641 görüntü]
      ├── sunflowers [699 görüntü]
      ├── tulips [799 görüntü]
      └── LICENSE.txt
```

```python
# Get the flowers dataset
flowers = tf.keras.utils.get_file(
    fname='flower_photos',
    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

# Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
# 228818944/228813984 [==============================] - 380s 2us/step

# print(flowers)
# '/Users/mustafamuratarat/.keras/datasets/flower_photos'

data_dir = pathlib.Path(flowers)
data_dir
#PosixPath('/Users/mustafamuratarat/.keras/datasets/flower_photos')
```

İndirdikten sonra (240.6MB), artık çiçek fotoğraflarının bir kopyasına sahip olmalısınız. Toplam 3670 resim vardır:

```python
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
#3670
```

## tf.keras.preprocessing.image_dataset_from_directory kullanarak görüntüleri okumak

`tf.keras.preprocessing.image_dataset_from_directory` kullanarak görüntü veri kümenizi direkt olarak herhangi bir dizinden okutabilirsiniz. Dökümantasyonunu [burada](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory) bulabilirsiniz. Bu fonksiyon dizindeki görüntü dosyalarından bir `tf.data.Dataset` oluşturur.

Sözdizimi aşağıdaki gibidir:

```
tf.keras.preprocessing.image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(256,
    256), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False
)
```

Bu fonksiyon ile desteklenen resim formatları: jpeg, png, bmp, gif'dir. Animasyonlu gifler ilk kareye kesilir.

Buradaki bazı argümanlara göz atalım. 

`directory`, verinin yer aldığı dizindir. Eğer `labels` argümanı  `inferred` olarak ayarlanmış ise, ana dizin altında her biri bir sınıfa ait görüntüleri içeren alt dizinler olmalıdır. `flower_photos` ana dizinin altında `daisy`, `roses`, `sunflowers`, `dandelion`, ve `tulips` isimli 5 alt dizin olması gibi). Aksi takdirde, dizin yapısı göz ardı edilir.

`labels` argümanı ya `inferred` olarak değer alır (etiketler, dizin yapısından üretilir, örneğin, ) veya dizinde bulunan görüntü dosyalarının sayısıyla aynı boyutta tamsayı etiketlerinin bir listesi (list) veya demetidir (tuple). Etiketler, görüntü dosyası yollarının alfasayısal sırasına göre sıralanmalıdır (Python'da `os.walk(directory)` aracılığıyla elde edilir).

`class_names` argümanı `labels` argümanı `inferred` olarak ayarlandığında geçerlidir. Sınıf adlarının açık listesidir (alt dizinlerin adlarıyla eşleşmelidir). Sınıfların sırasını kontrol etmek için kullanılır (aksi takdirde alfanümerik sıra kullanılır).

`batch_size` yığın büyüklüğüdür ve varsayılan olarak 32'dir. 

`image_size` diskten okunduktan sonra görüntüleri yeniden boyutlandırmak için gerekli olan boyuttur. Varsayılan olarak `(256, 256)`'dır. İletim hattı, tümü aynı boyutta olması gereken görüntü yığınlarını işlediğinden, bu argüman fonksiyona sağlanmalıdır. 

`label_mode` dört farklı değer alabilen ve etiketlerin tipini belirten argümandır. `int`: etiketlerin tamsayılar olarak kodlandığı anlamına gelir (örn. `sparse_categorical_crossentropy` kaybı kullanıldığı zaman). `categorical`, etiketlerin kategorik bir vektör olarak kodlandığı anlamına gelir (örneğin, `categorical_crossentropy` kaybı kullanıldığı zaman). `binary`, etiketlerin (yalnızca 2 tane olabilir), 0 veya 1 değerleriyle float32 skalar olarak kodlandığı anlamına gelir (örn. `binary_crossentropy`  kullanıldığı zaman). Herhangi bir etiket yoksa, `None` olarak değer atanabilir. 

Eğer `label_mode` argümanı `int` olarak ayarlandıysa, etiketler `(yığın_büyüklüğü,)` şekline sahip int32 tipli bir tensördür.
Eğer `label_mode` argümanı `binary` olarak ayarlandıysa, etiketler `(yığın_büyüklüğü, 1)` şekline sahip 0 ve 1'lerden oluşan float32 tipli bir tensördür.
Eğer `label_mode` argümanı `categorial` olarak ayarlandıysa, etiketler, `(yığın_büyüklüğü, sınıfların_sayısı)` şekline sahip ve sınıf indeksinin bir-elemanı-bir kodlamasını temsil eden float32 tipli bir tensördür.

`color_mode` argümanı görüntülerin 1, 3 veya 4 kanala dönüştürülüp dönüştürülmeyeceğine karar verir: `grayscale` (gri ölçekli), `rgb` (3 kanallı: red, green ve blue), `rgba` (4 kanall: red, green, blue ve alpha) değerlerinden birini alabilir.  Varsayılan: `rgb`'dir.

`shuffle` argümanı verilerin karıştırılıp karıştırılmayacağına karar verir. Varsayılan: `True`. `False` olarak ayarlanırsa, verileri alfasayısal sıraya göre sıralar.

`seed` argümanı ise karıştırma ve dönüşümler için isteğe bağlı rastgele tohumdur. Rastgele sayı üretimi sırasında kullanılır. 

`interpolation` argümanı görüntüleri yeniden boyutlandırırken kullanılacak interpolasyon yöntemini seçmek için kullanılır. Varsayılan olarak `bilinear` yöntemi kullanılır. `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`, `gaussian`, ve `mitchellcubic` yöntemleri de desteklenmektedir. 

`validation_split` ve `subset` argümanlarının ne işe yaradığını aşağıdaki "Veri Kümesini Parçalamak" isimli alt bölümde göreceğiz.

O halde, elimizdeki görüntüleri `image_dataset_from_directory` kullanarak diskten yükleyelim. Alt dizinleri (sınıf) ve görüntü dosyalarının adlarını (`.jpg`) içeren dizin yapısı şu şekildedir:

```
flower_photos/
...daisy/
......5547758_eea9edfd54_n.jpg
......5673551_01d1ea993e_n.jpg
..............................
...dandelion/
......7355522_b66e5d3078_m.jpg
......8181477_8cb77d2e0f_n.jpg
..............................
...roses/
......12240303_80d87f77a3_n.jpg
......22679076_bdb4c24401_m.jpg
..............................
...sunflowers/
......6953297_8576bf4ea3.jpg
......24459548_27a783feda.jpg
.............................
...tulips/
......10791227_7168491604.jpg
......11746080_963537acdc.jpg
.............................
```

İlk olarak yükleyici için yukarıda bahsedilen bazı parametreleri tanımlayalım:

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    label_mode='int',
    seed=123,
    color_mode ='rgb',
    image_size=(180, 180),
    batch_size=32, 
    shuffle=True)
#Found 3670 files belonging to 5 classes.

train_ds
#<BatchDataset shapes: ((None, 180, 180, 3), (None,)), types: (tf.float32, tf.int32)>
```

Toplamda 3670 görüntü vardır ve `image_dataset_from_directory` fonksiyonu doğru şekilde bu görüntüleri okuyabildi. 5 tane alt dizin bulunduğu için 5 tane sınıf olduğunu çıkarabildi çünkü `labels = 'inferred'` olarak ayarlandı, diğer bir ifade ile etiketler dizin yapısından üretildi. 

Burada `label_mode='int'` olarak ayarlandığı için etiket değişkeninin boyutu  `(None,)` olmuştur. `None`, yığın büyüklüğünü temsil etmektedir. Bu `tf.data.Dataset` nesnesi her seferinde 32 tane (180 x 180) boyutlu görüntüyü size geri döndürecektir çünkü `batch_size = 32` olarak ayarlanmıştır.

```python
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# (32, 180, 180, 3)
# (32,)
```

Aşağıda görüleceği üzere tek bir yığını alıp baktığımızda, etiketlerin tamsayı olarak kodlandığı kolaylıkla görülebilir:

```python
for image, label in train_ds.take(1):
    print(label)

#tf.Tensor([3 3 4 4 1 3 4 3 4 4 1 3 0 0 2 4 1 3 1 1 1 3 1 2 1 3 4 1 4 4 4 3], shape=(32,), dtype=int32)
```

`class_names` alt dizinlerin isimlerini okuyarak sınıfların isimlerini verecektir:

```python
train_ds.class_names
#['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
```

Peki, `label_mode='categorical'` olarak değer atandığında ne olacağına bakalım:

```python
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# (32, 180, 180, 3)
# (32, 5)

for image, label in train_ds.take(1):
    print(label)

# tf.Tensor(
# [[0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0.]], shape=(32, 5), dtype=float32)
```

Kolaylıkla anlaşılacağı üzere görüntüler 3 kanallı 180 x 180 boyutundadır ve etiketler bir-elemanı-bir olan vektörler olarak kodlanmıştır (one-hot encoding). Bu nedenle her bir resmin etiketinin boyutu 5'tir, çünkü elimizde 5 sınıf vardır. 

```
[1. 0. 0. 0. 0.] -> daisy
[0. 1. 0. 0. 0.] -> dandelion
[0. 0. 1. 0. 0.] -> roses
[0. 0. 0. 1. 0.] -> sunflowers
[0. 0. 0. 0. 1.] -> tulips
```

```python
train_ds.class_names
#['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
```

## Veri Kümesini Parçalamak

Modelinizi geliştirirken bir doğrulama kümesi kullanmak iyi bir uygulamadır. `tf.keras.preprocessing.image_dataset_from_directory()` fonksiyonunu kullanarak veri kümesini eğitim (training) ve doğrulama (validation) olarak ikiye kolaylıkla parçalayabilirsiniz. Bunun için `validation_split` ve `subset` argümanlarını kullanabilirsiniz. `validation_split` 0 ile 1 arasında kayan nokta olarak değer alır ve doğrulama için ayrılacak veri oranını gösterir. `subset` argümanı ise "training" veya "validation" değerlerinden birisini almalıdır, alt kümenin ya eğitim kümesi ya da doğrulama kümesi olup olmadığını gösterir. `validation_split` argümanı ayarlanmışsa `subset` argümanı kullanılır.

Şimdi elimizdeki çiçek görüntülerinin %80'ini eğitim için ve %20'sini doğrulama için kullanalım.

```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    label_mode='categorical',
    seed=123,
    color_mode ='rgb',
    validation_split=0.2,
    subset="training",
    image_size=(180, 180),
    batch_size=32)

# Found 3670 files belonging to 5 classes.
# Using 2936 files for training.

train_ds
#<BatchDataset shapes: ((None, 180, 180, 3), (None, 5)), types: (tf.float32, tf.float32)>

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    label_mode='categorical',
    seed=123,
    color_mode ='rgb',
    validation_split=0.2,
    subset="validation",
    image_size=(180, 180),
    batch_size=32)

# Found 3670 files belonging to 5 classes.
# Using 734 files for validation.

val_ds
#<BatchDataset shapes: ((None, 180, 180, 3), (None, 5)), types: (tf.float32, tf.float32)>
```

Veri kümesini parçalamak için diğer bir yol ise henüz ana TensorFlow gövdesine entegre edilmemiş, ancak yine de kullanıcıların test etmesi ve geri bildirim vermesi için açık kaynağın bir parçası olarak mevcut olan `tf.data.experimental.cardinality` kullanmaktır. Elimizdeki doğrulama kümesinden, bu fonksiyonu kullanarak kolaylıkla test kümesi de elde edebiliriz. 

Bunu yapmak için, `tf.data.experimental.cardinality`'yi kullanarak doğrulama setinde kaç veri yığını bulunduğunu belirleyin, ardından bunların %20'sini bir test setine taşıyın.

Elimizdeki doğrulama kümesinde 734 dosya (görüntü) vardır. Yığın büyüklüğünü 32 olarak yukarı belirlemiştik. O halde `tf.data.experimental.cardinality` fonksiyonunun çıktısı 23 ($734/32 \approx 22.9375$) olacaktır. Yani doğrulama kümesinde 32 görüntüden oluşan 23 tane yığın vardır.

```python
val_batches = tf.data.experimental.cardinality(val_ds)
#<tf.Tensor: shape=(), dtype=int64, numpy=23>

test_dataset = val_ds.take(val_batches // 5)
#<TakeDataset shapes: ((None, 180, 180, 3), (None, 5)), types: (tf.float32, tf.float32)>

val_ds = val_ds.skip(val_batches // 5)
#<SkipDataset shapes: ((None, 180, 180, 3), (None, 5)), types: (tf.float32, tf.float32)>

val_ds.cardinality()
#<tf.Tensor: shape=(), dtype=int64, numpy=19>
#19 yığın vardır.

test_dataset.cardinality()
#<tf.Tensor: shape=(), dtype=int64, numpy=4>
#4 yığın vardır.
```

Görüldüğü üzere test veri kümesi olan `test_dataset` 4 yığından ve doğrulama kümesi olan `val_ds` 19 yığından oluşmaktadır. Burada dikkat edilmesi gereken nokta her yığın 32 tane görüntüden oluşmayabilir. Bu nedenle, eğer bu iki veri kümesindeki görüntü sayılarını saymak isterseniz, aşağıdaki gibi basit döngüler yazabilirsiniz:

```python
test_elem = 0
for image_batch, _ in test_dataset:
    test_elem += image_batch.shape[0]
    
print(f'Test kümesindeki görüntü sayısı: {test_elem}')
#Test kümesindeki görüntü sayısı: 128

val_elem = 0
for image_batch, labels_batch in val_ds:
    val_elem += image_batch.shape[0]
    
print(f'Test kümesindeki görüntü sayısı: {val_elem}')
#Test kümesindeki görüntü sayısı: 606
```

## Veri Kümesinden Bazı Görüntüleri Görüntülemek

Yarattığınız `tf.data.Dataset` nesnesinden bazı görüntüleri görüntülemek oldukça kolaydır. Yapmanız gereken `tf.data.Dataset`'in `take` fonksiyonunu kullanmaktır. Burada `.take(1)` alındığı için 1 adet 32 büyüklüğünde yığın seçilecektir.

```python
class_names = train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy().argmax()])
        plt.axis("off")
```

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/images_examples.png?raw=true)

## tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory kullanarak görüntüleri okumak

TensorFlow ortamına görüntüleri okumak için diğer bir yöntem ise `.flow_from_directory()`  metodunu kullanmaktır. `flow_from_directory` bir `ImageDataGenerator` metodudur. `ImageDataGenerator` görüntüler için bir üreticidir (generator) ve gerçek zamanlı veri çeşitlendirme (real-time data augmentation) yaparak görüntü verilerini yığınlar olarak oluşturur. Veri çeşitlendirme, görüntü sınıflandırma (image classification), nesne algılama (object detection) veya görüntü bölütleme (image segmentation) gibi bir çok yöntem için için küçük bir görüntü kümesinden zengin, çeşitli bir görüntü kümesi oluşturur. Bunu, kırpma, doldurma, çevirme vb. görüntü tekniklerini kullanarak gerçekleştirir ve böylelikle veri miktarını arttırılır. Veri çeşitlendirme, modeli küçük varyasyonlara kadar daha sağlam (robust) hale getirir ve dolayısıyla modelin aşırı uyum sağlamasını (overfitting) önler. Çeşitlendirilmiş görüntü verilerini bellekte depolamak ne pratik ne de verimlidir ve işte tam burada Keras'ın `ImageDataGenerator` sınıfı devreye girer. Üretici tarafından üretilen çıktı görüntüleri, girdi görüntüleriyle aynı çıktı boyutlarına sahip olacaktır. `tf.keras.preprocessing.image.ImageDataGenerator()` kullanarak tek bir satır kod ile anında görüntü çeşitlendirmeyi bir sonraki yazıda göreceğiz. Şimdilik bu sınıfın sözdizimine bakalım:

```
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=False, vertical_flip=False, rescale=None,
    preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
)
```

Kolaylıkla anlaşılacağı üzere bu fonksiyon birden çok görüntü çeşitlendirici argümana sahiptir, örneğin, `rotation_range` argümanı görüntüler üzerinde gerçekleştirilecek rastgele rotasyonlar için derece aralığının atanmasını bekler ya da `rescale=1./255` olarak belirlerseniz görüntüleri anında kolaylıkla normalleştirebilirsiniz. 

Peki, şimdilik elimizde bulunan ve çiçek görüntülerinden oluşan veri kümesine herhangi bir değişiklik yapmadan `flow_from_directory` ile nasıl okuyacağımızı görelim. İlk yapmanız gereken `ImageDataGenerator` sınıfından bir örnek (instance) oluşturmaktır. Daha sonra `flow_from_directory` methodunu çağırmanız lazımdır. Bu methodun sözdizimi aşağıdaki gibidir:

```
flow_from_directory(
    directory, target_size=(256, 256), color_mode='rgb', classes=None,
    class_mode='categorical', batch_size=32, shuffle=True, seed=None,
    save_to_dir=None, save_prefix='', save_format='png',
    follow_links=False, subset=None, interpolation='nearest'
)
```

Buradaki argümanlara göz atalım.

* `directory`: `string`, hedef dizinin yolu. Sınıf başına bir alt dizin içermelidir. Alt dizinlerin her birinde bulunan herhangi bir PNG, JPG, BMP, PPM veya TIF formatlı görüntü üreticiye dahil edilecektir.
* `target_size`: Tamsayılardan oluşan bir demettir, `(yükseklik, genişlik)`, varsayılan olarak `(256,256)`. Bulunan tüm görüntüleri yeniden boyutlandırılacaktır.
* `color_mode`: görüntülerin 1, 3 veya 4 kanala dönüştürülüp dönüştürülmeyeceğine karar verir, `grayscale` (gri ölçekli), `rgb` (3 kanallı: red, green ve blue), `rgba` (4 kanall: red, green, blue ve alpha) değerlerinden birini alabilir. Varsayılan: `rgb`'dir.
* `classes`: İsteğe bağlı sınıf alt dizinlerin bir listesidir (ör. `['dogs', 'cats']`). Varsayılan: `None`. Eğer bu liste fonksiyona sağlanmazsa, sınıfların listesi, her bir alt dizinin farklı bir sınıf olarak ele alınacağı ana dizin altındaki alt dizin adlarından/yapısından otomatik olarak çıkarılacaktır (ve etiket indeksleriyle eşleşecek sınıfların sırası alfanümerik olacaktır). Sınıf isimlerinden sınıf indekslerine eşlemeyi içeren sözlük (dictionary), `class_indices` niteliğiyle aracılığıyla elde edilebilir.
* `class_mode`: `categorical`, `binary`, `sparse`, `input`, ve `None` seçeneklerinden birini alır. Varsayılan: `categorical`. `tf.keras.preprocessing.image_dataset_from_directory` fonksiyonundaki `label_mode` argümannda olduğu gibi döndürülen etiketlerin tipini belirten argümandır. Bu argümana `categorical` değeri atandıysa, etiketler 2 boyutlu bir-elemanı-bir olarak kodlanmış olacaktır. `binary` değeri atandıysa, etiketler 1 boyutlu ikili bir vektör olacaktır (sadece iki sınıftan oluşan veri kümeleri için kullanılır). `sparse` değeri atandıysa, etiketler, 1 boyutlu tamsayı olacaktır. `input` değeri atandıysa, etiketler, girdi görüntülerinin aynısı olacaktır (çoğunlukla otokodlayıcılar (autoencoders) ile çalışmak için kullanılır). Son olarak, eğer `None` değeri atandıysa, hiç bir etiket döndürülmeyecektir (üretici sadece görüntü verilerinin yığınlarını verecektir, ki bu `model.predict()` fonksiyonu kullanılırken çok kullanışlıdır).
* `batch_size`: Veri yığınlarının büyüklüğüdür (varsayılan: 32).
* `shuffle`: Verilerin karıştırılıp karıştırılmayacağına karar verir (varsayılan: `True`). `False` olarak ayarlanırsa, verileri alfasayısal sırada sıralar.
* `seed`: Karıştırma ve dönüşümler için isteğe bağlı rastgele tohum.
* `save_to_dir`: `None` veya str (varsayılan: `None`). Bu, isteğe bağlı olarak, oluşturulan çeşitlendirilmiş görüntüleri kaydetmek için bir dizin belirtmenize olanak tanır (çoğunlukla, ne yaptığınızı görselleştirmek için kullanışlıdır).
* `save_prefix`: Str. Kaydedilen resimlerin dosya adları için kullanılacak önektir (yalnızca `save_to_dir` argümanı ayarlanmışsa geçerlidir).
* `save_format`: Kaydedilecek çeşitlendirilmiş görüntülerin formatını tanımlar ve `png`, `jpeg` eğerlerinden birini alır (yalnızca `save_to_dir` argümanı ayarlıysa geçerlidir). Varsayılan: `png`.
* `follow_links`: Sınıf alt dizinleri içindeki sembolik bağların takip edilip edilmeyeceğine karar vermek için kullanılır (varsayılan: False).
* `subset`: `ImageDataGenerator` snıfında `validation_split` argümanı ayarlanmışsa, verilerin alt kümesinden eğitim ve doğrulama kümeleri oluşturmak için kullanılır. (alacağı değerler `training` veya `validation`).
* `interpolation`: Yüklenen görüntünün boyutu, `target_size` argümanıyla tanımlanan hedef boyutundan farklı ise, görüntüyü yeniden örneklemek için kullanılacak interpolasyon yöntemidir. Desteklenen yöntemler `nearest`, `bilinear`, ve `bicubic`'dir. PIL kütüphanesinin 1.1.3 veya daha yeni bir sürümü yüklüyse, `lanczos` da desteklenir. PIL sürüm 3.4.0 veya daha yenisi yüklüyse, `box` ve `hamming` de desteklenir. Varsayılan olarak `nearest` kullanılır.

Bu methodun çıktısı `(x, y)` demetlerini veren bir `DirectoryIterator`'dır. Burada, `x` `(yığın_büyüklüğü, *hedef_büyüklüğü, kanallar)` şekline sahip bir görüntüler yığını olan NumPy dizisidir ve `y` bu görüntülere karşılık gelen etiketlerin NumPy dizisidir.

```python
datagen = tf.keras.preprocessing.image.ImageDataGenerator()
#<tensorflow.python.keras.preprocessing.image.ImageDataGenerator at 0x7f9c6603a6d0>

aug_datagen = datagen.flow_from_directory(directory = flowers, 
                                          target_size=(180, 180), 
                                          color_mode='rgb', 
                                          classes=None,
                                          class_mode='categorical', 
                                          batch_size=32, 
                                          shuffle=True, 
                                          seed=123, 
                                          interpolation='nearest')

#Found 3670 images belonging to 5 classes.

aug_datagen 
#<tensorflow.python.keras.preprocessing.image.DirectoryIterator at 0x7f9c660c0580>

aug_datagen.class_indices
#{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

# Kaç tane görüntü olduğunu bulmak için
aug_datagen.samples
#3670

images, labels = next(aug_datagen)
print(images.dtype, images.shape)
#float32 (32, 180, 180, 3)

print(labels.dtype, labels.shape)
#float32 (32, 5)
```

Görültüğü üzere, TensorFlow, elimizdeki dizinden 32 tane resim seçecektir çünkü `batch_size = 32` olarak ayarlanmıştır ve her bir görüntünün boyutu `(180, 180, 3)`'dir. `class_mode='categorical'` olarak ayarlandığı için etiketler kategorik vektörler olarak yani bir-elemanı-bir olan vektörler olarak kodlanmıştır. Bu nedenle her bir resmin etiketinin boyutu 5'tir, çünkü elimizde 5 sınıf vardır.

```
[1. 0. 0. 0. 0.] -> daisy
[0. 1. 0. 0. 0.] -> dandelion
[0. 0. 1. 0. 0.] -> roses
[0. 0. 0. 1. 0.] -> sunflowers
[0. 0. 0. 0. 1.] -> tulips
```

`ImageDataGenerator`’ın genel veri yükleme performansı, modelinizin ne kadar hızlı eğitildiğini önemli ölçüde etkileyebilir. Gereksiz para harcamadan donanım kullanımını en üst düzeye çıkarmanız gereken durumların üstesinden gelmek için, TensorFlow’un veri modülü olan `tf.data.Dataset` gerçekten yardımcı olabilir. Bir önceki yöntem olan `tf.keras.preprocessing.image_dataset_from_directory`'nin çıktısı bir `tf.data.Dataset` nesnesiydi. `tf.data.Dataset` kullanılması, belleğe sığmayacak kadar büyük olan veri kümelerini modellere beslemenin bir yoludur. Her şeyi bir kerede yüklemek ve üzerinde yineleme yapmak yerine verileri yığın yığın olarak diskten yükleyebilirsiniz. Ancak, `tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory` bir veri yineleyici (iterator) oluşturmaktadır. Bu yineleyiciyi ise `tf.data.Dataset` nesnesine çevirmek oldukça kolaylıkdır. Tek yapmanız gereken `aug_datagen` nesnesini `tf.data.Dataset.from_generator()`'a lambda fonksiyonuyla beslemektir.

```python
ds = tf.data.Dataset.from_generator(
    lambda: aug_datagen, 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([None,180,180,3], [None,5])
)

ds.element_spec
# (TensorSpec(shape=(None, 180, 180, 3), dtype=tf.float32, name=None),
#  TensorSpec(shape=(None, 5), dtype=tf.float32, name=None))

for images, label in ds.take(1):
    print('images.shape: ', images.shape)
    print('labels.shape: ', labels.shape)
    
# images.shape:  (32, 180, 180, 3)
# labels.shape:  (32, 5)
```

## Veri Kümesini Parçalamak

Genel olarak, yalnızca eğitim kümesinde bulunan örneklere veri çeşitlendirme uygulanır. Bu nedenle, **eğer görüntüler sadece tek bir dizin içerisindeyse**, eğitim ve doğrulama veri kümeleri için farklı `ImageDataGenerator` yaratabilirsiniz. Burada dikkat edilmesi gereken her iki üretici için aynı `seed` argümanı kullanmaktır.

Aşağıdaki kod parçacağı, eğitim kümesindeki görüntülere çeşitli dönüşümler uygulayan ve doğrulama kümesindeki görüntüleri yalnızca 255 ile yeniden ölçeklendiren bir dönüşüm gerçekleştirir. Ardından, her iki veri kümesi için `.flow_from_directory` yöntemi kullanılır.

```python
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)  # val 20%

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)


train_data = train_datagen.flow_from_directory(directory = flowers, 
                                               target_size=(180, 180), 
                                               color_mode='rgb',
                                               batch_size=32, 
                                               class_mode='categorical',
                                               shuffle=True,
                                               subset = 'training',
                                               seed=123) 

val_data = val_datagen.flow_from_directory(directory = flowers, 
                                           target_size=(180, 180), 
                                           color_mode='rgb',
                                           batch_size=32, 
                                           class_mode='categorical',
                                           shuffle=False,
                                           subset = 'validation',
                                           seed=123)

# Found 2939 images belonging to 5 classes.
# Found 731 images belonging to 5 classes.
```

Görüldüğü üzere 3670 çiçek görüntüsünün 2939 tanesi (yaklaşık %80'i) eğitim kümesi için, 731 tanesi (yaklaşık %20'si) ise doğrulama kümesi için ayrılmıştır. Bu iki veri kümesi için iki farklı üretici kullanmıştır. 

Ancak hem eğitim kümesindeki hem de doğrulama kümesindeki görüntülere aynı çeşitlendirme parametrelerini uygulamak isterseniz (veya hiç bir şekilde dönüşüm uygulamak istemezseniz) yapmanız gereken sadece tek bir `ImageDataGenerator` yaratmaktır. Bu üretecin argümanlarından biri olan `validation_split` argümanına, ana veri kümenizdeki görüntülerin doğrulama kümesine ayırmak istediğiniz oranını (0 ile 1 arasında) değer olarak atama yapmalı ve daha sonra, `flow_from_directory` fonksiyonunda `subset` argümanını `training` ve `validation` olarak ayarlamalısınız:

```python
generator = ImagaDataGenerator(..., validation_split=0.3)

train_gen = generator.flow_from_directory(dir_path, ..., subset='training')
val_gen = generator.flow_from_directory(dir_path, ..., subset='validation')
```

**Eğer eğitim ve doğrulama kümelerine ait görüntüler farklı dizinler içerisindeyse**, yapılacak iş daha kolaydır. Yapmanız gereken her iki dizin için farklı `ImageDataGenerator` oluşturmak ve görüntüleri `flow_from_directory` ile okumaktır. Böylelikle eğitim örneklerine uygulayacağınız çeşitlendirmeleri doğrulama kümesinde bulunan örneklere uygulamak zorunda kalmazsınız, örneğin:

```python
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')
```

Burada `train_data_dir` ve `val_data_dir`, eğitim ve doğrulama kümelerinde ait dizinlerin yollarıdır.

# `image_dataset_from_directory` ve `flow_from_directory` arasındaki fark

`image_dataset_from_directory` ve `flow_from_directory` arasındaki fark ise şu şekildedir: `image_dataset_from_directory` bir dizindeki görüntü dosyalarından bir `tf.data.Dataset` oluşturur. `ImageDataGenerator().flow_from_directory` ise bir dizine giden yolu alır ve çeşitlendirilmiş veri (augmented data) yığınları oluşturur.

Döndürdükleri obje tipi farklı olsa da, temel fark, `flow_from_directory` bir `ImageDataGenerator` yöntemidir, `image_dataset_from_directory` ise görüntülerin olduğu dizini okumak için kullanılan bir ön işleme fonksiyonudur. Ancak `image_dataset_from_directory`, anında (on-the-fly) çeşitlendirilmiş görüntü oluşturma özelliği ile size kolaylık sağlamayacaktır. O halde, hangisini kullanmalısınız? CNN ile çalışırken çeşitlendirilmiş görüntüler oluşturmak oldukça yaygındır, bu nedenle `flow_from_directory` kullanmak daha iyidir. Çeşitlendirilmiş görüntüye ihtiyacınız yoksa, `ImageDataGenerator` parametreleriyle aynı şeyi kontrol edebilirsiniz.
