---
layout: post
title: "Önceden Eğitilmiş Visual Transformer (ViT) modeline İnce-Ayar Çekmek"
author: "MMA"
comments: true
---

Sıfırdan eğitim (training from scratch), bir modelin tamamen yeni bir görev baştan sona eğitilmesini içerir. Bu genellikle büyük veri setleri ve yüksek hesaplama gücü (computation power) gerektirir. Ayrıca, eğitim süreci genellikle günler veya haftalar sürebilir.Bu yöntem genellikle özel bir görev veya dil modeli oluşturmak isteyen araştırmacılar ve büyük şirketler tarafından kullanılır.

Ancak, bu işi hobi olarak yapan biri veya bir öğrenci için bir modeli sıfırdan oluşturmak o kadar kolay değildir. Büyük veri ve yüksek hesaplama gücünün yanında, aynı zamanda oluşturulacak modelin mimarisini (architecture) belirleme ve bu modele ait hiperparametreleri ayarlama (hyperparameter tuning) süreci de zorludur. 

Bu sebeple, transfer öğrenme (transfer learning) adı verilen bir konsept literatürde yerini almıştır. 

Öğrenimleri bir problemden yeni ve farklı bir probleme uyarlamak Transfer Öğrenme fikrini temsil eder. Şöyle düşünürsek insanın öğrenmesi büyük ölçüde bu öğrenme yaklaşımına dayanmaktadır. Transfer öğrenimi sayesinde Java öğrenmek size oldukça kolay gelebilir çünkü öğrenme sürecine girildiğinde zaten programlama kavramlarını ve Python sözdizimini anlıyorsunuzdur.

Aynı mantık derin öğrenme (deep learning) için de geçerlidir. Transfer öğrenme, genellikle önceden-eğitilmiş (pre-trained) bir modelin (örneğin, Hugging Face tarafından sağlanan bir dil modeli) özel bir görev veya veri kümesine uyarlanmasıdır. Diğer bir deyişle, önceden eğitilmiş bir modelin ağırlıkları yeni veriler üzerinde eğitilir. Böylelikle, önceden eğitilmiş model yeni bir görev için hazır hale gelir. 

Önceden eğitilmiş bir model kullanmanın önemli faydaları vardır. Hesaplama maliyetlerini ve karbon ayak izinizi azaltır ve sıfırdan eğitim almanıza gerek kalmadan son teknoloji ürünü modelleri kullanmanıza olanak tanır

🤗 Hugging Face Transformers, çok çeşitli görevler için (örneğin, doğal dil işleme ve bilgisayarlı görü) önceden eğitilmiş binlerce modele erişim sağlar (https://huggingface.co/models). Önceden eğitilmiş bir model kullandığınızda, onu görevinize özel bir veri kümesi üzerinde eğitirsiniz. Bu, inanılmaz derecede güçlü bir eğitim tekniği olan ince-ayar (fine-tuning) olarak bilinir.

Transformer-tabanlı modeller genellikle görevden bağımsız gövde (task-independent body) ve göreve özel kafa (task-specific head) olarak ikiye ayrılır. Genellikle görevden bağımsız kısım, Hugging Face tarafından sağlanan ağırlıklara (weights) sahiptir. Bu kısımdaki ağırlıklar dondurulmuştur ve herhangi bir güncellemeye (updates) sahip olmazlar. Göreve özel kafa'da, elinizdeki görev için ihtiyacınız kadar nöron oluşturulur ve sadece bu katmanda eğitim özel veri kümeniz kullanılarak gerçekleştirilir.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/fine_tuning_example.png?raw=true)

Ancak, ince ayar, sinir ağının tamamında veya yalnızca katmanlarının bir alt kümesinde yapılabilir; bu durumda, ince ayarı yapılmayan katmanlar "dondurulur (frozen)" (geri yayılım (backpropagation) adımı sırasında güncellenmez).

İşte, bu tutorial'da özel bir veri kümesi (a custom dataset) için önceden eğitilmiş bir modele ince ayar yapacaksınız.

İlk olarak bu tutorial'da kullanaca

# Google Colab'e Giriş

Burada gerçekleştireceğiniz analizleri GPU'ya sahip bir makinede yapmanızda fayda var. Çünkü kullanılacak ViT modeli ve veri kümesi oldukça büyük. Bu nedenle modele ince-ayar çekmek oldukça zaman alabilir. 

İlk olarak bu tutorial'da kullanacağınız tüm Python kütüphanelerini aşağıdaki şekilde içe aktaralım:

```python
from google.colab import drive
import os
import copy
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import torch
from datasets import load_dataset, Image, load_from_disk
from transformers import (ViTFeatureExtractor,
                          ViTForImageClassification,
                          TrainingArguments,
                          TrainerCallback,
                          Trainer,
                          AutoModelForImageClassification,
                          AutoFeatureExtractor)
import evaluate
from huggingface_hub import notebook_login, create_repo
```

Tabii ki, bu kütüphaneler kişisel bilgisayarınızda veya Colab ortamınızda yoksa, öncelikle bunları yüklemeniz (install) etmeniz gerekmektedir:

```python
!pip3 install datasets
!pip install transformers
!pip install transformers[torch]
!pip install accelerate -U
!pip install evaluate
!pip install scikit-learn
```

Gerekli kütüphaneler yüklendikten ve bu kütüphaneler python ortamına içeri aktarıldıktan sonra, yapmanız gereken işlem Depolama (Storage) alanını ayarlamaktır.

Google Colab'in bir faydası, Google Drive'ınıza bağlanmanıza olanak sağlamasıdr. Böylelikle, elinizdeki veri Drive'da barınırken, kodlarınızı GPU destekli bir Jupyter Not Defterinde çalıştırabilirsiniz.

Öncelikle Google Drive'ı Colab'a bağlayalım. Google Drive'ınızın tamamını Colab'a bağlamak için `google.colab` kütüphanesindeki `drive` modülünü kullanabilirsiniz:

```python
drive.mount('/content/gdrive')
```

Google Hesabınıza erişim izni verdikten sonra Drive'a bağlanabilirsiniz.

Drive bağlandıktan sonra `"Mounted at /content/gdrive"` mesajını alırsınız ve dosya gezgini bölmesinden Drive'ınızın içeriğine göz atabilirsiniz.

Şimdi, Google Colab'in çalışma dizinini (working directory) kontrol edelim:

```python
!pwd
# /content
```

Daha sonra Python'un yerleşik (built-in) kütüphanelerinden olan `os` kütüphanesini kullanarak `project` isimli klasörü Drive'da yaratalım.

```python
path = "./gdrive/MyDrive/project"
os.mkdir(path)
```

Artık `project` klasörüne yarattığımıza göre, geçerli çalışma dizinini (current working directory) bu klasör olarak değiştirelim:

```python
os.chdir('./gdrive/MyDrive/project')
```

Artık gerçekleştireceğimiz tüm işlemler bu dizin altında yapılacak, kaydedilecek tüm dosyalar bu dizin altında kaydedilecektir.

# Veri Kümesi

Bilgisayar kullanımıyla hastalıkların otomatik tespiti önemli ancak henüz keşfedilmemiş bir araştırma alanıdır. Bu tür yenilikler tüm dünyada tıbbi uygulamaları iyileştirebilir ve sağlık bakım sistemlerini iyileştirebilir. Bununla birlikte, tıbbi görüntüleri içeren veri kümeleri neredeyse hiç mevcut değildir, bu da yaklaşımların tekrarlanabilirliğini ve karşılaştırılmasını neredeyse imkansız hale getirmektedir.

Bu nedenle burada gastrointestinal (GI) sistemin içerisinden görüntüler içeren bir veri kümesi olan Kvasir'in ikinci versiyonunu (`kvasir-dataset-v2`) kullanıyoruz. Bu veri

Kvasir veri kümesi yaklaşık 2.3GB büyüklüğündedir ve yalnızca araştırma ve eğitim amaçlı kullanılmak suretiyle ücretsizdir: https://datasets.simula.no/kvasir/

Veri kümesi, her biri 1.000 görüntüye sahip olan 8 sınıftan, yani toplam 8.000 görüntüden oluşmaktadır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/kvasir_v2_examples.png?raw=true)

Bu sınıflar patolojik bulgular (özofajit, polipler, ülseratif kolit), anatomik işaretler (z-çizgisi, pilor, çekum) ve normal ve düzenli bulgular (normal kolon mukozası, dışkı) ve polip çıkarma vakalarından (boyalı ve kaldırılmış polipler, boyalı rezeksiyon kenarları) oluşmaktadır

JPEG görüntüleri ait oldukları sınıfa göre adlandırılan ayrı klasörlerde saklanmaktadır.

Veri seti, $720 \times 576$'dan $1920 \times 1072$ piksele kadar farklı çözünürlükteki görüntülerden oluşur ve içeriğe göre adlandırılmış ayrı klasörlerde sıralanacak şekilde düzenlenmiştir.

Şimdi yukarıdaki websayfasında bulunan ve görüntüleri içeren `kvasir-dataset-v2.zip` isimli zip dosyasını indirelim:

```python
!wget https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip

# --2023-10-04 08:44:25--  https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip
# Resolving datasets.simula.no (datasets.simula.no)... 128.39.36.14
# Connecting to datasets.simula.no (datasets.simula.no)|128.39.36.14|:443... connected.
# HTTP request sent, awaiting response... 200 OK
# Length: 2489312085 (2.3G) [application/zip]
# Saving to: ‘kvasir-dataset-v2.zip’
# 
# kvasir-dataset-v2.z 100%[===================>]   2.32G  14.4MB/s    in 2m 34s  
# 
# 2023-10-04 08:47:00 (15.4 MB/s) - ‘kvasir-dataset-v2.zip’ saved [2489312085/2489312085]
```

İndirme tamamlandıktan sonra, bu zip dosyasını unzip'leyelim ve görüntü verilerimizi elde edelim:

```python
!unzip kvasir-dataset-v2.zip
```

Bu işlemden sonra `project` klasörünün altında `kvasir-dataset-v2` isimli yeni bir klasör oluşacaktır.

zip dosyasıyla işimiz bittiği için yer kaplamaması için silelim:

```python
!rm -rf kvasir-dataset-v2.zip
```

Daha sonra, `os` kütüphanesini kullanarak, kolaylık olması açısından `kvasir-dataset-v2` isimli dosyayı `image_data` olarak yeniden isimlendirelim:

```python
os.rename('kvasir-dataset-v2', 'image_data')
```

Son durumda görüntülerden oluşan veri kümesi yapısı şu şekilde görünecektir:

```
image_data/dyed-lifted-polyps/0a7bdce4-ac0d-44ef-93ee-92dfc8fe0b81.jpg
image_data/dyed-lifted-polyps/0a7ece5b-caaa-496e-9e6e-0c7eab171527.jpg
...
image_data/dyed-resection-margins/0a0b455d-d3dd-4be4-a6a3-90f81d8c8f36.jpg
image_data/dyed-resection-margins/0a2a2f35-c798-447c-a883-8f2f448bfe07.jpg
...
image_data/ulcerative-colitis/00a436bc-67ee-4a43-b1a7-25130a2d4e72.jpg
image_data/ulcerative-colitis/cat/0aacb7fa-19fb-4bd6-9a43-3c0a246e7a58.jpg
```

Bu özel (custom) veri kümesini HuggingFace ortamına yüklemek için bir veri kümesinin yapısını (structure) ve içeriğini (content) değiştirmek için birçok araç sağlayan Hugging Face'in `datasets`  modülündeki `load_dataset`  fonksiyonunu kullanabilirsiniz. Bu fonksiyon ya `Dataset` ya da `DatasetDict` döndürecektir:

```python
full_dataset = load_dataset("imagefolder", data_dir="./image_data", split="train")
full_dataset
# Dataset({
#     features: ['image', 'label'],
#     num_rows: 8000
# })
```

Veri kümesinde 8000 satır olduğunu (çünkü 8000 görüntü var) ve görüntülerin etiketlerinin (`label`) otomatik oluşturulduğunu kolaylıkla görebilirsiniz.

```python
full_dataset.features
# {'image': Image(decode=True, id=None),
#  'label': ClassLabel(names=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'], id=None)}
```

Bu bir sözlüktür (dictionary). `label` anahtarını kullanarak kolaylıkla etiketleri (labels) çekebilirsiniz:

```python
labels = full_dataset.features['label']
labels
# ClassLabel(names=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'], id=None)
```

Spesifik bir görüntüye ulaşmak da oldukça kolaydır. Yapmanız gereken bir indeks kullanmaktır. Şimdi veri kümesindeki ilk görüntüye ulaşalım:

```python
full_dataset[0]
# {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=720x576>,
# 'label': 0}
```

Görüldüğü üzere, etiketlere otomatik olarak tamsayı atanmıştır. Burada `0` tamsayısına sahip etiketin ismi `dyed-lifted-polyps`'dır.

Bu görüntününün moduna ve etiketine de kolaylıkla erişilebilir:

```python
full_dataset[0]['image'].mode
# 'RGB
```

```python
full_dataset[0]['label'], labels.names[full_dataset[0]['label']]
# (0, 'dyed-lifted-polyps')
```

Şimdi etiketler ve bu etiketlere ait id'leri (numaraları) içerisinde tutan iki sözlük (dictionary) oluşturalım:

```python
id2label = {str(i): label for i, label in enumerate(labels.names)}
print(id2label)
# {'0': 'dyed-lifted-polyps', '1': 'dyed-resection-margins', '2': 'esophagitis', '3': 'normal-cecum', '4': 'normal-pylorus', '5': 'normal-z-line', '6': 'polyps', '7': 'ulcerative-colitis'}

label2id = {v: k for k, v in id2label.items()}
print(label2id)
# {'dyed-lifted-polyps': '0', 'dyed-resection-margins': '1', 'esophagitis': '2', 'normal-cecum': '3', 'normal-pylorus': '4', 'normal-z-line': '5', 'polyps': '6', 'ulcerative-colitis': '7'}
```

Bu iki sözlüğü daha sonra kullanacağız.

`datasets` modülünde bulunan `Image` fonksiyonu, bir görüntü nesnesini döndürmek için `image` sütunundaki verilerin kodunu otomatik olarak çözer. Şimdi görselin ne olduğunu görmek için `image` sütununu çağırmayı deneyiniz:

```python
full_dataset[0]["image"]
```

![](https://raw.githubusercontent.com/mmuratarat/turkish/049aed9bc3e11c206d7c3b90f644162d288174e5/_posts/images/ksavir_vit_image0.png)

```python
full_dataset[1000]["image"]
```

![](https://raw.githubusercontent.com/mmuratarat/turkish/049aed9bc3e11c206d7c3b90f644162d288174e5/_posts/images/ksavir_vit_image1000.png)


# Model

Burada görüntü sınıflandırma (image classification) gerçekleştireceğiz. Veri kümesinde 8 farklı sınıf var. Bu nedenle çok-sınıflı sınıflandırma (multi-class classification) problemi ile karşı karşıyayız. 

Çok-sınıflı görüntü sınıflandırma problemi için önceden-eğitilmiş (pre-trained) bir modele ihtiyacımız var. HuggingFace Hub'da görüntü sınıflandırma için oldukça fazla önceden-eğitilmiş model bulabilirsiniz - https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads&library=pytorch

Görüntü sınıflandırma için transfer öğrenmenin arkasındaki sezgi, eğer bir model yeterince geniş ve genel bir veri kümesi üzerinde eğitilirse, bu model etkili bir şekilde görsel dünyanın genel bir modeli olarak hizmet edebilir. Daha sonra büyük bir modeli büyük bir veri kümesi üzerinde eğiterek sıfırdan başlamanıza gerek kalmadan bu öğrenilen özellik haritalarından (feature maps) yararlanabilirsiniz.

Birçok görevde bu yaklaşım, hedeflenen verileri kullanarak bir modeli sıfırdan eğitmekten daha iyi sonuçlar vermiştir.

Artık ihtiyaçlarımıza uyacak şekilde ince ayar yapacağımız temel modelimizi seçebiliriz.

Burada Visual Transformer (ViT) denilen bir görsel transformer modelini kullanacağız (https://huggingface.co/google/vit-base-patch16-224). 

ViT, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" isimli makalede 2021 yılında Dosovitskiy ve arkadaşları tarafından tanıtılmıştır. ViT modeli ImageNet-21k (14 milyon görüntü, 21.843 sınıf) veri kümesi üzerinde önceden eğitilmiş ve ImageNet 2012 (1 milyon görüntü, 1.000 sınıf) veri kümesi üzerinde ince ayar çekilmiştir. Girdi görüntüleri  $224 x 224$ çözünürlüğe sahiptir. Genel olarak üç tür ViT modeli vardır:

* ViT-base: 12 katmana, 768 gizli boyuta ve toplam 86M parametreye sahiptir.
* ViT-large: 24 katmana, 1024 gizli boyuta ve toplam 307M parametreye sahiptir.
* ViT-huge: 32 katmanı, 1280 gizli boyutu ve toplam 632M parametresi vardır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/Screenshot%202023-10-04%20at%205.17.43%20PM.png?raw=true)

Bu tutorial için Hugging Face'te bulunan [the google/vit-base-patch16-224-in21k model](https://huggingface.co/google/vit-base-patch16-224-in21k) modelini kullanacağız.

İlk olarak biraz veri temizliği yapalım ve RGB olmayan (tek kanallı, gri tonlamalı) görüntüleri veri kümesinden kaldıralım:

```python
# Remove from dataset images which are non-RGB (single-channel, grayscale)
condition = lambda data: data['image'].mode == 'RGB'
full_dataset = full_dataset.filter(condition)
full_dataset
# Dataset({
#     features: ['image', 'label'],
#     num_rows: 8000
# })
```

Görüntülerin hepsi üç kanallı, yani RGB modundadır. Bu nedenle herhangi bir filtreleme işlemi gerçekleşmemiştir.

# Veri kümesini eğitim/test olarak parçalamak

Modeli seçtiğimizde göre ilk olarak yapmamız gereken veri kümesini eğitim (train) ve test olacak şekilde ikiye parçalamak. Bunun için `train_test_split()` fonksiyonunu kullanabilir ve parçalanmanın (splitting) boyutunu belirlemek için `test_size` parametresini belirtebilirsiniz. Burada test kümesinin büyüklüğünü belirlemek için %15 kullandık, yani, 6800 görüntü modeli eğitmek için, geri kalan 1200 görüntü modeli test etmek için kullanılacaktır.

```python
dataset = full_dataset.shuffle().train_test_split(test_size=0.15, stratify_by_column = 'label')
dataset
# DatasetDict({
#     train: Dataset({
#         features: ['image', 'label'],
#         num_rows: 6800
#     })
#     test: Dataset({
#         features: ['image', 'label'],
#         num_rows: 1200
#     })
# })
```

Kolaylıkla anlaşılacağı üzere bir DatasetDict nesnesine sahibiz. Bu bir sözlüktür. Anahtarları `train` ve `test`'tir. Bu anahtarlardaki değerleri (yani veri kümelerini) ayrı ayrı değişkenlere atayalım:

```python
train_dataset = dataset["train"]
train_dataset
# Dataset({
#     features: ['image', 'label'],
#     num_rows: 6800
# })
```

```python
test_dataset = dataset["test"]
test_dataset
# Dataset({
#     features: ['image', 'label'],
#     num_rows: 1200
# })
```

# Öznitelik Çıkarıcı

Vision Transformer modeli temel olarak iki önemli bileşenden oluşur: bir sınıflandırıcı (classifier) ve bir öznitelik çıkarıcı (feature extractor).

ViT modelini kullanarak sınıflandırma gerçekleştirmeden önce Öznitelik Çıkarıcı (feature extractor) adı verilen bir işlem gerçekleştirmeliyiz. 

Öznitelik çıkarsama, ses veya görüntü modelleri için girdi özniteliklerinin (input features) hazırlanmasından sorumludur. Bu öznitelik çıkarsama adımını, Doğal Dil İşleme (Natural Language Processing) görevlerindeki Token'laştırma (Tokenizing) adımı olarak düşünebilirsiniz.

Öznitelik çıkarsama, elimizdeki görüntüleri normalleştirmek (normalizing), yeniden boyutlandırmak (resizing) ve yeniden ölçeklendirmek (rescaling) üzere, görüntülerin "piksel değerlerinin" tensörlerine ön-işleme gerçekleştirmek için kullanılır. 

ViT modeline ait Feature Extractor'ı Hugging Face Transformers kütüphanesinden şu şekilde başlatıyoruz:

```python
# modeli içe aktar
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
feature_extractor
# ViTFeatureExtractor {
#   "do_normalize": true,
#   "do_rescale": true,
#   "do_resize": true,
#   "image_mean": [
#     0.5,
#     0.5,
#     0.5
#   ],
#   "image_processor_type": "ViTFeatureExtractor",
#   "image_std": [
#     0.5,
#     0.5,
#     0.5
#   ],
#   "resample": 2,
#   "rescale_factor": 0.00392156862745098,
#   "size": {
#     "height": 224,
#     "width": 224
#   }
# }
```

Öznitelik çıkarıcıya ait yapılandırma (configuration), normalleştirme, ölçekleme ve yeniden boyutlandırmanın `true` olarak ayarlandığını göstermektedir.

Sırasıyla `image_mean` ve `image_std`de saklanan ortalama ve standart sapma değerleri kullanılarak üç renk kanalında (RGB) normalleştirme gerçekleştirilir.

Çıktı boyutu `size` anahtarı ile $224 \times 224$ piksel olarak ayarlanır.

Bir görüntüyü öznitelik çıkarıcıyla işlemeyi tek bir görüntü üzerinde şu şekilde gerçekleştiririz:

```python
example = feature_extractor(train_dataset[0]['image'], return_tensors='pt')
example
# {'pixel_values': tensor([[[[-0.9608, -0.9608, -0.9608,  ..., -0.9608, -0.9608, -0.9686],
#          [-0.9686, -0.9608, -0.9608,  ..., -0.9608, -0.9686, -0.9686],
#          [-0.9686, -0.9608, -0.9686,  ..., -0.9686, -0.9686, -0.9765],
#          ...,
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9843, -0.9843, -0.9843],
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9843, -0.9843, -0.9843],
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9843, -0.9843, -0.9843]],
#
#         [[-0.9608, -0.9608, -0.9529,  ..., -0.9608, -0.9608, -0.9686],
#          [-0.9686, -0.9608, -0.9608,  ..., -0.9608, -0.9686, -0.9686],
#          [-0.9686, -0.9608, -0.9686,  ..., -0.9686, -0.9686, -0.9686],
#          ...,
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9843, -0.9843, -0.9843],
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9843, -0.9843, -0.9843],
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9843, -0.9843, -0.9843]],
#
#         [[-0.9686, -0.9608, -0.9608,  ..., -0.9686, -0.9608, -0.9765],
#          [-0.9686, -0.9686, -0.9686,  ..., -0.9686, -0.9686, -0.9686],
#          [-0.9686, -0.9765, -0.9765,  ..., -0.9686, -0.9686, -0.9686],
#          ...,
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9922, -0.9922, -0.9843],
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9922, -0.9843, -0.9843],
#          [-0.9843, -0.9843, -0.9843,  ..., -0.9922, -0.9843, -0.9843]]]])}
```

Oluşacak tensörün boyutu şu şekildedir:

```python
example['pixel_values'].shape
```

Kolaylıkla anlaşılacağı üzere, ön işleme adımından sonra 4 boyutlu bir tensör elde ediliyor. Burada ilk boyut (dimension) yığın büyüklüğünü (batch size), ikinci boyut görüntülerdeki kanal sayısını (number of channels, RGB görüntüler ile çalıştığımız için üç kanal var), üçüncü ve döndüncü boyutlar, sırasıyla görüntülerin yüksekliğini (height) ve genişliğini (width) temsil etmektedir.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/image3.jpeg?raw=true)

Burada not edilmesi gereken diğer bir durum `pixel_values` anahtarının sahip olduğu değer, modelin beklediği temel girdi olmasıdır.

Bu ön işleme adımını **tüm veri kümesine** daha verimli bir şekilde uygulamak için, `preprocess` adı verilen bir fonksiyon oluşturalım ve dönüşümleri `map` yöntemini kullanarak gerçekleştirelim:


```python
def preprocess(examples):
    # take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(examples['image'], return_tensors='pt')
    # include the labels
    inputs['label'] = examples['label']
    return inputs

# apply to train-test datasets
prepared_train = train_dataset.map(preprocess, batched=True)
prepared_test = test_dataset.map(preprocess, batched=True)

prepared_train
# Dataset({
#     features: ['image', 'label', 'pixel_values'],
#     num_rows: 6800
# })

prepared_test
# Dataset({
#     features: ['image', 'label', 'pixel_values'],
#     num_rows: 1200
# })
```

Kolaylıkla görülebileceği üzere artık eğitim ve test kümelerinde artık `'pixel_values'` isimli yeni bir özniteliğe sahibiz.

**EK NOT**

Ön-işleme (pre-processing) çok zaman alabilir. Özellikle not defterinin (notebook) tamamını tekrar tekrar çalıştırmanız gerektiğinde.

Bunu önlemek için veri kümelerinizi (eğitim ve test veri kümeleri) ön-işleme tabi tuttuktan sonra diske kaydedin ve tekrar kullanmanız gerektiğinde bu ön-işlenmiş kümeleri tekrar yükleyin.

Bu nedenle ilk olarak, ön-işlemeden geçirilmiş eğitim ve test kümelerinin saklanacağı klasörleri çalışma dizinimizde (yani `./model`) yaratalım. Burada, eğitim ve test kümeleri için ayrı dosyalar oluşturuyoruz çünkü Hugging Face bu veri kümelerini shard'lara ayırdıktan sonra diske kaydetme işlemi yapmaktadır:

```python
os.makedirs('./prepared_datasets/train')
os.makedirs('./prepared_datasets/test')
prepared_train.save_to_disk("./prepared_datasets/train")
prepared_test.save_to_disk("./prepared_datasets/test")
```

Artık ihtiyacımız olduğunda kaydettiğimiz bu ön-işlenmiş veri kümelerini tekrar geri yükleyip işlerimize devam edebiliriz:

```python
prepared_train = load_from_disk("./prepared_datasets/train")
prepared_test = load_from_disk("./prepared_datasets/test")
```

# ViT'i yüklemek

Elimizdeki görüntüleri kullanacağımız modelin istediği uygun formata biçimlendirdikten sonra, bir sonraki adım ViT'yi indirip başlatmaktır (initialize).

Burada da, öznitelik çıkarıcıyı (feature extractor) yüklemek (load) için kullandığımız `from_pretrained` yöntemiyle Hugging Face Transformers'ı kullanıyoruz.
