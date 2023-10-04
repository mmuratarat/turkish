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

Bu nedenle burada gastrointestinal (GI) sistemin içerisinden görüntüler içeren bir veri kümesi olan Kvasir'in ikinci versiyonunu (`kvasir-dataset-v2`) kullanıyoruz.

Kvasir veri kümesi yaklaşık 2.3GB büyüklüğündedir ve ücretsiz bir şekilde indirebilir: https://datasets.simula.no/kvasir/

Veri kümesi, her biri 1.000 görüntüye sahip olan 8 sınıftan, yani toplam 8.000 görüntüden oluşmaktadır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/kvasir_v2_examples.png?raw=true)

Bu sınıflar patolojik bulgular (özofajit, polipler, ülseratif kolit), anatomik işaretler (z-çizgisi, pilor, çekum) ve normal ve düzenli bulgular (normal kolon mukozası, dışkı) ve polip çıkarma vakalarından (boyalı ve kaldırılmış polipler, boyalı rezeksiyon kenarları) oluşmaktadır

JPEG görüntüleri ait oldukları sınıfa göre adlandırılan ayrı klasörlerde saklanmaktadır.

Veri seti, $720 \times 576$'dan $1920 \times 1072$ piksele kadar farklı çözünürlükteki görüntülerden oluşur ve içeriğe göre adlandırılmış ayrı klasörlerde sıralanacak şekilde düzenlenmiştir.

Şimdi yukarıdaki websayfasında bulunan ve görüntüleri içeren `kvasir-dataset-v2.zip` isimli zip dosyasını indirelim:
