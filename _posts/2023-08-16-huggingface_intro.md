---
layout: post
title: "Hugging Face'e Kısa bir Giriş"
author: "MMA"
comments: true
---

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/huggingface_ecosystem.png?raw=true)

Doğal Dil İşleme (Natural Language Processing - NLP) alanında çalışıyorsanız, Hugging Face'i muhtemelen duymuşsunuzdur.

2016 yılında Fransız girişimciler Clément Delangue, Julien Chaumond ve Thomas Wolf tarafından kurulan Hugging Face (adını popüler bir emoji olan 🤗'den almıştır. Türkçesi: sarılan yüz) bir chatbot şirketi olarak işe başlamış ve daha sonra Doğal Dil İşleme teknolojilerinin açık kaynak sağlayıcısına dönüşmüştür. Birinci kullanım alanı NLP görevleri için olsa da, o zamandan beri ses sınıflandırması (audio classification) ve konuşma tanıma (speech recognition) gibi ses (audio) ile ilgili görevlerde ve görüntü sınıflandırma (image classification) ve görüntü bölümleme (image segmentation) gibi bilgisayarlı görü (computer vision) görevleri için de kullanılmaya başlanmıştır. https://huggingface.co/tasks

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/huggingface_tasks.png?raw=true)

Çoğunuzun bildiği gibi transfer öğrenme (transfer learning), önceden-eğitilmiş (pre-trained) modellerin yeni görevler için yeniden kullanılmasını mümkün kıldığından, Transformer modellerinin başarısını yönlendiren temel faktörlerden biridir. Hugging Face'i benzersiz yapan da tam olarak budur.

Hugging Face, metin sınıflandırma, duygu analizi, soru yanıtlama ve bunlar gibi çeşitli Doğal Dil İşleme görevleri için önceden-eğitilmiş dil modelleri sağlayan bir kütüphanedir. Bu modeller, derin öğrenme algoritmalarına dayalıdır ve belirli NLP görevleri için ince-ayar çekilmiştir, bu da Doğal Dil İŞlemeye direkt başlamayı başlamayı kolaylaştırır.

Kendi göreviniz için sıfırdan bir model eğitmenize bile gerek yok. Yapmanız gereken Hugging-Face ile önceden-eğitilmiş bir modeli yüklemek ve özel göreviniz için modele ince-ayar çekmektedir. Bu kadar basit.

Bu önceden-eğitilmiş modeller, TensorFlow, PyTorch ve JAX gibi farklı derin öğrenme kitaplıkları kullanılarak eğitilmiştir. Her derin öğrenme kütüphanesine aşina olmak zorunda değilsiniz. Hugging Face bu farklılıkları standartlaştırır. Ayrıca, bu modeller de çok büyük olduğundan, HugginFace bu modellerin ağırlıklarını sunucudan yükler.

Hugging Face'in bu kadar popüler olmasının bir diğer nedeni de kullanım kolaylığıdır. Basit ve kullanıcı dostu bir arayüze sahiptir. Bu, geliştiricilerin NLP'ye hızlı bir şekilde başlamalarını kolaylaştırır. Muhteşem topluluk desteğini de unutmayalım. Hugging Face topluluğu aktif ve destekleyicidir. Topluluk ayrıca önceden-eğitilmiş modelleri paylaşarak kütüphanenin büyümesine katkıda bulunur.

## Hugging Face'in Bileşenleri

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/huggingface_components.png?raw=true)

Hugging Face, çeşitli görevler için bir dizi güçlü araç sağlayan harika bir platformdur. Hugging Face ekosisteminin ana bileşenleri, Hub, Transformers, Sentence-Transformers & Diffusers kütüphaneleri, Inference API'sı ve bir web uygulaması oluşturucusu Gradio'dur.

### The Hub

Hub, makine öğrenmesiyle ilgili her şey için merkezi bir yer olarak çalışır.

PyTorch (https://pytorch.org/hub/) ve TensorFlow'un (https://www.tensorflow.org/hub) da kendilerine ait birer hub sunduğunu ve Hugging Face Hub'da belirli bir model veya veri kümesi yoksa, bu hub'ların kontrol edilmeye değer olduğunu unutmayınız.

#### Models

https://huggingface.co/docs/hub/models

Bu Hub'da, topluluk tarafından paylaşılan on binlerce açık kaynaklı makine öğrenmesi modelini keşfedebilir ve kullanabilirsiniz. Hub, yeni modelleri keşfetmenize ve herhangi bir projeye başlamanıza yardımcı olur. Hugging Face Hub, 290.000'den fazla ücretsiz modele ev sahipliği yapmaktadır. Görevler, kütüphaneler, veri kümeleri, diller ve bunlar gibi, Hub'da gezinmenize ve gelecek vaat eden model adaylarını hızlı bir şekilde bulmanıza yardımcı olmak için tasarlanmış filtreler vardır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/huggingface_models_filters.png?raw=true)

#### Datasets

https://huggingface.co/docs/hub/datasets

Veri kümelerinin yüklenmesi, işlenmesi ve depolanması, özellikle veri kümeleri dizüstü bilgisayarınızın RAM'ine sığamayacak kadar büyüdüğünde, külfetli bir süreç olabilir. Ek olarak, verileri indirmek ve standart bir biçime dönüştürmek için genellikle çeşitli betikler yazmanız gerekmektedir.

Hugging Face Datasets kütüphanesi, standart bir arayüz sağlayarak bu işlemi basitleştirir. 100'den fazla dilde 50.000 veri kümesi içerir. Bu veri kümelerini Doğal Dil İşleme, Görüntü İşleme ve Ses genelinde çok çeşitli görevler için kullanabilirsiniz. Hub, veri kümelerini bulmayı, indirmeyi ve karşıya yüklemeyi (upload) kolaylaştırır.

Ayrıca akıllı önbelleğe alma (caching) sağlar (böylece kodunuzu her çalıştırdığınızda ön işlemenizi (pre-processing) yeniden yapmak zorunda kalmazsınız) ve bir dosyayı daha verimli bir şekilde modifiye etmek üzere, bu dosyanın içeriğini sanal bellekte depolayan ve birden çok işlemi (processes) etkinleştiren bellek eşleme (memory mapping) adı verilen özel bir mekanizmadan yararlanarak RAM sınırlamalarını önler. 

Datasets kütüphanesi ayrıca Pandas ve NumPy gibi popüler yazılım çerçeveleri ile birlikte çalışabilir, bu nedenle favori veri düzenleme araçlarınızın rahatlığını bırakmanız gerekmez.

##### Modelerl ve Veri Kümeleri için Kartlar

Hub ayrıca, modellerin ve veri kümelerinin içeriklerini belgelemek ve bunların sizin için doğru olup olmadığı konusunda bilinçli bir karar vermenize yardımcı olmak için model ve veri kümesi kartları (model and dataset cards) sağlar. 

Hub'ın diğer bir güzel özelliği ise herhangi bir modeli doğrudan göreve-özgü etkileşimli widget'lar aracılığıyla deneyebilmenizdir:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/compute_modelcard_huggingface.png?raw=true)

##### Evaluate

Geçmişte, Datasets kütüphanesi altında "metrics" adı verilen bir bileşen daha vardı (web sayfasını Hugging Face ana sayfasında halen görebilirsiniz - https://huggingface.co/metrics). Ancak, metrics kütüphanesi artık kullanımdan kaldırılmıştır. Makine öğrenmesi modellerini ve veri kümelerini kolayca değerlendirmek için artık "Evaluate" adlı yeni bir kütüphane vardır.

https://huggingface.co/docs/evaluate/index

Tek bir kod satırı ile farklı alanlar için onlarca değerlendirme ölçütüne erişim sağlayabilirsiniz. İster yerel makinenizde ister dağıtılmış sisteminizde, modellerinizi tutarlı ve tekrarlanabilir bir şekilde değerlendirebilirsiniz!

Evaluate, ROC AUC'den BLEU'ya, Pearson korelasyon katsayısından CharacTer'e kadar çok çeşitli değerlendirme araçlarına erişim sağlar.

#### Spaces

https://huggingface.co/docs/hub/spaces

Spaces kütühanesi, makine öğrenmesi destekli demoları dakikalar içinde oluşturmanızı ve dağıtmanızı kolaylaştırır. Bu uygulamalar makine öğrenmesi portföyünüzü oluşturmanıza, projelerinizi sergilemenize ve diğer kişilerle işbirliği içinde çalışmanıza yardımcı olur. Spaces, Gradio, Streamlit ve Docker kütüphanelerini kullanmanızı da izin verir. Bir Space (alan) içerisinde, sadece HTML, CSS ve JavaScript sayfaları olan statik Alanlar (spaces) da oluşturabilirsiniz.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/huggingface_spaces.png?raw=true)

Her Spaces ortamı, varsayılan olarak ücretsiz olarak kullanabileceğiniz 16 GB RAM, 2 CPU çekirdeği ve 50 GB (kalıcı olmayan) disk alanı ile sınırlıdır. Rekabetçi bir fiyat karşılığında çeşitli GPU hızlandırıcıları ve kalıcı depolama dahil olmak üzere daha iyi bir donanıma yükseltme yapabilirsiniz.

### Libraries

Hub, açık-kaynaklı ekosistemde düzinelerce kütüphaneyi de destekler. En popüler üç kütüphane, Transformers, Diffusers ve Sentence-Transformers'dır.

##### The Transformers Library

https://huggingface.co/docs/transformers/index
https://github.com/huggingface/transformers

Transformers kütüphanesi, API'ler ve araçlar sağlayan açık kaynaklı bir kütüphanedir. Böylece, Transformer tabanlı son teknoloji önceden-eğitilmiş modelleri kolayca indirebilir ve eğitebilirsiniz. Transformers, PyTorch yazılım çerçevesi üzerine inşa edilmiştir, ancak, TensorFlow ve JAX ile de kullanabilirsiniz. Bu, bir modelin her aşamasında farklı bir yazılım çerçevesi kullanma esnekliği sağlar. Örneğin, bir yazılım çerçevesi kullanarak üç satırlık bir kod ile bir model eğitebilir ve çıkarsamalar yapmak için başka bir yazılım çerçevesini kullanarak bu modeli geri yükleyebilirsiniz. 

Kurulum için Python'un paket yöneticisi olan pip'i kullanabilirsiniz.

##### The Diffusers library

https://huggingface.co/docs/diffusers/index
https://github.com/huggingface/diffusers

Diffusers kütüphanesi, Hugging Face ekosistemine yeni bir eklemedir. Bilgisayarlı görü ve ses görevleri için önceden eğitilmiş Diffusion modellerini kolayca paylaşmanın, versiyonlamanın ve yeniden üretmenin bir yolunu sağlar. Bu kütüphane, diffusion modellerine ve özellikle Ağustos 2022'den beri açık kaynak olan Stable Diffusion'a odaklanmaktadır. Diffusers kitaplığı, Stable Diffusion'ı kolayca kullanmanızı sağlar.

#### The Sentence-Transformers library

https://github.com/UKPLab/sentence-transformers

Sentence-Transformers kütüphanesi, cümleler, paragraflar ve görüntüler için yoğun vektör temsillerini hesaplamak için kolay bir yöntem sağlar. Bu kütüphanede bulunan modeller, BERT / RoBERTa / XLM-RoBERTa vb. gibi Transformer ağlarına dayalıdır ve çeşitli görevlerde son teknoloji performans elde eder. Metinler için vektör uzayında gömülmeleri (embeddings) hesaplayabilir, kosinüs benzerliğini (cosine similarity) kullanarak birbirine benzeyen metinleri kolaylıkla bulabilirsiniz. Bu kütüphane, 100'den fazla dilde ince-ayar çekilmiş çeşitli kullanım durumları için, son teknoloji önceden eğitilmiş modeller sağlar. Ayrıca, spesifik görevinizde maksimum performans elde etmek üzere, gömülmeleri hesaplamak için kullanılan özel modellere kolay bir biçimde ince-ayar çekmenize olanak sağlar.

Yukarıda kısaca bahsedilen üç kütüphaneye ek olarak, Hugging Face ekosistemi tarafından desteklenen pek çok kütüphane daha vardı: https://huggingface.co/docs/hub/models-libraries

### Accelerate

https://huggingface.co/docs/accelerate/index

PyTorch'ta kendi eğitim betiğinizi yazmak zorunda kaldıysanız, dizüstü bilgisayarınızda çalışan kodu, organizasyonunuzun sunucu kümesinde çalışan koda port'lamaya çalışırken bazı sıkıntılar yaşamış olabilirsiniz.  Accelerate, eğitim altyapısı için gerekli tüm özel mantığın çaresine bakan normal eğitim döngülerinize bir soyutlama katmanı ekler. Bu, gerektiğinde altyapı değişikliğini basitleştirerek iş akışınızı tam anlamıyla hızlandırır (accelerates).

### The Inference API

Hugging Face'teki herhangi bir modeli üretim ortamına koymak (yanı dağıtmak) ve bu modelden çıkarsamalar (inferences) yapmak istediğinizi varsayalım. Bunun için Hugging Face'in Inference API'sini kullanabilirsiniz.

Inference Uç Noktaları (endpoints), tüm Transformers, Sentence-Transformers ve Diffusion görevlerinin yanı sıra Transformers kütüphanesi tarafından desteklenmeyen özel görevleri de destekler.

Bunun nasıl yapılacağına bir göz atalım. İlk adım, hangi modeli çalıştıracağınızı seçmektir. Inference API'nin Python ile nasıl kullanılacağını aşağıdaki kodda inceleyebilirsiniz:

```python
import json
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"
def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
data = query({"inputs": "The answer to the universe is [MASK]."})
``` 

Özetle, Inference API'si, karmaşık kod yazmak zorunda kalmadan modellerinizi mevcut uygulamalarınıza entegre etmenizi kolaylaştırır

### Tokenizers

https://huggingface.co/docs/tokenizers/index
https://github.com/huggingface/tokenizers

Veri iletim hattının (data pipeline) her birinin arkasında, ham metni (raw text) andaç (token) adı verilen daha küçük parçalara bölen bir andaçlaştırıcı (tokenizer) adımı vardır. Bir andaç bir sözcük olabilir, sözcüğün bir kısmı olabilir veya  noktalama işaretleri gibi yalnızca karakterler olabilir. Transformer modelleri, bu andaçların sayısal temsilleri üzerinde eğitilmiştir, bu nedenle bu adımı doğru yapmak tüm Doğal Dil İşleme projeleri için oldukça önemlidir!

Hugging Face'in Tokenizers kütüphanesi andaçlara ayırmak için bir çok strateji sağlar ve Rust arka ucu sayesinde bir metni andaçlarına ayırmada son derece hızlıdır. Ayrıca, girdilerin normalleştirilmesi ve model çıktılarının gerekli formata dönüştürülmesi gibi tüm ön ve son işleme adımlarıyla da ilgilenir. Tokenizers kütüphanesinin yardımıyla, Transformers kütüphanesi ile önceden eğitilmiş model ağırlıklarını yükleyebildiğimiz gibi bir andaçlaştırıcıyı da kolaylıkla yükleyebiliriz.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/footer_HF.png?raw=true)
