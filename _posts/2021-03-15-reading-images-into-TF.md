---
layout: post
title:  "Derin Öğrenme Modelleri için TensorFlow'a Görüntü Veri Kümelerini Yükleme"
author: "MMA"
comments: true
---

Evrişimsel sinir ağı (Convolutional Neural Network), derin öğrenme sinir ağlarının bir sınıfıdır. CNN'ler, görüntü tanımada büyük yenilikler getiriyor. Genellikle görsel görüntüleri analiz etmek üzere çeşitli bilgisayarlı görü görevleri için kullanılırlar, örneğin, görüntü sınıflandırma, nesne tanıma, görüntü bölütleme v.b. Facebook'un fotoğraf etiketlemesinden tutun da sürücüsüz arabalara kadar her şeyin merkezinde bulunabilir. Sağlık hizmetlerinden internet güvenliğine kadar her konuda perde arkasında yer alırlar. Hızlıdırlar ve verimlidirler. Ancak, bu alana yeni giren birinin ilk sorduğu soru, bir bilgisayarlı görü görevi için bir CNN modelini nasıl eğitiriz?

Evrişimsel bir sinir ağını eğitirken görüntü verilerinin en iyi şekilde nasıl hazırlanacağını bilmek zordur. Bu, modelin hem eğitimi hem de değerlendirilmesi sırasında hem piksel değerlerinin ölçeklendirilmesini hem de görüntü verisi çeşitlendirme (data augmentation) tekniklerinin kullanılmasını içerir.

Modelinizi eğitmek için bir görüntüyü bir CNN'e nasıl besleyeceğinizi öğrenmek istiyorsunuz. Bunu yapmak için, eğitim setinizdeki görüntüleri vektörleştirilmiş bir forma dönüştürmeniz (bir dizi veya matrise çevirmeniz) gerekir. Bunu yapma yönteminiz kullandığınız dile ve/veya yazılım iskeletine bağlıdır (örn. Numpy, Tensorflow, Scikit Learn, vb.). En önemlisi, bunu yapmaya ne karar verirseniz verin, söz konusu yöntemin hem eğitim hem de test boyunca tutarlı olmasıdır. İşte bu nedenle, iki yazılık bu seride önce görüntü verilerini TensorFlow ortamına kolaylıkla nasıl okutulacağından daha sonra okutulan bu görüntüleri modelin performansını arttırması açısından nasıl çeşitlendirilebileceğinden bir uygulama ile bahsedeceğim.