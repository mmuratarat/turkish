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
