---
layout: post
title:  "Kanarya ve Mavi-Yeşil Dağıtım Stratejileri"
author: "MMA"
comments: true
---

# Kanarya Dağıtım Stratejisi / Sürümü (Canary Deployment Strategy / Release)

Kanarya Dağıtım Stratejisi, bir yazılımdaki değişikliği tüm altyapıya yaymadan ve herkesin kullanımına sunmadan önce küçük bir kullanıcı alt kümesine aşamalı bir şekilde dağıtarak üretimde yeni bir yazılım sürümü sunma riskini azaltmaya yönelik bir tekniktir.

Aşağıda bahsedeceğimiz Mavi-Yeşil Dağıtım Stratejisine benzer şekilde, ilk olarak, yazılımınızın yeni sürümünü, altyapınızın hiçbir kullanıcının yönlendirilmediği bir alt kümesine dağıtarak başlarsınız.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/canary_1.png?raw=true)

Yeni sürümden memnun olduğunuzda, seçilen birkaç kullanıcıyı ona yönlendirmeye başlayabilirsiniz. Hangi kullanıcıların yeni sürümü göreceğini seçmek için farklı stratejiler vardır: basit bir strateji rastgele bir örneklem kullanmaktır; bazı şirketler dünyaya yayınlamadan önce, yazılımlarının yeni sürümünü, dahili kullanıcılarına ve çalışanlarına yayınlamayı seçiyor; Daha sofistike bir diğer yaklaşım ise kullanıcıları profillerine ve diğer demografik özelliklerine göre seçmektir.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/canary_2.png?raw=true)

Yeni sürüme duyduğunuz güven arttıkça, onu altyapınızdaki daha fazla sunucuya yayınlamaya ve daha fazla kullanıcıyı ona yönlendirmeye başlayabilirsiniz.

Yeni sürüme geçiş aşaması, tüm kullanıcılar bu yeni sürüme yönlendirilene kadar sürer. Bu noktada eski altyapıyı devre dışı bırakabilirsiniz. Yeni sürümle ilgili herhangi bir sorun bulursanız, geri alma stratejisi, siz sorunu çözene kadar kullanıcıları eski sürüme geri yönlendirmektir.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/canary_3.png?raw=true)

Kanarya sürümlerini kullanmanın bir yararı, sorun bulunursa güvenli bir geri alma stratejisi ile bir üretim ortamında yeni sürümün kapasite testini yapılabilme kolaylığıdır. Yükü yavaşça artırarak, yeni sürümün üretim ortamını nasıl etkilediğiyle ilgili ölçütleri (metrics) izleyebilir ve yakalayabilirsiniz. Bu, tamamen ayrı bir kapasite test ortamı oluşturmaya alternatif bir yaklaşımdır, çünkü ortam olabildiğince üretime benzer olacaktır.

Bu tekniğin adı, kömür madenlerinde bir kafeste kanarya taşıyan madencilerden gelmektedir. Madenciler madene girmeden önce, kafesteki kanaryayı madene salardı. Madene zehirli gazlar sızmışsa, madencileri öldürmeden önce kanaryayı öldürürdü. Bir kanarya sürümü, tüm üretim altyapınızı veya kullanıcı tabanınızı etkilemeden önce olası sorunlar için benzer bir erken uyarı biçimi sağlar. Kanarya dağıtım stratejisi, aşamalı sunum (phased rollout) veya arttırımlı sunum (incremental rollout) olarak adlandırılır.

Büyük, dağıtımlı (distributed) senaryolarda, hangi kullanıcıların yeni sürüme yönlendirileceğine karar vermek için bir yönlendirici (router) kullanmak yerine, farklı parçalama stratejileri kullanmak da yaygındır. Örneğin, coğrafi olarak dağıtılmış kullanıcılarınız varsa, yeni sürümü önce bir bölgeye veya belirli bir konuma sunabilirsiniz; Birden fazla markanız varsa, önce tek bir markaya geçiş yapabilirsiniz, vb. 

Kanarya sürümleri, teknik uygulamadaki benzerlikler nedeniyle A/B testini gerçekleştirmenin bir yolu olarak kullanılabilir. Ancak, bu iki yöntemi karıştırarak kullanmamak tercih edilir: kanarya sürümleri sorunları ve gerilemeleri tespit etmenin iyi bir yolu iken, A/B testi, varyant uygulamaları kullanarak bir hipotezi test etmenin bir yoludur. Bir kanarya dağıtım stratejisi ile gerilemeleri tespit etmek için bir iş ölçütünü (metric) izlerseniz, bu ölçütü, A/B testi için kullanmak da sonuçları etkileyebilir. Daha pratik bir not olarak, bir A/B testi için istatistiksel önemi göstermek üzere yeterli veri toplamak günler sürebilirken, bir kanarya sunumunun dakikalar veya saatler içinde tamamlanmasını isteyebilirsiniz.

Kanarya sürümlerini kullanmanın bir dezavantajı, yazılımınızın birden çok sürümünü aynı anda yönetmenizi gerektirmesidir. Üretimde aynı anda ikiden fazla sürüm çalıştırmaya bile karar verebilirsiniz, ancak en iyisi eşzamanlı sürümlerin sayısını minimumda tutmaktır.

Kanarya sürümlerini kullanmanın zor olduğu başka bir senaryo, kullanıcıların bilgisayarlarına veya mobil cihazlarına yüklenen yazılımı dağıttığınız zamandır. Bu durumda, yeni sürüme yükseltmenin ne zaman yapılacağı üzerinde daha az kontrole sahip olursunuz.

Veritabanı değişikliklerini yönetmek, kanarya sürümleri yapılırken de dikkat gerektirir. Yine, dağıtım aşamasında veritabanının uygulamanın her iki sürümünü de desteklemesine izin verebilirsiniz.

# Mavi-Yeşil Dağıtım Stratejisi (Blue-Green Deployment Strategy)

Dağıtımı otomatikleştirmedeki zorluklardan biri, yazılımı testin son aşamasından canlı üretime alan geçişisin (cutover) kendisidir. Arıza süresini (downtime) en aza indirmek için genellikle bunu hızlı bir şekilde yapmanız gerekir. Mavi-yeşil dağıtım yaklaşımı (kırmızı-siyah dağıtım olarak da bilinir - literatüre baktığınızda, bazen, kırmızı / siyah dağıtım  stratejisinin farklı bir strateji olduğunu görebilirsiniz. Kırmızı-siyah dağıtım, Netflix, Istio ve konteyner düzenlemeyi destekleyen diğer yazılım iskeletleri ve platformlar tarafından kullanılan daha yeni bir terimdir. Her iki terim de aynı şeyi ifade eder ve ikisi arasındaki herhangi bir teknik fark muhtemelen yalnızca belirli bir ekip veya şirket içinde anlamlı olacaktır <sup>[1](#myfootnote1)</sup>), mümkün olduğunca aynı iki üretim ortamına sahip olmanızı sağlayarak bunu gerçekleştirir. Örneğin, herhangi bir zamanda, mavi ortam, yazılımınızın n'inci versiyonunu çalıştıran canlı bir prodüksiyon ortamı olsun. Yeşim ortam ise yazılımınızın n+1'inci versiyonunu çalıştıran ve mavi ortamın tam bir kopyası olsun.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/blue_green1.png?raw=true)

Yazılımınızın yeni bir sürümünü hazırlarken, yeşil ortamda ise testlerinizin son aşamasını yaparsınız (yani, yeni modelin hem teknik hem de iş ölçütlerinin tamamını karşılayıp karşılamadığını kontrol edersiniz.) Tüm testler başarılı bir şekilde çalıştıktan sonra yazılımınızın yeni sürümü yeşil ortamda çalışmaya başlar ve böylelikle yönlendiriciyi (router), gelen tüm isteklerin yeşil ortama gitmesi için değiştirirsiniz - mavi olan artık boşta (Bir süre herhangi bir sorun bulunmazsa, mavi ortamı kaldırabilirsiniz.)

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/blue_green2.png?raw=true)

Mavi-yeşil dağıtım aynı zamanda hızlı bir geri dönüş (rollback) yolu sağlar - herhangi bir sorun olursa, yönlendiriciyi (eğer silmediyseniz) tekrar mavi ortamınıza yönlendirebilirsiniz. Hala yeşil ortam canlıyken kaçırılan işlemlerle başa çıkma sorunu var, ancak tasarımınıza bağlı olarak, yeşil canlıyken mavi ortamı yedek olarak tutacak şekilde işlemleri her iki ortama da besleyebilirsiniz. Veya uygulamayı kesmeden önce salt okunur moduna getirebilir, bir süre salt okunur modda çalıştırabilir ve ardından okuma-yazma moduna geçirebilirsiniz. Bu, çözülmemiş birçok sorunu gidermek için yeterli olabilir.

Yeşil ortamınızı hayata geçirdikten ve istikrarından memnun olduğunuzda, bir sonraki dağıtımınız (uygulamanızın bir sonraki versiyonu) için son test adımı gerçekleştirmek üzere hazırlık ortamınız olarak mavi ortamı kullanırsınız. Bir sonraki sürümünüz için hazır olduğunuzda, daha önce maviden yeşile yaptığınız gibi yeşilden maviye geçersiniz. Bu şekilde, hem yeşil hem de mavi ortamlar düzenli olarak canlı, önceki sürüm (geri alma için) ve bir sonraki sürümü ayarlama arasında gidip gelir.

Bu yaklaşımın bir avantajı, otomatik yedekleme çalışmaları için ihtiyaç duyduğunuzla aynı temel mekanizma olmasıdır. Dolayısıyla bu, olağanüstü durum kurtarma prosedürünüzü (disaster-recovery procedure) her sürümde test etmenize olanak tanır. 

Temel fikir, aralarında geçiş yapmak için kolayca değiştirilebilir iki ortama sahip olmaktır, ayrıntıları değiştirmenin birçok yolu vardır.

<a name="myfootnote1">1</a>:https://stackoverflow.com/questions/45259589/whats-the-difference-between-red-black-deployment-and-blue-green-deployment
