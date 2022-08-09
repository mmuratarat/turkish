---
layout: post
title: "MongoDB Sunucusu ve MongoDB Kabuğu Kurulumu"
author: "MMA"
comments: true
---

Veri modelleri (data models), verilerin nasıl temsil edildiğini açıklar. Verileri temsil etmeyi seçtiğiniz yol, sadece sistemlerinizin inşa edilme şeklini değil, aynı zamanda sistemlerinizin çözebileceği problemleri de etkiler. Birbirine zıt görünen ama aslında birbirlerine yakınsayan iki tür veri modeli vardır: ilişkisel modeller (relational models) ve NoSQL modeller.

İlişkisel veri modeli (relational data model), e-ticaretten finansa, sağlıktan sosyal ağlara kadar birçok kullanım örneğine genellenebilmiştir. Ancak, belirli kullanım durumları için bu model kısıtlayıcı olabilir. Örneğin, ilişkisel model, verilerinizin katı bir şema (schema) izlemesini talep eder ve şema yönetimi (schema management) zahmetlidir.

İlişkisel veri modeli etrafında oluşturulan bir veritabanı, ilişkisel veritabanı (relational database) olarak adlandırılır. Günümüzde ilişkisel veritabanları için en popüler sorgu dili SQL'dir. Ancak, ilişkisel-olmayan (non-relational) veritabanları da mevcuttur. Bu nedenle iş sadece SQL ile bitmiyor. 

NoSQL, başlangıçta ilişkisel-olmayan veritabanlarını tartışmak üzere toplanılan bir buluşma için bir hashtag olarak kullanılmıştır. Bu nedenle, bu tür veritabanları NoSQL olarak isimlendirilmiştir. NoSQL ismi "Not only SQL (Sadece SQL değil)"den gelmektedir. Bunun sebebi birçok NoSQL veri sisteminin ilişkisel modelleri de desteklemesidir.

MongoDB, platformlar-arası, doküman-odaklı (document-oriented) bir veritabanı programıdır (www.mongodb.com). NoSQL veritabanı programı olarak sınıflandırılan MongoDB, isteğe bağlı şemalarla JSON benzeri dokümanlar kullanır. Stackoverflow’un her yıl gerçekleştirdiği anketin 2022 yılındaki sonucuna göre veritabanları sıralamasında MongoDB en üst sıralarda yerini almaktadır - https://lnkd.in/dsWQubUd

Bir önceki yazılarımda bir PostgreSQL veritabanını kişisel bilgisayarınıza nasıl yükleyeceğinizi ve bu veritabanı ile etkileşimde kalmak için kullanabileceğiniz Grafik Kullanıcı Arayüzlerinden (Graphical User Interface - kısaca, GUI) bahsettim. https://www.linkedin.com/posts/mmuratarat_postgresql-ve-pgadmin4-grafik-ara-y%C3%BCz%C3%BC-kurulumu-activity-6961967826848878592-oatK?utm_source=linkedin_share&utm_medium=member_desktop_web

Herhangi bir ilişkisel veritabanını bilgisayarınıza kurabildiğiniz gibi, bir MongoDB veritabanını da kişisel bilgisayarınıza yerel olarak kurabilir ve bol bol pratik yapabilsiniz. MongoDB için aynı şirket tarafından özel olarak kullanılan Compass isimli grafik kullanıcı arayüzünü de indirebilirsiniz. Ancak, MongoDB için benim kişisel tercihim, Compass'ı kullanmak yerine, komutları daha kolay öğrenebilmek adına MongoDB Kabuğu'nu (MongoDB Shell) kullanmaktır. Aşağıdaki dokümanda, MongoDB Sunucusu'nu (MongoDB Server) ve MongoDB Kabuğunu MacOS bilgisayarınıza nasıl kolaylıkla kurabileceğinizi ve bir MongoDB veritabanını NoSQL ile hemen kullanmaya nasıl başlayabileceğinizi anlatan Türkçe yönergeyi bulabilirsiniz.

<embed src="https://mmuratarat.github.io/turkish/files/mongodb_instructions.pdf" width="800" height="700" frameborder="0" allowfullscreen>
