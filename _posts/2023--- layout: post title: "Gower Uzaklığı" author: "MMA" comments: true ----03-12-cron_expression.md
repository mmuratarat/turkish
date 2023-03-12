---
layout: post
title: "CRON İfadesi"
author: "MMA"
comments: true
---

Aynı görevi tekrar tekrar gerçekleştirmek oldukça zor olabilir. Yapılacak işleri zamanlamak, kullanıcıların bir sanal makine veya herhangi bir Unix benzeri işletim sistemi üzerindeki görevleri otomatikleştirmesine olanak tanır. Bu, değerli zamandan tasarruf sağlayarak kullanıcıların diğer temel görevlere odaklanmasını sağlar.

Bir cron işi (cron job), gelecekte yürütülecek görevleri planlamak için kullanılan bir Linux komutudur. Bu komut normalde periyodik olarak yürütülen bir işi planlamak için kullanılır - örneğin, her sabah bir bildirim göndermek için. Doğaları gereği cron işleri, sunucular gibi 7/24 çalışan bilgisayarlar için harikadır.


Bir cron işi oluşturmak için önce cron'un sözdizimini (syntax) ve biçimlendirmesini anlamanız gerekir. Aksi takdirde, cron görevlerinin doğru şekilde ayarlanması mümkün olmayabilir.

```
# ┌───────────── dakika (0 - 59)
# │ ┌───────────── saat (0 - 23)
# │ │ ┌───────────── ayın günü (1 - 31)
# │ │ │ ┌───────────── ay (1 - 12)
# │ │ │ │ ┌───────────── haftanın günü (0 - 6) (Pazar'dan Cumartesi'ye;
# │ │ │ │ │                                 7 bazı sistemlerde Pazar olarak da kullanılır) 
# │ │ │ │ │
# │ │ │ │ │
# * * * * * <command to execute>
```

Crontab sözdizimi, aşağıdaki olası değerlere sahip beş alandan oluşur:
* **Dakika (Minute)**. Komutun çalışacağı saatin dakikası, 0-59 arasında değişir.
* **Saat (Hour)**. 24 saatlik gösterimde 0-23 arasında değişen, komutun çalışacağı saat.
* **Ayın günü (Day of the month)**. 1-31 arasında değişen, kullanıcının komutun çalışmasını istediği ayın günü.
* **Ay (Month)**. Kullanıcının komutun çalışmasını istediği ay, 1-12 arasında değişir ve bu nedenle Ocak-Aralık'ı temsil eder.
* **Haftanın günü (Day of the week)**. Pazar-Cumartesi'yi temsil eden, 0-6 arasında değişen, bir komutun çalıştırılacağı haftanın günü. Bazı sistemlerde 7 değeri Pazar gününü temsil eder.

Alanlardan hiçbirini boş bırakmamalısınız.

Örneğin, her Cuma saat 17:37'de `root/backup.sh` betiğitini çalıştırmak için bir cron işi ayarlamak istiyorsanız, cron komutunuz şöyle görünmelidir:

```
37 17 * * 5 root/backup.sh
```

Yukarıdaki örnekte, 37 ve 17, 17:37'yi temsil eder. Ayın Günü ve Ay alanları için her iki yıldız da olası tüm değerleri belirtir. Bu, tarih veya ay ne olursa olsun görevin tekrarlanması gerektiği anlamına gelir. Son olarak, 5 Cuma gününü temsil eder. Sayı kümesini daha sonra görevin konumu takip eder.

Cron sözdizimini manuel olarak yazmak konusunda emin değilseniz, komutunuz için istediğiniz saat ve tarih için doğru sayıları üretmek üzere [Crontab Generator](https://crontab-generator.org/) veya [Crontab.guru](https://crontab.guru/) gibi ücretsiz araçları kullanabilirsiniz.

Cron komutunuz için doğru zamanı ayarlamak için, cron iş operatörleri hakkında bilgi sahibi olmak önemlidir. Her alana hangi değerleri girmek istediğinizi belirtmenize olanak tanırlar. Tüm crontab dosyalarında uygun operatörleri kullanmanız gerekir.

* **Yıldız (Asterisk) (`*`)**. Bir alandaki tüm olası değerleri belirtmek için bu operatörü kullanın. Örneğin cron işinizin dakikada bir çalışmasını istiyorsanız Dakika alanına yıldız işareti yazınız.
* **Virgül (Comma) (`,`)**. Birden çok değeri listelemek için bu operatörü kullanın. Örneğin, haftanın günü alanına `1,5` yazıldığında, her Pazartesi ve Cuma günü gerçekleştirilecek görev planlanır.
* **Tire (Hyphen) (`-`)**. Bir değerler aralığı belirlemek için bu operatörü kullanın. Örneğin Haziran'dan Eylül'e kadar bir cron işi kurmak istiyorsanız Ay alanına 6-9 yazmak işinizi görecektir.
* **Ayırıcı (Separator) (`/`)**. Her `n`'inci zaman aralığında çalışmayı belirtmek için bu operatörü kullanın. Örneğin, bir betiğin (script) on iki saatte bir çalışmasını istiyorsanız, Saat (Hour) alanına `*/12` yazın.
* **Son (Last) (`L`)**. Bu operatör ayın günü (Day of the month) ve haftanın günü (Day of the week) alanlarında kullanılabilir. Örneğin, haftanın günü alanına `3L` yazılması, ayın son Çarşamba günü anlamına gelir.
* **Hafta içi (Weekday) (`W`)**. Belirli bir zamandan haftanın en yakın gününü belirlemek için bu operatörü kullanın. Örneğin, ayın 1'i Cumartesi ise, ayın günü (Day of the month) alanına `1W` yazmak, komutu bir sonraki Pazartesi (3.) günü çalıştıracaktır.
* **Hask (`#`)**. Haftanın gününü (day of the week) belirlemek için, `#` operatöründen sonra 1 ile 5 arasında bir sayı kullanın. Örneğin, `1#2`, ayın ikinci Pazartesi günü anlamına gelir. 
* **Soru işareti (Question Mark) (`?`)**. "Ayın günü (Day of the month)" ve "haftanın günü (Day of the week)" alanlarına "belirli bir değer yok" girmek için bu operatörü kullanın.

Kullanıcının girilecek mantıksal sayı kümesini bulması gerekmeden cron işlerini zaman aralıklarında programlamak için özel dizgiler (strings) kullanılır. Bunları kullanmak için, bir `@` ve ardından basit bir ifade yazarsınız.

|         **Girdi**         |                        **Tanım**                       | **Karşılığı** |
|:-------------------------:|:------------------------------------------------------:|:-------------:|
| @yearly (veya @annually)  |        1 Ocak gece yarısı yılda bir kez çalıştır       |  `0 0 1 1 *`  |
|         @monthly          | Ayda bir kez, ayın ilk gününün gece yarısında çalıştır |  `0 0 1 * *`  |
|          @weekly          |    Pazar sabahı gece yarısı haftada bir kez çalıştır   |  `0 0 * * 0`  |
|  @daily (veya @midnight)  |          Gece yarısı günde bir kez çalıştırın          |  `0 0 * * *`  |
|          @hourly          |            Saat başında saatte bir çalıştır            |  `0 * * * *`  |
|          @reboot          |                  Başlangıçta çalıştır                  |       —       |

Cron ifadelerini insan tarafından okunabilir açıklamalara dönüştüren bir .NET kitaplığı da kullanabilirsiniz.
* Github: https://github.com/bradymholt/cron-expression-descriptor
* Bağlantı: https://bradymholt.github.io/cron-expression-descriptor/
