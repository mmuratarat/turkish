---
layout: post
title:  "Streamlit ile Heroku'da Bir Web Uygulaması Dağıtımı"
author: "MMA"
comments: true
---

Bu eğiticide, bir Python kütüphanesi olan Streamlit'i kullanarak Excel dosyalarını bir web uygulamasına nasıl dönüştüreceğinizi göstereceğim. Streamlit, makine öğrenmesi ve veri bilimi projeleri için güzel, özel web uygulamaları oluşturmayı ve paylaşmayı kolaylaştıran açık kaynaklı bir Python kütüphanesidir. Streamlit'in güzelliği, HTML, CSS veya JavaScript bilmenize gerek kalmadan doğrudan Python'da web uygulamaları oluşturabilmenize olanak sağlamasıdır. Web Uygulaması tamamen etkileşimlidir ve Excel dosyasında her değişiklik yaptığınızda güncellenecektir.

İlk olarak Masaüstü'nde (Desktop) `ExcelStreamlit_WebApp` isimli bir klasör oluşturalım. 


## Sanal Ortam Oluşturmak

Bu eğiticiye başlamadan önce bilgisayarınızdaki düzeni bozmamak adına, aşağıda yapacağımız tüm işlemleri sanal bir ortam içerisinde gerçekleştirelim. Basitçe ifade etmek gerekirse, bir sanal ortam, diğer projeleri etkileme endişesi olmadan belirli bir proje üzerinde çalışmanıza izin veren, Python’un yalıtılmış bir çalışma kopyasıdır. Her proje için birden çok Python versiyonunun aynı makineye kurulumuna olanak tanır. Aslında Python’un ayrı kopyalarını kurmaz, ancak ortam değişkenlerini ve paketleri izole ettiği için farklı proje ortamlarını izole tutmanın akıllıca bir yolunu sağlar. Yanlış paket versiyonlarindan şikayet eden hata mesajlarının bir çaresidir 

Python'da sanal ortam oluşturmak için kullanabileceğiniz bazı popüler kütüphaneler/araçlar şöyledir: `virtualenv`, `virtualenvwrapper`, `pvenv` ve `venv`. Burada `virtualenv` paketine odaklanacağız.

`virtualenv`'in sisteminizde zaten kurulu olması muhtemeldir. Bununla birlikte, bu paketin global sisteminizde kurulu olup olmadığını, eğer kurulu ise, hangi sürümü kullandığınızı kontrol edin:

```
which virtualenv
```

veya

```
virtualenv --version`
```

Eğer bu paket kurulu değilse, `virtualenv` paketini `pip3 install virtualenv` komutunu kullanarak yüklemeniz gerekmektedir.

Bunu işlemi gerçekleştirdikten sonra, önce bir Terminal penceresi açın ve oluşturduğunuz `ExcelStreamlit_WebApp` isimli dizine gidin:

```
(base) Arat-MacBook-Pro-2:~ mustafamuratarat$ cd Desktop/ExcelStreamlit_WebApp/
```

ve aşağıda verilenleri gerçekleştirin. Aşağıdaki kod, kurduğumuz `virtualenv` modülünü çağıracak ve mevcut klasörümüzün içinde `myEnvironment` adlı yeni bir klasör oluşturacak (yani `myEnvironment` isminde bir sanal ortam) ve `myEnvironment`'te yeni bir Python kurulumu yükleyecek (tabii ki kurmak istediğiniz Python sürümünü değiştirebilirsiniz). Ben burada sistemimde kurulu olan aynı Python 3 sürümünü sanal ortamımda da kullanmak istediğim için `which python3` komutunu kullandım:

```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ python3 -m venv myEnvironment
```

Yukarıdaki komut, projenizde tüm bağımlılıkların (dependencies) kurulu olduğu bir `myEnvironment` dizini oluşturur. Kurulum tamamlandıktan sonra, bu sanal ortamı kullanmak istiyorsanız, izole edilmiş ortamınızı etkinleştirmeniz gerekir:

```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ source myEnvironment/bin/activate
(myEnvironment) (base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ 
```

Burada dikkat etmeniz gereken nokta, komut satırının bilgisayarınızın adından, yeni oluşturduğunuz sanal ortamın ismine dönüşmesidir. Proje üzerinde her çalışmak istediğinizde bu sanal ortamı aktive etmelisiniz. Bu nedenle `source  myEnvironment/bin/activate` kodunu çalıştırmayı unutmayın. Ancak, her seferinde bu kodu yazmak istemediğinizde ve bilgisayarınız başlar başlamaz, sanal ortamınızın aktive olmasını istiyorsanız [şu sayfada](https://askubuntu.com/a/1175106/1187527){:target="_blank"} bulunan adımları takip ederek bir betik yazabilirsiniz.

Artık bu ortamda istediğiniz paketleri ve bu paketlerin versiyonlarını global sisteminizi etkilemeden kurabilirsiniz.

## Web uygulaması için gerekli kütüphaneleri sanal ortama kurmak

Web uygulaması için kullanacağımız paketler `pandas`, `streamlit`, `plotly`, `plotly-express` ve `Pillow`'dur. Bu kütüphaneleri sanal ortamımıza yüklememiz gerekmektedir. Bunun için pip Python paket yöneticisini kullanabilirsiniz (Anaconda kullanıyorsanız, gerekli komutları tercih ediniz.)

```
(myEnvironment) (base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ pip3 install plotly plotly-express pandas streamlit Pillow
```

Bu komut satırı çalıştıktan sonra gerekli kütüphaneler sanal ortamınızda kullanılmaya hazır olacaktır. 

## Veri Seti

Burada kullanacağımız veri seti, Kaggle (www.kaggle.com)'da bulunan Dünya Mutluluk Raporu 2021 veri dosyasıdır. Aşağıdaki bağlantıya giderek 21.69 kB büyüklüğündeki bu csv dosyasını indiriniz ve dosyayı, oluşturduğumuz `ExcelStreamlit_WebApp` klasörüne taşıyınız. 

## Visual Studio Code

Burada JupyterLab yerine Visual Studio Code kullanacağım. Siz istediğiniz Entegre Geliştirme Ortamını (Integrated Development Environment - IDE) tercih edebilirsiniz. 

Visual Studio Code'u çalıştırdıktan sonra `ExcelStreamlit_WebApp` klasörünü pencereye sürükleyiniz. Burada önemli olan Visual Studio Code'un oluşturduğunuz sanal ortamı tanımasıdır. Varsayılan olarak, Python uzantısı sistem yolunda bulduğu ilk Python yorumlayıcısını arar ve kullanır. Belirli bir ortamı seçmek için, klavyenizde `⇧⌘P` tuşlarına aynı anca basarak (ya da `View > Command Palette` ile), Command Palette'ini (Komut Paleti) açmalı ve boşluğa `Python: Select Interpreter` (Python: Yorumlayıcı Seç) komutunu yazmalısınız.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc1.png?raw=true)

Gelen dropdown menüsünde, `myEnvironment` sanal ortamı aramalı ve bulduğunuzda ona tıklamalısınız.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc2.png?raw=true)

Visual Studio Code penceresinin altındaki Durum Çubuğu (Status Bar) geçerli yorumlayıcıyı gösterecektir!

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc3.png?raw=true)

## Uygulamayı Yazma

Artık uygulamayı yazmaya başlayabiliriz. Visual Studio Code penceresinde New File (Yeni Dosya) butonuna tıklayalım:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc4.png?raw=true)

Daha sonra, açılacak boşluğa `app.py` yazalım ve `Enter` tuşuna basalım:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc5.png?raw=true)

Bunu yapmak, `ExcelStreamlit_WebApp` klasörünün içinde, Python kaynak kodunu içerisine yazabileceğiniz, `app` isimde `.py` uzantılı bir dosya yaratacaktır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc6.png?raw=true)

İlk olarak `pandas` kütüphanesini içe aktaralım ve bu kütüphanedeki `read_csv()` fonksiyonunu kullanarak veriyi python ortamına okutalım. 

Grafikleri çizmek için `plotly-express` ve web uygulamasını yazmak üzere kullanacağınız `streamlit` kütüphaneleri de içe aktaralım:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc7.png?raw=true)

```python
import pandas as pd
import plotly_express as px
import streamlit as st

data = pd.read_csv(filepath_or_buffer = "world-happiness-report-2021.csv")

print(data)
``` 

İlk adım olarak, `streamlit` kütüphanesindeki `set_page_config` fonksiyonunu kullanarak web uygulamasının bazı yapılandırmasını ayarlayacağım. Başlığı ayarladıktan sonra, web uygulamamıza bir favicon'da atayacağım. Bir favicon, web tarayıcılarında bir web sitesini simgeleyen küçük kare bir resimdir. Bir resim yerine, Streamlit, bir emoji de desteklemektedir. Ben, bir gülücük emojisi kullanacağım (`:smile:`) ([Buradaki](https://www.webfx.com/tools/emoji-cheat-sheet/) web sayfasında çeşitli emojileri bulabilirsiniz. Yapmanız gereken, istediğiniz emoji'yi seçmeniz ve onun kod ismini (codename) streamlit uygulamanıza kopyalamanız.) Ardından, sayfa içeriğinin nasıl düzenlenmesi gerektiğini belirleyeceğim. Varsayılan sayfa düzeni (layout) "ortalanmış (centered)" olarak ayarlanmıştır. Ancak, ben tüm ekranı kullanmak istiyorum. Bunu, `set_page_config` fonksiyonundaki `layout` argümanını `"wide"` olarak ayarlayarak yapabiliriz.

```python
import pandas as pd
import plotly_express as px
import streamlit as st


st.set_page_config(page_title="Dünya Mutluluk Rporu 2021",
                   page_icon=":smile:",
                   layout="wide")
```

Sayfa ayarlarını tamamladıktan sonra, web uygulamanızı ziyaret eden kullanıcılar için güzel bir başlık ve açıklama ekleyelim. Bunun için `streamlit` kütüphanesindeki `title` ve `markdown` fonksiyonları kullanılır:

```python
import pandas as pd
import plotly_express as px
import streamlit as st


st.set_page_config(page_title="Dünya Mutluluk Rporu 2021",
                   page_icon=":smile:",
                   layout="wide")

st.title("Dünya Mutluluk Raporu Analitiği")
st.markdown("Hoş geldiniz! Aramanızı daraltmak için lütfen ekranınızın solunda bulunan kenar çubuğundaki filtrelerden seçim yapınız.")
```

NOT: Birazdan kenar çubuğunu (sidebar) da ekleyeceğiz!

Streamlit web uygulamasını çalıştırmadan önce, `DataFrame`'i konsolda yazdırmak yerine sayfa üzerinde görüntüleyeceğim. Bunun için `streamlit` kütüphanesindeki `dataframe` fonksiyonunu kullanırız.

```python
import pandas as pd
import plotly_express as px
import streamlit as st


st.set_page_config(page_title="Dünya Mutluluk Raporu 2021",
                   page_icon=":smile:",
                   layout="wide")

st.title("Dünya Mutluluk Raporu Analitiği")
st.markdown("Hoş geldiniz! Aramanızı daraltmak için lütfen ekranınızın solunda bulunan kenar çubuğundaki filtrelerden seçim yapınız.")

data = pd.read_csv(filepath_or_buffer = "world-happiness-report-2021.csv")

st.dataframe(data=data)
```

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc7-2.png?raw=true)

Bu kod satırını da ekledikten sonra, Visual Studio Code penceresindeki Terminal üzerinde `streamlit run app.py` komutunu çalıştırırız. Bu komutu ayritten başka bir Terminal penceresinde de çalıştırabilirsiniz. Yapmanız gereken `app.py` Python betiğiniz (script) ile aynı klasörde olduğunuzdan emin olmak.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc8.png?raw=true)

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc9.png?raw=true)

Bu komut satırını çalıştırdığınızda, sayfa ismini (page title), sayfa ikonunu (page ikon) ve `DataFrame'i içeren web uygulaması http://localhost:8501 lokal URL'inde web tarayıcınızda açılacaktır. 

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc10.png?raw=true)

Tamam şimdiye kadar her şey çok iyi!

Artık web uygulamamız çalıştığına göre, kenar çubuğu (sidebar) bölümünü oluşturacağım. Kullanıcının filtre kriterlerini toplamak ve bu filtreleri veri kümemize uygulamak için kenar çubuğunu kullanacağız.

Veri kümesinde sadece iki tane kategorik değişken vardır (`Country name` ve `Regional Indicator`). Bu nedenle, veri kümesini farklı ülke (country) ve bölgelere (region) göre filtrelemeyi planlıyorum.

İlk olarak aşağıdaki kod satırı ile kenar çubuğunu başlatıyoruz:

```python
st.sidebar.header("Lütfen burada filtreleyin:")
```

Filtreleme gerçekleştirmek için `streamlit` kütüphanesindeki `multiselect` bileşenini kullanacağız. İlk olarak bölge için gerçekleştirelim. `multiselect` fonksiyonundaki `label` (etiket) argümanı için `"Bölge seçiniz:"` yazacağız ve `options` argümanı için veri kümesindeki `Regional indicator` sütunundaki benzersiz değerleri (unique values) kullanacağız. `pandas` kütüphanesi ile bu değerlere `data['Regional indicator'].unique()` kodu ile kolaylıkla ulaşılabilir ve çıktısı:

```python
array(['Western Europe', 'North America and ANZ',
       'Middle East and North Africa', 'Latin America and Caribbean',
       'Central and Eastern Europe', 'East Asia', 'Southeast Asia',
       'Commonwealth of Independent States', 'Sub-Saharan Africa',
       'South Asia'], dtype=object)
```

olacaktır.

Web uygulamasını başlatırken görülebilecek grafik için, `multiselect` fonksiyonundaki `default` anahtar kelime argümanını kullanarak varsayılan değeri de seçebiliriz. Ben, sadece Batı Avrupa'da (West Europe) bulunan ülkeleri görüntülemek istiyorum: 

```python
region_selected = st.sidebar.multiselect(label = "Bölge seçiniz:", 
                                         options = data['Regional indicator'].unique(),
                                         default = data['Regional indicator'].unique())
```


Web uygulamasının başında tüm `DataFrame`'i

```python
st.dataframe(data=data)
```

kod satırı ile yazdırmıştık. Bunun yerine yaptığımız bölge seçimlerine göre `DataFrame`'i web uygulaması üzerinde gösterelim. Bunu gerçekleştirmek oldukça kolaydır. Yapmanız gereken `pandas` kütüphanesindeki `query` fonksiyonunu kullanmaktır.

```python
data_regions = data.query("`Regional indicator` == @region_selected")
```

`pandas` kütüphanesinin 0.25 ve sonrası versiyonlarında, kesme işareti (backtick yani ` `` `) kullanarak arasında boşluk bulunan sütun isimlerine sahip sütunları kullanabilirsiniz.

`@` ile bir değişkene referans verebilirsiniz.

Daha sonra yapmamız gereken, benzer komutla `st.dataframe(data=data_regions)` ile veri kümesini yazdırmak.

Tamam, bunu da tamamladıktan sonra yaptığımız değişikliği kaydedelim ve tarayıcımızda açık olan web uygulamamızı yenileyelim.

Aşağıdaki ekran görüntüsünde görüldüğü üzere, tüm bölgeler (region) kenar çubuğunda varsayılan olarak görülmektedir. 

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc11.png?raw=true)

Filtreleme gerçekleştirip, örneğin sadece Batı Avrupa'da bulunan ülkelere ait verinin `DataFrame'ini gösterebiliriz:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc12.png?raw=true)

Bölgeye göre filtrelemeye ek olarak, bir de kaydırma (slider) şeklinde bir filtreleme ekleyelim. Bunun için `slider` fonksiyonunu kullanılır. 

```python
score = st.sidebar.slider(label = 'Minimum Merdiven Skoru Seçiniz:', min_value=5, max_value=10, value = 10) 
```

Bu kod satırı, veri kümesindeki `Ladder score` sütunu için kullanılacaktır.

Peki, `Ladder score` sütunu nedir? Dünya Mutluluk Raporundaki değerler bir Cantril merdiven anketine dayanmaktadır. "Basamakları sıfırdan 10’a kadar numaralanmış bir merdiven hayal edin. En üst basamak, sahip olabileceğiniz en iyi hayatı, en alt basamak ise sahip olabileceğiniz en kötü hayatı simgeliyor. Şu anda kaç numaralı basamaktasınız?" gibi bir soru cevaplandırılır.  

İşte veri kümesindeki `Ladder score` merdiven skoru olarak adlandırılabilir. Filtreleme olarak minimum değer 5, ve maximum değer 10 seçilebilir (bu sütunun alabileceği değerlere göre `min_value` ve `max_value` argümanlarına karar verilir). Uygulama ilk başladığında, `value` argümanına atanan değer gösterilecektir. Ben bu argümanının değerini `10` olarak ayarladım.

Bölge filtrelemede yaptığımıza benzer şekilde sekilde, bu filtreleme için de bir `DataFrame yazdırabiliriz:

```python
data_ladderScore = data[data['Ladder score'] <= score] 

st.dataframe(data=data_ladderScore)
```

Ayrıca, `markdown` fonksiyonu kullanılarak sayfa içerisinde farklı başlıklar verilebilir. Bu başlıkların şekillendirmesi HTML ile gerçekleştirilebilir:

```python
st.markdown(body = '<p style="font-family:sans-serif; font-size: 24px;">Merdiven Skoruna Göre Veri Kümesi</p>', unsafe_allow_html=True)
```

Betiği kaydedip, sayfayı yeniledikten sonra, yaptıklarımıza göz atabiliriz:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc13.png?raw=true)

Bunlara ek olarak, anasayfada, tüm ülkelerin (`Ladder score` değişkeni ile verilen) merdiven skorlarının ortancasını; `Healthy life expectancy` değişkeni ile verilen  yaşam beklentilerinin tüm dünya için ortalamasını ve en büyük merdiven skoruna sahip ülkeyi yani en mutlu  ülkenin ismini ve bayrağını üç farklı sütuna (bir sol, orta ve sağ sütun) yazdıralım. Bunu yapmak için `streamlit` kütüphanesindeki `columns` fonksiyonu kullanılır. Sütunlar yaratıldıktan sonra, bağlam yöneticisi (context manager) kullanılarak, içerik farklı sütunlara yerleştirilir:

```python
# Ortanca Merdiven Skoru
median_ladderScore = np.median(data['Ladder score'])
howMany_Ladders = "\U0001fa9c" * int(median_ladderScore)

# Ortalama yaşam beklentisi
mean_lifeExpectancy = np.mean(data['Healthy life expectancy'])

# En mutlu ülke
happiest_country = data[data['Ladder score'] == np.max(data['Ladder score'])]['Country name'].item()
finland_flag = ":flag-fi:"

left_column, middle_column, right_column = st.columns(3)

with left_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Ortanca Merdiven Skoru:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{median_ladderScore} {howMany_Ladders}")
with middle_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">Ortalama Yaşam Beklentisi:</p>', unsafe_allow_html=True)
    st.subheader(body =  f"{mean_lifeExpectancy: .2f} yıl")
with right_column:
    st.markdown(body = '<p style="font-size: 20px; font-weight:bold;">En Mutlu Ülke:</p>', unsafe_allow_html=True)
    st.subheader(body=f"{finland_flag} {happiest_country}")

st.markdown("---")
```

Burada, `\U0001fa9c`kodu merdiven emojisi için Python kodudur  (https://emojiterra.com/ladder/) ve `:flag-fi:` kodu Finlandiya bayrağı için kısa koddur (https://emojiterra.com/flag-for-finland/).

Ayrıca, `markdown` fonksiyonu içerisinde üç kısa çizgi kullanılarak bir ayırıcı eklenebilir (`st.markdown("---")`).

Betiği kaydedip, sayfayı yeniledikten sonra, yaptıklarımıza göz atabiliriz:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc14.png?raw=true)

Bu web uygulamasının bir sonraki bölümü için bir saçılım grafiği (scatter plot) ve bir çubuk grafiği (bar plot) ekleyelim.

Streamlit kütüphanesi, Matplotlib, Seaborns, Ploty, Altair gibi birkaç farklı grafik kütüphanesini desteklemektedir. Ayrıca, tek bir kod satırı tarafından çağrılabilen çizgi grafiği ve alan grafiği gibi birkaç yerel grafik sağlar, örneğin:

```python
#Line Chart
st.line_chart(data=None, width=0, height=0, use_container_width=True)
#Area Chart
st.area_chart(data=None, width=0, height=0, use_container_width=True)
```

Ancak bu eğiticide, saçılım grafiği ve çubuk grafik için `plotly-express` kütüphanesini kullanacağız.

Saçılım grafiği, "kişi başına GSYİH (logged GDP per capita)" değişkeni ile "sağlıklı yaşam beklentisi (healthy life expectancy)" değişkenleri arasında olsun. 

Çubuk grafiği ise merdiven skoru (ladder score) ve ülke isimleri (country name) değişkenleri arasında olsun.

```python
# Saçılım Grafiği
fig_gdp_lifeExpect = px.scatter(data_frame = data_regions,
x="Logged GDP per capita",
y="Healthy life expectancy",
size="Ladder score",
color="Regional indicator",
hover_name="Country name",
size_max=10,
title='Kişi başına GSYİH ve Yaşam Beklentisi',
template='ggplot2')

# Çubuk grafiği
fig_ladderByCountry = px.bar(data_frame= data_regions, x='Ladder score', y='Country name', orientation='h', title='Ülkeler bazında merdiven skorları', template='ggplot2')

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_gdp_lifeExpect, use_container = True)
right_column.plotly_chart(fig_ladderByCountry, use_container = True)
```

Burada iki farklı yazım türü de gösterilmiştir. 

Betiği kaydedip, sayfayı yeniledikten sonra, yaptıklarımıza göz atabiliriz:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc15.png?raw=true)

Tabii ki, bu grafikleri daha da güzelleştirmek tamamiyle sizin elinizde...

Son olarak, sayfa üst bilgisini (header), sayfa alt bilgisini (footer) ve ana menüyü silelim:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc16.png?raw=true)

Bunu özel CSS kullanarak gerçekleştirebiliriz:

```python
# ---- STREAMLIT STİLİNİ SAKLA ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
```

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc17.png?raw=true)

## Temayı Kişileştirme 

Streamlit Temalar ile  uygulamaların görünümünü değiştirme olanağına sahipsiniz (https://docs.streamlit.io/library/advanced-features/theming). Özel temalar, yapılandırma dosyasında da tanımlanabilir: `./.streamlit/config.toml`. `[theme]` bölümünün altında, özel bir tema oluşturmak için renk değişkenleri tanımlanabilir:

```
[theme]

# Primary accent for interactive elements
primaryColor = '#7792E3'

# Background color for the main content area
backgroundColor = '#273346'

# Background color for sidebar and most interactive widgets
secondaryBackgroundColor = '#B9F1C0'

# Color used for almost all text
textColor = '#FFFFFF'

# Font family for all text in the app, except code blocks
# Accepted values (serif | sans serif | monospace) 
# Default: "sans serif"
font = "sans serif"
```

Web uygulamamız için temayı değiştirmek üzere, ilk olarak, `ExcelStreamlit_WebApp` klasörü altında, `.streamlit` dizinini yaratmamız gerekmektedir. Bunun için Terminal penceresinde aşağıdaki komut satırları yazılır.:

```
(base) Arat-MacBook-Pro-2:~ mustafamuratarat$ cd Desktop/ExcelStreamlit_WebApp/
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ mkdir .streamlit
```

Daha sonra `.streamlit` dizine gidilir ve `config.toml` dosyası `touch` komutu ile yaratılır:

```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ cd .streamlit/
(base) Arat-MacBook-Pro-2:.streamlit mustafamuratarat$ touch config.toml
```

`config.toml` dosyasının içine, tercih ettiğiniz metin editörü ile yukarıda verilen tema kodu yapıştırılır ve özelleştirdiğiniz temaya göre gerekli düzeltmeler  gerçekleştirilir. Tema içerisinde, birincil renk, arkaplan rengi, ikincil arkaplan rengi gibi çeşitli parametreleri ayarlayabilirsiniz.

## requirements.txt dosyasını oluşturma

Web uygulaması böylelikle sonlandı. Artık, tüm Python bağımlılıklarının (dependencies) olduğu `requirements.txt` dosyasını oluşturalım. Bunun için `pipreqs` kütüphanesini kullanabiliriz. Bu kütüphane yüklü değilse, `pip3 install pipreqs` komutu ile kişisel bilgisayarınıza yükleyebilirsiniz.

Daha sonra bir Terminal penceresinden, `app.py` dosyamızın olduğu klasöre gidelim. Daha sonra sanal ortamımızı aktive edelim. Aktivasyondan sonra, `pipreqs ./` komutu ile kullanılan tüm paketleri `requirements` isimli metin dosyasına yazdıralım:

```
(base) Arat-MacBook-Pro-2:~ mustafamuratarat$ cd Desktop/ExcelStreamlit_WebApp/
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp/ mustafamuratarat$ pipreqs ./
INFO: Successfully saved requirements file in ./requirements.txt
```

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc18.png?raw=true)

`requirements.txt` metin dosyasını oluşturmamızın sebebi, uygulamayı canlı ortama dağıtırken kullanacağımız bulut sunucusunun gerekli gereksinimleri yüklemesi içindir.

Artık uygulamamıza sahip olduğumuza göre, uygulamayı dağıtmaya başlamaya hazırız.

## Heroku aracılığıyla dağıtım

Uygulamalarınızı Heroku'da veya digital ocean, AWS veya Google Cloud gibi herhangi bir özel bulut sistemi dağıtıcısında barındırabilirsiniz. Burada, ücretsiz bir çözüm olduğu için Heroku'da (https://www.heroku.com/) barındırma (hosting) yöntemini kullanabiliriz. Heroku, uygulamaları bulutta çalıştırmanıza izin veren bir Hizmet-olarak-Platform (Platform-as-a-Service - PaaS)'dır.

Heroku'da barındırmak için, `requirements.txt` dosyasına ihtiyacınız olacaktır. Bunun yanı sıra, 2 ek dosyaya ihtiyacınız olacak:

1. `Procfile`: `Procfile`, önce setup.sh dosyasını yürütmek ve ardından uygulamayı çalıştırmak için Streamlit run'ı çağırmak için kullanılır.
  
```
web: sh setup.sh && streamlit run app.py
```
  
  Procfile yazma hakkında daha fazla bilgi için dokümantasyona bakınız: https://devcenter.heroku.com/articles/preparing-a-codebase-for-heroku-deployment

2. `setup.sh`: `setup.sh` dosyasında, içerisinde bir `config.toml` dosyası olan bir `streamlit` gizli klasörü oluşturacağız.

```bash
mkdir -p ~/.streamlit/

echo "[general]
email = \"your-email@domain.com\"
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
enableCORS=false
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

`setup.sh` ve `Procfile` dosyalarını kullanarak Heroku'ya uygulamayı başlatmak için gerekli komutları söyleyebilirsiniz.

Yani aşağıdaki gibi klasör yapısına sahip olmalısınız:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc19.png?raw=true)

Bu iki ekstra dosyayı oluşturduktan sonra yapmamız gereken, aşağıdaki komutları kullanarak `ExcelStreamlit_WebApp` klasöründe bir Git deposu (depository) başlatmak (https://git-scm.com/docs/git-init). Heroku, Git ve Docker dahil olmak üzere birçok farklı teknolojiyi kullanarak dağıtıma (deployment) izin verir. Bu eğiticide Git'i basit olduğu için kullanacağız:

```
(base) Arat-MacBook-Pro-2:~ mustafamuratarat$ cd Desktop/ExcelStreamlit_WebApp/
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ git init
hint: Using 'master' as the name for the initial branch. This default branch name
hint: is subject to change. To configure the initial branch name to use in all
hint: of your new repositories, which will suppress this warning, call:
hint: 
hint: 	git config --global init.defaultBranch <name>
hint: 
hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
hint: 'development'. The just-created branch can be renamed via this command:
hint: 
hint: 	git branch -m <name>
Initialized empty Git repository in /Users/mustafamuratarat/Desktop/ExcelStreamlit_WebApp/.git/
```

Burada bir Heroku hesabı oluşturmanız gerekecek. Genel olarak, Heroku'yu kullanmak ücretsizdir ancak uygulama ölçümleri veya ücretsiz SSL gibi daha fazla özellik elde etmek için ödeme yapmanız gerekir.

Heroku hesabı oluşturduktan sonra https://devcenter.heroku.com/articles/heroku-cli bağlantısına giderek Heroku Komut Satırı Yorumlayıcısını (CLI-Command Line Interpreter) indirmeniz gerekmektedir. Gerekli kurumları tamamladıktan sonra, Terminal penceresine geri dönün ve aşağıdaki komut satırını yazın:

```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ heroku login
```

komutunu yazınız. Bu tarayıcıda yeni bir sekme açacak ve Heroku hesabınıza giriş yapmanızı isteyecek. 

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc20.png?raw=true)

Giriş yaptıktan sonra Terminal pernceresine geri dönün ve aşağıdaki komut satırını yazın:

```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ heroku create worldhappiness-streamlit
Creating ⬢ worldhappiness-streamlit... done
https://worldhappiness-streamlit.herokuapp.com/ | https://git.heroku.com/worldhappiness-streamlit.git
```

Burada web uygulamasının adını `worldhappiness-streamlit` olarak ayarladım. Siz uygulamanızın adını istediğiniz gibi seçebilirsiniz. Komutu çalıştırdıktan sonra uygulamanızın URL'si hazır olacaktır.

Şimdi, uygulama kodumuzu, Heroku bulut sunucusuna Git kullanarak push edebilir ve ardından commit mesajını yazabiliriz:

```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ git add .
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ git commit -m "initial commit"
```

Daha sonra `git push heroku master` komutu ile uygulama dosyalarımızı Heroku'ya push edebiliriz.


```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ git push heroku master
```

`git push heroku master` komutunu çalıştırırken, Heroku'nun bir Python uygulamanız olduğunu otomatik olarak algıladığını ve `requirements.txt` dosyasındaki tümn kütüphaneleri otomatik olarak yüklediğini fark etmelisiniz. Komut çalışmayı bittikten sonra aşağıdakine benzer bir şey görmelisiniz:

```
remote: -----> Launching...
remote:        Released v3
remote:        https://worldhappiness-streamlit.herokuapp.com/ deployed to Heroku
remote: 
remote: Verifying deploy... done.
To https://git.heroku.com/worldhappiness-streamlit.git
 * [new branch]      master -> master
 ```
 
`heroku ps:scale web=1` komutunu kullanarak uygulamanın başarıyla dağıtılıp dağıtılmadığını kontrol edebilirsiniz.

```
(base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ heroku ps:scale web=1
Scaling dynos... done, now running web at 1:Free
```

Son olarak, `heroku open` komutu ile uygulamayı açabilirsiniz. Bu, varsayılan tarayıcınızı kullanarak uygulamayı açacaktır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/streamlitHeroku_images/sc21.png?raw=true)
  
Oluşturduğumuz web uygulamasına [https://worldhappiness-streamlit.herokuapp.com/](https://worldhappiness-streamlit.herokuapp.com/){:target="_blank"} bağlantısından erişilebilir.

Oluşturduğumuz tüm dosyaları [burada](https://github.com/mmuratarat/WorldHappinessApp_Streamlit){:target="_blank"} bulunan Github deposunda bulabilirsiniz.
