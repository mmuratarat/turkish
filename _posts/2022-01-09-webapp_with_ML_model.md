---
layout: post
title:  "Arkaplanında bir Makine Öğrenmesi Modeli bulunan bir web uygulaması oluşturmak"
author: "MMA"
comments: true
---

Verilerle çalışan biriyseniz, en az bir kez bir makine öğrenmesi modeli yaratmış olabilirsiniz. Hepimiz her gün çeşitli makine öğrenmesi modelleriyle karşılaşıyoruz. Her model farklıdır ve farklı amaçlar için kullanılır. İyi bir model oluşturmak için farklı algoritmalar kullanmamız gereken çeşitli problemlerimiz var. Ama modelimizi inşaa ettikten sonra ne yapmalıyız? Çoğu insan model oluşturur, ancak modeli gerçek hayatta uygulamadan pek bir faydası olmayacağı gerçeğini görmezden gelirler. Bu eğiticide, bir model oluşturma ve onu web'de dağıtma sürecinin tamamını gözden geçireceğiz.

Diyelim bir bankada çalışıyorsunuz ve patronunuz sizden kredi uygunluk sürecini otomatikleştiren bir web uygulaması oluşturmanızı istedi. Yapmanız gereken çok basit. Geçmiş verileri alarak bir model inşaa etmek, bu model kaydetmek ve ardından bir web uygulamasına dönüştürmek. 

İlk olarak Masaüstü'nde (Desktop) `LoanPredictionApp` isimli bir boş klasör oluşturalım.

## Sanal Ortam Oluşturmak

Bu eğiticiye başlamadan önce bilgisayarınızdaki düzeni bozmamak adına, aşağıda yapacağımız tüm işlemleri sanal bir ortam içerisinde gerçekleştirelim. Basitçe ifade etmek gerekirse, bir sanal ortam, diğer projeleri etkileme endişesi olmadan belirli bir proje üzerinde çalışmanıza izin veren, Python’un yalıtılmış bir çalışma kopyasıdır. Her proje için birden çok Python versiyonunun aynı makineye kurulumuna olanak tanır. Aslında Python’un ayrı kopyalarını kurmaz, ancak ortam değişkenlerini ve paketleri izole ettiği için farklı proje ortamlarını izole tutmanın akıllıca bir yolunu sağlar. Yanlış paket versiyonlarindan şikayet eden hata mesajlarının bir çaresidir 

Python'da sanal ortam oluşturmak için kullanabileceğiniz bazı popüler kütüphaneler/araçlar şöyledir: `virtualenv`, `virtualenvwrapper`, `pvenv` ve `venv`. Burada `virtualenv` paketine odaklanacağız.

`virtualenv`'in sisteminizde zaten kurulu olması muhtemeldir. Bununla birlikte, bu paketin global sisteminizde kurulu olup olmadığını, eğer kurulu ise, hangi sürümü kullandığınızı kontrol edin:

```
which virtualenv
```

veya

```
virtualenv --version
```

Eğer bu paket kurulu değilse, `virtualenv` paketini `pip3 install virtualenv` komutunu kullanarak yüklemeniz gerekmektedir.

Bunu işlemi gerçekleştirdikten sonra, önce bir Terminal penceresi açın ve oluşturduğunuz `LoanPredictionApp` isimli dizine gidin:

```
(base) Arat-MacBook-Pro-2:~ mustafamuratarat$ cd Desktop/LoanPredictionApp/
```

ve aşağıda verilenleri gerçekleştirin. Aşağıdaki kod, kurduğumuz `virtualenv` modülünü çağıracak ve mevcut klasörümüzün içinde `myEnvironment` adlı yeni bir klasör oluşturacak (yani `myEnvironment` isminde bir sanal ortam) ve `myEnvironment`'te yeni bir Python kurulumu yükleyecek (tabii ki kurmak istediğiniz Python sürümünü değiştirebilirsiniz). Ben burada sistemimde kurulu olan aynı Python 3 sürümünü sanal ortamımda da kullanmak istediğim için `which python3` komutunu kullandım:

```
(base) Arat-MacBook-Pro-2:LoanPredictionApp mustafamuratarat$ python3 -m venv myEnvironment
```

Yukarıdaki komut, projenizde tüm bağımlılıkların (dependencies) kurulu olduğu bir `myEnvironment` dizini oluşturur. 

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc1.png?raw=true)

Kurulum tamamlandıktan sonra, bu sanal ortamı kullanmak istiyorsanız, izole edilmiş ortamınızı `source` komutu ile etkinleştirmeniz gerekir:

```
(base) Arat-MacBook-Pro-2:LoanPredictionApp mustafamuratarat$ source myEnvironment/bin/activate
(myEnvironment) (base) Arat-MacBook-Pro-2:LoanPredictionApp mustafamuratarat$
```

Burada dikkat etmeniz gereken nokta, komut satırının bilgisayarınızın adından, yeni oluşturduğunuz sanal ortamın ismine dönüşmesidir. Proje üzerinde her çalışmak istediğinizde bu sanal ortamı aktive etmelisiniz. Bu nedenle `source  myEnvironment/bin/activate` kodunu çalıştırmayı unutmayın. Ancak, her seferinde bu kodu yazmak istemediğinizde ve bilgisayarınız başlar başlamaz, sanal ortamınızın aktive olmasını istiyorsanız [şu sayfada](https://askubuntu.com/a/1175106/1187527){:target="_blank"} bulunan adımları takip ederek bir betik yazabilirsiniz.

Artık bu ortamda istediğiniz paketleri ve bu paketlerin versiyonlarını global sisteminizi etkilemeden kurabilirsiniz.

## Jupyter not defterine sanal ortama ait çekirdek (kernel) eklemek

Aşağıdaki eğitici JupyterLab üzerinden gerçekleştirilecektir. Ancak buradaki problem ise, kişisel bilgisayarınızdaki JupyterLab’ın global sisteminizde bulunan Python sürümü için çekirdeğinin (kernel) bulunmasıdır. Bu nedenle, sanal ortamımıza yüklediğimiz Python için çekirdek elle oluşturmamız için bazı işlemler daha yapmamız gerekmektedir. Bunun için ilk olarak Jupyter için IPython çekirdeğini sağlayan ipykernel‘i sanal ortamımızda kurmamız gerekmektedir:

```
(myEnvironment) (base) Arat-MacBook-Pro-2:LoanPredictionApp mustafamuratarat$ pip3 install ipykernel
Collecting ipykernel
  Using cached ipykernel-6.6.1-py3-none-any.whl (126 kB)
Collecting nest-asyncio
  Using cached nest_asyncio-1.5.4-py3-none-any.whl (5.1 kB)
Collecting matplotlib-inline<0.2.0,>=0.1.0
  Using cached matplotlib_inline-0.1.3-py3-none-any.whl (8.2 kB)
Collecting appnope
  Using cached appnope-0.1.2-py2.py3-none-any.whl (4.3 kB)
Collecting ipython>=7.23.1
  Using cached ipython-7.31.0-py3-none-any.whl (792 kB)
Collecting debugpy<2.0,>=1.0.0
  Using cached debugpy-1.5.1-cp38-cp38-macosx_10_15_x86_64.whl (1.7 MB)
Collecting jupyter-client<8.0
  Using cached jupyter_client-7.1.0-py3-none-any.whl (129 kB)
Collecting traitlets<6.0,>=5.1.0
  Using cached traitlets-5.1.1-py3-none-any.whl (102 kB)
Collecting tornado<7.0,>=4.2
  Using cached tornado-6.1-cp38-cp38-macosx_10_9_x86_64.whl (416 kB)
Collecting jedi>=0.16
  Using cached jedi-0.18.1-py2.py3-none-any.whl (1.6 MB)
Collecting prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0
  Using cached prompt_toolkit-3.0.24-py3-none-any.whl (374 kB)
Collecting pickleshare
  Using cached pickleshare-0.7.5-py2.py3-none-any.whl (6.9 kB)
Collecting pygments
  Using cached Pygments-2.11.2-py3-none-any.whl (1.1 MB)
Requirement already satisfied: setuptools>=18.5 in ./myEnvironment/lib/python3.8/site-packages (from ipython>=7.23.1->ipykernel) (56.0.0)
Collecting backcall
  Using cached backcall-0.2.0-py2.py3-none-any.whl (11 kB)
Collecting decorator
  Using cached decorator-5.1.1-py3-none-any.whl (9.1 kB)
Collecting pexpect>4.3
  Using cached pexpect-4.8.0-py2.py3-none-any.whl (59 kB)
Collecting parso<0.9.0,>=0.8.0
  Using cached parso-0.8.3-py2.py3-none-any.whl (100 kB)
Collecting python-dateutil>=2.1
  Using cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)
Collecting entrypoints
  Using cached entrypoints-0.3-py2.py3-none-any.whl (11 kB)
Collecting pyzmq>=13
  Using cached pyzmq-22.3.0-cp38-cp38-macosx_10_9_x86_64.whl (1.3 MB)
Collecting jupyter-core>=4.6.0
  Using cached jupyter_core-4.9.1-py3-none-any.whl (86 kB)
Collecting ptyprocess>=0.5
  Using cached ptyprocess-0.7.0-py2.py3-none-any.whl (13 kB)
Collecting wcwidth
  Using cached wcwidth-0.2.5-py2.py3-none-any.whl (30 kB)
Collecting six>=1.5
  Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
Installing collected packages: wcwidth, traitlets, six, ptyprocess, parso, tornado, pyzmq, python-dateutil, pygments, prompt-toolkit, pickleshare, pexpect, nest-asyncio, matplotlib-inline, jupyter-core, jedi, entrypoints, decorator, backcall, appnope, jupyter-client, ipython, debugpy, ipykernel
Successfully installed appnope-0.1.2 backcall-0.2.0 debugpy-1.5.1 decorator-5.1.1 entrypoints-0.3 ipykernel-6.6.1 ipython-7.31.0 jedi-0.18.1 jupyter-client-7.1.0 jupyter-core-4.9.1 matplotlib-inline-0.1.3 nest-asyncio-1.5.4 parso-0.8.3 pexpect-4.8.0 pickleshare-0.7.5 prompt-toolkit-3.0.24 ptyprocess-0.7.0 pygments-2.11.2 python-dateutil-2.8.2 pyzmq-22.3.0 six-1.16.0 tornado-6.1 traitlets-5.1.1 wcwidth-0.2.5
WARNING: You are using pip version 21.1.1; however, version 21.3.1 is available.
You should consider upgrading via the '/Users/mustafamuratarat/Desktop/LoanPredictionApp/myEnvironment/bin/python3 -m pip install --upgrade pip' command.
```

Ardından, sanal ortamınız için oluşturduğunuz Python çekirdeğini aşağıdaki komutu kullanarak Jupyter’e ekleyebilirsiniz:

```
(myEnvironment) (base) Arat-MacBook-Pro-2:LoanPredictionApp mustafamuratarat$ python3 -m ipykernel install --user --name=myEnvironment
Installed kernelspec myEnvironment in /Users/mustafamuratarat/Library/Jupyter/kernels/myenvironment
```

`/Users/mustafamuratarat/Library/Jupyter/kernels/myenvironment` isimli klasörün içinde, her şeyi doğru yaptıysanız aşağıdaki şekilde görünmesi gereken bir `kernel.json` dosyası bulacaksınız:

```
(base) Arat-MacBook-Pro-2:~ mustafamuratarat$ cd /Users/mustafamuratarat/Library/Jupyter/kernels/myenvironment
(base) Arat-MacBook-Pro-2:myenvironment mustafamuratarat$ ls
kernel.json	logo-32x32.png	logo-64x64.png
(base) Arat-MacBook-Pro-2:myenvironment mustafamuratarat$ cat kernel.json
{
 "argv": [
  "/Users/mustafamuratarat/Desktop/LoanPredictionApp/myEnvironment/bin/python3",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "myEnvironment",
 "language": "python",
 "metadata": {
  "debugger": true
 }
```

Hepsi bu kadar! Artık `myenvironment` isiml, sanal ortamını Jupyter’de çekirdek olarak seçebilirsiniz. JupyterLab’de bunun nasıl görüneceği aşağıda görülmektedir:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc2.png?raw=true)

Bu butona tıklayarak, `myenvironment` çekirdeğine sahip Jupyter not defteri üzerinde model geliştirmeye başlayabilirsiniz.

Yeni bir Jupyter not defteri başlattığınızda, sağ üst köşedeki daire içerisinde çekirdeğinizin ismi yazmalıdır:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc3.png?raw=true)

## Web uygulaması için gerekli kütüphaneleri sanal ortama kurmak

Web uygulaması için kullanacağımız paketler `numpy`, `pandas`, `sklearn` ve `streamlit`'tir. Bu kütüphaneleri sanal ortamımıza yüklememiz gerekmektedir. Bunun için pip Python paket yöneticisini kullanabilirsiniz (Anaconda kullanıyorsanız, gerekli komutları tercih ediniz.)

```
(myEnvironment) (base) Arat-MacBook-Pro-2:ExcelStreamlit_WebApp mustafamuratarat$ pip3 install numpy pandas sklearn streamlit
```

Bu komut satırı çalıştıktan sonra gerekli kütüphaneler sanal ortamınızda kullanılmaya hazır olacaktır. 

## Veri Kümesi

Model oluşturabilmek için bir veri setine ihtiyacımız var. Bu eğitici için kredi uygunluk sürecini otomatikleştirmek istediğimizden, aşağıdaki Kaggle bağlantısında bulunan Loan Prediction (Kredi Tahmin) veri kümesi kullanılacaktır.
https://www.kaggle.com/ninzaami/loan-predication/home

Kaggle'a üye olup, `train_u6lujuX_CVtuZ9i (1).csv` isimli veri dosyasını indiriniz. Ardından, `LoanPredictionApp` klasörünün altında `data` isimli yeni bir klasör yaratalım. İndirdiğimiz csv dosyasını, `data` klasörünün içerisine taşıyınız. `train_u6lujuX_CVtuZ9i (1).csv` dosyasının ismini `full_dataset.csv` olarak değiştiriniz.

`full_dataset.csv` dosyasında 12 tane sütun vardır:

|      Değişken     |                       Tanımı                       | Veri Tipi |                       Aldığı değerler                       |
|:-----------------:|:--------------------------------------------------:|:---------:|:-----------------------------------------------------------:|
|      Loan_ID      |                Kredi Banka Numarası                |           |                                                             |
|       Gender      |             Başvuru sahibinin cinsiyeti            | Kategorik |                Male (Erkek) / Female (Kadın)                |
|      Married      |               Başvuru sahibi evli mi?              | Kategorik |                   Yes (Evet) / No (Hayır)                   |
|     Dependents    |         Bakmakla yükümlü olunan kişi sayısı        | Kategorik |                        0 / 1 / 2 / 3+                       |
|     Education     |          Başvuru sahibinin eğitim seviyesi         | Kategorik |       (Graduate (Lisansüstü) / Under Graduate (Lisans)      |
|   Self_Employed   |       Başvuru Sahibi kendi işini mi yapıyor?       | Kategorik |                  Yes (Evet) / No (Hayır)                    |
|  ApplicantIncome  |              Başvuru sahininin geliri              |  Nümerik  |                                                             |
| CoapplicantIncome | Başvuru sahibiyle birlikte başvuran kişinin geliri |  Nümerik  |                                                             |
|     LoanAmount    |            Verilen Kredinin tutarı (bin)           |  Nümerik  |                                                             |
|  Loan_Amount_Term |              Ay cinsinden kredi vadesi             |  Nümerik  |                                                             |
|   Credit_History  |                    Kredi geçmişi                   |  Nümerik  |                          1.0 / 0.0                          |
|   Property_Area   |        Kredi istenilen mülk alanı nerededir?       | Kategorik | Semiurban (Yarı Kentsel) / Urban (Kentsel) / Rural (Kırsal) |
|    Loan_Status    |                 Kredi onaylandı mı?                | Kategorik |                     Y (Evet) / N (Hayır)                    |

Artık modelleme aşamasına geçmeye hazırız.

İlk olarak, `full_dataset.csv` veri kümesi $\%80 - \%20$ oranı kullanılarak iki kısıma ayrılacaktır. $\%80$'lik kısıma eğitim veri kümesi (training dataset) denilecek ve bu küme model eğitimi (model training) için, $\%20$'lik kısıma ise test veri kümesi (testing dataset) denilecek ve bu küme model değerlendirme (model evaluation) aşamasında kullanılacaktır. Test veri kümesine modelleme boyunca kesinlikle dokunulmayacaktır ve bu veri kümesi modelimizin görmediği verileri (unseen data) içermelidir. Model doğrulaması (model validation), hiperparametre seçimi (hyperparameter selection) ve çapraz doğrulama (cross validation) adımları `train_ctrUa4K.csv` veri kümesinden ayrılan $\%80$'lik kısım kullanılarak gerçekleştirilecektir. 

## Model Oluşturma

İlk olarak, model oluştururken kullanılacak tüm kütüphaneleri tek bir Jupyter not defteri hücresinde içe aktaralım:

```python
import pandas as pd
import numpy as np
import pickle 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
```

Daha sonra, `pandas` kütüphanesini kullanarak verilerimizi inceleyelim. Burada, `read_csv` fonksiyonu, verileri Python ortamına okutmak için yeterlidir. 

```python
# Tam Veri Kümesi (Full Dataset)
full_set = pd.read_csv('./data/full_dataset.csv')
full_set.head(5)
```

Veri kümesindeki değişkenlerin veri tiplerine (data types) bakalım:

```python
full_set.dtypes

# Gender                object
# Married               object
# Dependents            object
# Education             object
# Self_Employed         object
# ApplicantIncome        int64
# CoapplicantIncome    float64
# LoanAmount           float64
# Loan_Amount_Term     float64
# Credit_History       float64
# Property_Area         object
# Loan_Status           object
# dtype: object
```

Veri kümesi hem nicel (nümerik, sayısal) hem de kategorik (nitel) değişkenlerden oluşmaktadır.

Tam veri kümesinin (`full_dataset.csv`) şekline bakalım:

```python
print(full_set.shape)
# (614, 13)
```

Etiketli (labelled) veri kümesinde 614 gözlem (observation) ve 13 sütun vardır. Burada `Loan_ID` sütunu modelimiz için herhangi bir bilgi taşımamaktadır, bu  nedenle, bu sütunu elimine edebiliriz. `Loan_Status` değişkeni yapacağımız analiz için bağımlı değişkendir ve 2 sınıftan oluşmaktadır. Elimizdeki veri kümesi tarihi bir veridir (historical data). Geçmişte, müşterilerin kredi başvurusunun onaylanıp onaylanmadığı bilgisini taşımaktadır. Burada iki sınıflı sınıflandırma problemi ile karşı karşıyayız. Müşterilerin (yani gözlemlerin veya deneklerin) bilgisi ise `Gender`, `Married`, `Dependents`, `Education`, `Self_Emplyed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History` ve `Property_Area` değişkenleri ile verilmektedir.

İlk olarak bu tam veri kümesinden, analizimizde kullanmayacağımız `Loan_ID` sütununu düşürelim:  

```python
full_set.drop(['Loan_ID'], axis=1, inplace=True)
full_set.head()
```

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc4.png?raw=true)

Bağımlı değişken (dependent variable) olan `Loan_Status` değişkeni `Y` (Yes) ve `N` (No) olmak üzere iki kategoriden oluşmaktadır. Bu değişkeni nümerikleştirelim. `Y` kategorisi için `1.0`; `N` kategorisi için `0.0` eşleşmesi gerçekleştirelim:

```python
full_set['Loan_Status'] = full_set['Loan_Status'].map({'N':0,'Y':1})
full_set.head()
```

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc5.png?raw=true)

Ayrıca, elimizdeki tam veri kümesinde kayıp gözlem (missing value) olup olmadığını kontrol edelim:

```python
full_set.isnull().sum()

# Gender               13
# Married               3
# Dependents           15
# Education             0
# Self_Employed        32
# ApplicantIncome       0
# CoapplicantIncome     0
# LoanAmount           22
# Loan_Amount_Term     14
# Credit_History       50
# Property_Area         0
# Loan_Status           0
# dtype: int64
```

Görüldüğü üzere bir kaç değişkende kayıp gözlem vardır. Ancak, kayıp gözlem imputasyonunu (missing value imputation) tam veri kümesine uygulamak veri sızıntısına (data leakage) neden olacaktır. Bu nedenle, öncelikle tam veri kümesini, $\%80 - \%20$ oranı kullanılarak iki kısıma ayıralım. $\%80$'lik kısım eğitim veri kümesi (training dataset), $\%20$'lik kısıma test veri kümesi (testing dataset) olacaktır. İşte şimdi doğru bir şekilde eğitim veri kümesindeki parametreleri kullanarak test veri kümesi üzerinde veri imputasyonu gerçekleştirebiliriz.

Veri kümesine ayırma işlemi Scikit-Learn kütüphanesindeki `train_test_split` fonksiyonu kullanılarak yapılır. Ancak, bu fonksiyon, bağımlı değişken(ler) ile bağımsız değişken(ler)i ayrı ayrı kabul etmektedir. Bu nedenle, bağımlı değişken olan `Loan_Status` değişkenini başka bir değişkene atılalım:

```python
X_full = full_set.drop(["Loan_Status"], axis=1)
y_full = full_set["Loan_Status"]
```

Burada, `X_full` bağımsız değişkenlerin (özniteliklerin - features) olduğu bir matristir.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc6.png?raw=true)

Artık, tam veri kümemizi parçalayabiliriz.

```python
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size  = 0.2, random_state=42)

print(f"Eğitim Kümesinin şekli: {X_train.shape}, \nTest Kümesinin Şekli: {X_test.shape}")

# Eğitim Kümesinin şekli: (491, 11), 
# Test Kümesinin Şekli: (123, 11)
```

Kolaylıkla anlaşılabileceği üzere, eğitim veri kümesinde 491 müşteriye, test veri kümesinde ise 123 müşteriye ait bilgi bulunmaktadır. Bağımsız değişkenlerin sayısı ise 11'dir.

Burada eğitim veri kümesi çok büyük olmadığı için model doğrulama (model validation), çapraz doğrulama (cross validation) kullanılarak elde edilecektir. Çapraz doğrulama gerçekleştirilirken de, kullanacağımız makine öğrenmesi modelinin hiperparametrelerine Izgara Arama (Grid Search) kullanılarak ince ayar (fine tuning) verilecektir.

Ancak, veri kümesinde hem sayısal (nümerik) hem de kategorik (categorical) değişkenler vardır. Buna ek olarak, eğitim kümesinde kayıp gözlemler de mevcuttur. Bu farklı değişkenlere ayrı ayrı veri önişleme (data pre-processing) gerçekleştirilecektir. Örneğin, nümerik değişkenler ölçeklendirilecektir (scaling) ve kategorik değişkenler bire-bir-kodlama (one-hot-encoding) kullanılarak nümerikleştirilecektir.

Fakat, herhangi bir veri ön-işleme gerçekleştirirken, veri sızıntısını (data leakage) engellemek için , bu ön-işleme adımını, çapraz doğrulamanın her parçasında ayrı ayrı tekrarlamanız gerekmektedir. Bu adımlar gerek veri ölçekleme (data scaling) olabilir, gerekse sentetik veri oluşturma (Synthetic Data Generation - SMOTE, ADASYN, Oversampling, Undersampling) olabilir, veya kayıp değer tamamlama (missing value imputation) ya da kategorik değişken kodlama (categorical variable encoding) olabilir. 

Verideki nümerik (sayısal) ve kategorik değişkenleri ayırdıktan sonra, bu değişkenlerin türüne göre farklı veri önişleme adımlarını Scikit-Learn kütüphanesindeki `Pipeline` fonksiyonunu kullanarak bir iletim hattı üzerine oturtabilir ve daha sonra gene Scikit-Learn kütüphanesinde bulunan `ColumnTransformer` fonksiyonu yardımı ile bu dönüşleri ayrı ayrı uygulayabilirsiniz.

Daha sonra başka bir Pipeline (İletim Hattı) oluşturarak, önce bir önişleme gerçekleştirir ve daha sonra #GridSearchCV fonksiyonu ile istediğiniz makine öğrenmesi modeli için çapraz doğrulama ile Izgara Arama (Grid Search) kullanarak hiperparametre araştırması gerçekleyebilirsiniz. İşte, arka planda k-parça çapraz doğrulama (k-fold cross validation) yapılırken, nümerik ve kategorik değişkenler için ayrı ayrı tanımladığınız tüm önişleme adımlarının parametreleri k-1 parçadaki verilerden elde edilecek ve daha sonra bu parametreler k-ıncı parçada uygulanacaktır ve bu, k defa aynı şekilde tekrarlanacaktır.

Öncelikle veri kümemizdeki nümerik ve kategorik değişkenlerin isimlerini otomatik olarak çekelim:

```python
num_features = full_set.drop(['Loan_Status'], axis = 1).select_dtypes(include = 'number').columns
num_features
# Index(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History'],
#       dtype='object')
```

`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, ve `Credit_History` değişkenleri nümerik veri yapısına sahiptir.

```python
cat_features = full_set.select_dtypes(include = 'object').columns
cat_features
# Index(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
#        'Property_Area'],
#       dtype='object')
```

Benzer şekilde, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, ve `Property_Area` değişkenleri kategorik veri yapısına sahiptir.

Yukarıda da bahsedildiği gibi, nümerik ve kategorik değişkenlere ayrı ayrı veri ön-işleme adımları gerçekleştirilecektir. Nümerik değişkenler aynı birime sahip olmaları için Minimum-Maximum Ölçekleme (Mix-Max Scaling) yöntemi ile ölçeklenecektir ve varsa kayıp gözlemler ortalama (mean) ile impute edilecektir. Benzer şekilde, kategorik değişkenler, makine öğrenmesi modelinin anlayabileceği şekilde bire-bir-kodlama (one-hot-encoding) yöntemi kullanılarak nümerikleştirilecektir ve bu değişkenlerde kayıp gözlem varsa, bu kayıp gözlem sabit bir değer ile (örneğin, `missing` yani bilinmiyor bilgisi) ile impute edilecektir. Bu iki farklı değişken türü için gerçekleştirilecek işlemler, iki farklı iletim hattı (pipeline) ile gerçekleştirilecektir.

```python
# Sayısal değişkenlere uygulanacak veri ön-işleme adımları için iletim hattı (pipeline)

num_transformer = Pipeline(steps = [('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                                    ('MinMaxScaler', MinMaxScaler())])
# Pipeline(steps=[('Imputer', SimpleImputer()), ('MinMaxScaler', MinMaxScaler())])

# Kategorik değişkenlere uygulanacak veri ön-işleme adımları için iletim hattı (pipeline)

cat_transformer = Pipeline(steps = [('Imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
                                    ('OneHotEncoder', OneHotEncoder(categories = 'auto', drop=None, handle_unknown = 'ignore'))])

# Pipeline(steps=[('Imputer',
#                  SimpleImputer(fill_value='missing', strategy='constant')),
#                 ('OneHotEncoder',
#                  OneHotEncoder(drop='None', handle_unknown='ignore'))])
```

İletim hatlarını tanımladıktan sonra artık sütun dönüşümlerine (column transformers) geçebiliriz. 

```python
preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_features), 
                                                 ('cat', cat_transformer, cat_features)],
                                 remainder = 'drop',
                                 n_jobs = -1,
                                 verbose = False)

# ColumnTransformer(n_jobs=-1,
#                   transformers=[('num',
#                                  Pipeline(steps=[('Imputer', SimpleImputer()),
#                                                  ('MinMaxScaler',
#                                                   MinMaxScaler())]),
#                                  Index(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History'],
#       dtype='object')),
#                                 ('cat',
#                                  Pipeline(steps=[('Imputer',
#                                                   SimpleImputer(fill_value='missing',
#                                                                 strategy='constant')),
#                                                  ('OneHotEncoder',
#                                                   OneHotEncoder(drop='None',
#                                                                 handle_unknown='ignore'))]),
#                                  Index(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
#        'Property_Area'],
#       dtype='object'))])
```

Artık veri ön-işleme hattımız hazır!

Şimdi, bu veri önişleme adımı ile, oluşturacağımız Rastgele Ormanlar (random forest) algoritmasını bir araya getirebilir, kolaylıkla çapraz doğrulama uygulayabiliriz. Böylelikle, yukarıda bahsedilen veri sızıntısından da kaçınırız. 

```python
pipe = Pipeline(steps = [('preprocess', preprocessor),
                         ('RF_model', RandomForestClassifier(class_weight = "balanced", n_jobs=-1))],
                verbose=False)         
```

Izgara arama gerçekleştirirken kullanacağımız parametreler için olası değerleri içeren bir sözlük hazırlayalım:

```python
parameters_grid = [{'RF_model__n_estimators':[10, 20, 50],
                    'RF_model__max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75],
                    'RF_model__max_depth' : [2, 4, 5, 6, 7, 8]}
                  ]
```

Gerekli tanımlamaları yaptıktan sonra artık çapraz doğrulamalı ızgara aramayı ayarlayabiliriz. Burada 10-katlı çapraz doğrulama (10-fold cross validation) kullanacağız ve performans ölçütü olarak doğruluk oranını (accuracy rate) tercih ediyoruz. 

```python
search = GridSearchCV(estimator = pipe, param_grid = parameters_grid, cv = 10, scoring = 'accuracy', return_train_score=False, verbose=1, n_jobs=-1)
```

Artık elimizdeki eğitim verisini modellerimize uydurmaya (fitting) başlayabiliriz:

```python
best_model = search.fit(X_train, y_train)
```

Tam tamına 90 farklı moel denenecektir. Bu modellerden en iyisi `max_depth=2`, `max_features=0.75`, ve `n_estimators=20` hiperparametrelerine sahip bir Rastgele Ağaç modelidir.

Bazı değerlendirme ölçütleri ile bu modelin eğitim kümesi ve test kümesi üzerinde performanslarını kontrol edebiliriz. Oluşturabileceğimiz fonksiyon şu şekilde olabilir:

```python
def TrainTestScores(y_train, y_train_pred, y_test, y_test_pred):
    
    scores = {"train_set": {"Accuracy" : accuracy_score(y_train, y_train_pred),
                            "Precision" : precision_score(y_train, y_train_pred),
                            "Recall" : recall_score(y_train, y_train_pred),                          
                            "F1 Score" : f1_score(y_train, y_train_pred),
                           "AUC": roc_auc_score(y_train, y_train_pred)},
    
              "test_set": {"Accuracy" : accuracy_score(y_test, y_test_pred),
                           "Precision" : precision_score(y_test, y_test_pred),
                           "Recall" : recall_score(y_test, y_test_pred),                          
                           "F1 Score" : f1_score(y_test, y_test_pred),
                          "AUC:": roc_auc_score(y_test, y_test_pred)}}
    
    return scores
```

`predict` metodu kullanarak tahminler kolaylıkla elde edilebilir:

```python
ytrain_pred = best_model.predict(X_train)
ytest_pred = best_model.predict(X_test)
```

Her iki küme için değerlendirme ölçütleri (evaluation metrics) şu şekildedir:

```python
TrainTestScores(y_train, ytrain_pred , y_test, ytest_pred)
# {'train_set': {'Accuracy': 0.8167006109979633,
#   'Precision': 0.8,
#   'Recall': 0.9824561403508771,
#   'F1 Score': 0.8818897637795275,
#   'AUC': 0.709348875544566},
#  'test_set': {'Accuracy': 0.7886178861788617,
#   'Precision': 0.7596153846153846,
#   'Recall': 0.9875,
#   'F1 Score': 0.8586956521739131,
#   'AUC:': 0.7030523255813954}}
```

Burada amaç arkasında makine öğrenme modeli bulunan bir web uygulaması geliştirmek olduğu için bu modeli uygun kabul edelim ve pickle nesnesi olarak kaydedelim:

```python
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(best_model, pickle_out) 
pickle_out.close()
```

Artık dizinimizde `classifier.pkl` modeli olacaktır.

Bu modeli istediğimiz zaman, aşağıdaki iki kod satırı kullanarak geri yükleyebiliriz:

```python
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
```

Web uygulaması oluştururken de yapacağımız şey de budur!

## Web Uygulaması

Modelimizi elde edip, pickle nesnesi olarak kaydettikten sonra, web uygulaması geliştirmeye geçebiliriz. Bu web uygulamasının kodunu Streamlit kütüphanesi kullanarak yazacağız.

Streamlit, makine öğrenmesi ve veri bilimi ekipleri tarafından model dağıtımı için kullanılan popüler bir açık kaynaklı yazılım çerçeves,dir. Ve en iyi yanı, ücretsiz ve tamamen Python tabanlı olmasıdır

Burada JupyterLab yerine Visual Studio Code kullanacağız. Siz istediğiniz Entegre Geliştirme Ortamını (Integrated Development Environment - IDE) tercih edebilirsiniz.

Visual Studio Code’u çalıştırdıktan sonra `LoanPredictionApp` klasörünü pencereye sürükleyiniz. Burada önemli olan Visual Studio Code’un oluşturduğunuz sanal ortamı tanımasıdır.

İlk olarak `app` isimli ve `.py` uzantılı boş bir Python yürütülebilir dosyası (Python executable file) yaratalım:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc7.png?raw=true)

Tüm gerekli Python kütüphanelerini içe aktaralım:

```python
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
```

İlk adım olarak, `streamlit` kütüphanesindeki `set_page_config` fonksiyonunu kullanarak web uygulamasının bazı yapılandırmasını ayarlayacağız. Oluşturacağınız web uygulamasının başlığını ve kullanacağınız favicon'u bu şekilde yerleştirebilirsiniz:

```python
st.set_page_config(page_title="Kredi Uygunluğu", page_icon=":bank:", layout="wide")
```

Daha sonra, web uygulamasını tanıtan bir kaç metin ekleyelim:

```python
st.markdown("<h1 style='text-align: center; font-size: 40px;'>Arat Banka Hoşgeldiniz!</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 20px;'>Aşağıda verilen gerekli bilgileri girerek müşterinin kredi uygunluğuna karar verebilirsiniz.</h1>", unsafe_allow_html=True)
st.markdown("##")
```

Ardından, pickle nesnesi olarak kaydettiğimiz modelimizi, `pickle` kütüphanesindeki `load` fonksiyonunu kullanarak yükleyelim. Burada önemli olan, `app.py` dosyası ile `classifier.pkl` dosyasının aynı dizin içinde yer alması. 

```python
# Eğitilmiş modeli yüklemek
pickle_in = open('classifier.pkl', 'rb') 
model = pickle.load(pickle_in)
```

Daha sonra, kullanıcıdan elimizdeki değişkenlere göre girdi talep ederiz. Nümerik değişkenler için bir kaydırıcı (`slider`), kategorik değişkenler için bir seçim kutusu (`selectbox`) kullanabiliriz:

```python
# Kullanıcı girdisi
Gender_input = st.selectbox(label = 'Başvuru sahibinin cinsiyeti nedir?', options = ("Erkek", "Kadın"))
Married_input = st.selectbox(label = 'Başvuru sahibi medeni hali', options = ("Evli", "Bekar"))
Dependents_input = st.selectbox(label = 'Başvuru sahibinin bakmakla yükümlü olduğu kişi sayısı kaçtır?', options = ("0"," 1", "2", "3+"))
Education_input = st.selectbox(label = 'Başvuru sahibinin eğitim seviyesi nedir?', options = ("Lisansüstü", "Lisans"))
Self_Employed_input = st.selectbox(label = 'Başvuru sahibi kendi işinin sahibi midir?', options = ("Evet", "Hayır"))
ApplicantIncome_input = st.slider(label = 'Başvuru sahibinin geliri ne kadardır?', min_value = 0, max_value = 100000)
CoapplicantIncome_input = st.slider(label = 'Başvuru sahibiyle birlikte başvuran kişinin geliri nedir?', min_value = 0.0, max_value = 100000.0)
LoanAmount_input = st.slider(label = 'İstenilen kredinin tutarı ne kadardır?', min_value = 0.0, max_value = 100000.0)
Loan_Amount_Term_input = st.slider(label = 'İstenilen kredinin vadesi ay cinsinden ne kadardır?', min_value = 0.0, max_value = 480.0, step=1.0)
Credit_History_input = st.selectbox(label = 'Başvuru sahibinin kredi geçmişi var mı?', options = (1.0, 0.0))
Property_Area_input = st.selectbox(label = 'Kredi istenilen mülk alanı nerededir?', options = ("Yarı Kentsel", "Kentsel", "Kırsal"))
```

Kullanıcı girdilerini aldıktan sonra, bu girdileri bir sözlüğe (dictionary) taşıyalım ve bu sözlüğü web uygulamamız üzerinde bir `DataFrame` olarak yazdıralım. Bu `DataFrame` bize başvuru sahibi ile ilgili özet bilgileri göstersin:

```python
st.markdown("<h1 style='text-align: center; font-size: 40px;'>Başvuru Sahibinin Özet Bilgileri:</h1>", unsafe_allow_html=True)
summary_dictionary = {'Cinsiyeti': Gender_input,  'Medeni Hali': Married_input, 'Bağımlı Sayısı': Dependents_input, 'Eğitim': Education_input,  'Kendi İşi': Self_Employed_input,
'Geliri': ApplicantIncome_input, 'Birlikte Başvuranın Geliri': CoapplicantIncome_input, 'Kredi Miktarı': LoanAmount_input, 'Kredi Vadesi': Loan_Amount_Term_input,
'Kredi Geçmişi': Credit_History_input, 'Mülk Alanı': Property_Area_input}

summary_df  = pd.DataFrame([summary_dictionary])
st.table(summary_df)
```

Kullanıcının girdiği verileri kullanarak tahmin yapacak bir fonksiyonu tanımlamamız gerekmektedir. Bu fonksiyona `predict_` adını verelim. Burada, aynı zamanda, Scikit-Learn kütüphanesi kullanarak oluşturduğumuz, pickle nesnesi olarak kaydettiğimiz ve `app.py` dosyası içerisinde geri yüklediğimiz makine öğrenmesi modelimizin `predict` metodunu kullanmalıyız.

Ayrıca, kullanıcıdan bazı girdileri Türkçe almamızdan dolayı, bu Türkçe girdileri, elimizdeki verilere göre uyarlamamız gerekmektedir. 

```python
def predict_(model, Gender_input, Married_input, Dependents_input, Education_input, Self_Employed_input, ApplicantIncome_input, CoapplicantIncome_input, LoanAmount_input,
 Loan_Amount_Term_input, Credit_History_input, Property_Area_input):
    
    # kullanıcı girdisini ön işleme
    if Gender_input == "Erkek":
        Gender_var = "Male"
    else:
        Gender_var = "Female"
    
    if Married_input == "Evli":
        Married_var = "Yes"
    else:
        Married_var = "No"

    if Education_input == "Lisansüstü":
        Education_var = "Graduate"
    else:
        Education_var = "Not Graduate"

    if Self_Employed_input == "Evet":
        Self_Employed_var = "Yes"
    else:
        Self_Employed_var = "No"
    
    if Property_Area_input == "Yarı Kentsel":
        Property_Area_var = "Semiurban"
    elif Property_Area_input == "Kentsel":
        Property_Area_var = "Urban"
    else:
        Property_Area_var = "Rural"

    features = {'Gender': Gender_var,  'Married': Married_var, 'Dependents': Dependents_input, 'Education': Education_var,  'Self_Employed': Self_Employed_input, 
    'ApplicantIncome': ApplicantIncome_input, 'CoapplicantIncome': CoapplicantIncome_input, 'LoanAmount': LoanAmount_input, 'Loan_Amount_Term': Loan_Amount_Term_input,
    'Credit_History': Credit_History_input, 'Property_Area': Property_Area_var}

    features_df  = pd.DataFrame([features])
    
    prediction_ = model.predict(features_df)

    if prediction_ == 0:
        pred = 'red edildi.'
    else:
        pred = 'onaylandı.'
    
    return pred
```

Daha sonra bu `predict_` fonksiyonunu kullanarak, kullanıcı girdileriyle birlikte canlı tahminleri (live predictions) elde ederiz ve gerekli şekilde ekrana yazdırırız:

```python
# Tahmin butonuna tıklandığında, kaydedilmiş modelden tahmin elde et ve yazdır
st.markdown("---")

st.markdown("<h1 style='text-align: left; font-size: 20px;'>Girilen bilgilere göre başvuru sahibine kredi verilip verilmemesini öğrenmek için aşağıdaki butona tıklayınız:</h1>", unsafe_allow_html=True)

if st.button('Kredi verilsin mi?'):
    
    result_ = predict_(model, Gender_input, Married_input, Dependents_input, Education_input, Self_Employed_input, ApplicantIncome_input, CoapplicantIncome_input, LoanAmount_input, Loan_Amount_Term_input, Credit_History_input, Property_Area_input)

    if result_ == 'red edildi.':
        st.error('Krediniz {}'.format(result_))
    else:
        st.success('Krediniz {}'.format(result_))
        st.write(f'İstenilen kredi miktarı {LoanAmount_input * 1000: ,.2f} Türk lirasıdır.')
``` 

Uygulama kodunun yazılması böylelikle sona ermiştir. Terminal penceresi üzerinde `streamlit run app.py` komutu çalıştırıldığında, web uygulaması http://localhost:8501 lokal URL’inde web tarayıcınızda açılacaktır.

Değişkenlere birkaç değer verelim ve bir kaç sonuç görelim.

Kredi kabul edildiğinde yeşil arkaplanıyla "Krediniz kabul edildi." ibaresi belirecektir. Bunu `st.success` fonksiyonuyla gerçekleştirebiliriz. Ardından ne kadar kredi verileceği bilgisi yazdırılacaktır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc8.png?raw=true)

Kredi red edildiğinde ise kırmızı arka planla "Krediniz red edildi." ibaresi ekrana yazdırılacaktır.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc9.png?raw=true)

## requirements.txt dosyasını oluşturma

Tüm Python bağımlılıklarının (dependencies) olduğu `requirements.txt` dosyasını oluşturmak için `pipreqs` kütüphanesini kullanabiliriz. Bu kütüphane yüklü değilse, `pip3 install pipreqs` komutu ile kişisel bilgisayarınıza yükleyebilirsiniz.

Daha sonra bir Terminal penceresinden, app.py dosyamızın olduğu klasöre gidelim ve `pipreqs ./` komutu ile kullanılan tüm paketleri requirements isimli metin dosyasına yazdıralım:

```
(base) Arat-MacBook-Pro-2:~ mustafamuratarat$ cd Desktop/LoanPredictionApp/
(base) Arat-MacBook-Pro-2:LoanPredictionApp/ mustafamuratarat$ pipreqs ./
INFO: Successfully saved requirements file in ./requirements.txt
```

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc10.png?raw=true)

`requirements.txt` metin dosyasını oluşturmamızın sebebi, uygulamayı canlı ortama dağıtırken kullanacağımız bulut sunucusunun gerekli gereksinimleri yüklemesi içindir.

Artık uygulamamıza sahip olduğumuza göre, uygulamayı dağıtmaya başlamaya hazırız.

## Streamlit Sharing aracılığıyla dağıtım

Uygulamalarınızı Heroku, digital ocean, AWS veya Google Cloud gibi herhangi bir özel bulut sistemi dağıtıcısında barındırabilirsiniz. Bir önceki yazımda bir Streamlit uygulamasını nasıl Heroku'da dağıltabileceğinizi göstermiştim. Burada ise biraz farklılık yaparak, Streamlit Sharing ile nasıl dağıtabileceğinizi öğreteceğim!

Sonunda, kaydedilmiş modelimizin pickle nesnesini (`classifier.pkl`), web uygulama kodumuzun olduğu `app.py` dosyasını ve `requirements.txt` dosyasını hazır hale getirdikten sonra, bunları Github hesabımızdaki herkese açık bir depoya push edebiliriz. 

Ardından [Streamlit Sharing](https://streamlit.io/){:target="_blank"} web sitesini ziyaret edeceğiz. Bu web sayfasının sağ üst köşesinde bulunan `Sign in`butonuna tıklayarak,, tarafından kullanılmak üzere Streamlit Sharing'e erişim izni verilen Github hesabınızla oturum açın.

Giriş yaptıktan sonra, New App (Yeni uygulama) butonuna tıklayacağız.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc11.png?raw=true)

Ve daha sonra gerekli alanların değerlerini giriniz:

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc12.png?raw=true)

ve "Deploy! (Dağıt!)" butonuna tıklayınız. Hatırlanması gereken önemli bir nokta, `app.py` dosyanızın, modelin `.pkl` ve `requirements.txt` dosyalarıyla birlikte Streamlit Sharing'de dağıtmak istediğiniz dalda olması gerektiğidir.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/webappML/sc13.png?raw=true)

Bir kaç dakika sonra uygulamanız canlı ortama dağıtılmış olacaktır! İşte, arkaplanında bir makine öğrenmesi modeli bulunan web uygulamanıza [https://share.streamlit.io/mmuratarat/loanpredictionapp/main/app.py](https://share.streamlit.io/mmuratarat/loanpredictionapp/main/app.py){:target="_blank"} bağlantısından ulaşabilirsiniz!

Oluşturduğumuz tüm dosyaları [burada](https://github.com/mmuratarat/LoanPredictionApp){:target="_blank"} bulunan Github deposunda bulabilirsiniz. Github sayfasındaki dosyaları her güncellemenizde, canlı uygulama da yeniden inşaa edilip, güncellenecektir.
