# JSON nedir?


Daha yaygın olarak JSON kısaltmasıyla bilinen JavaScript Nesne Gösterimi (**J**ava**S**cript **O**bject **N**otation), hem insan hem de makine tarafından okunabilen bir açık veri değişim biçimidir. Genellikle web uygulamalarında veri iletimini sağlamak için kullanılır (örneğin, bir web sayfasında görüntülenebilmesi için sunucudan (server) istemciye (client) bazı verilerin gönderilmesi veya tersi). Douglas Crockford, JSON veri formatını popüler hale getirdi. İsminin içerisinde JavaScript geçmesine rağmen, JSON herhangi bir programlama dilinden bağımsızdır ve çok çeşitli uygulamalarda ortak bir API çıktısıdır.
 
JSON yalnızca iki veri yapısını tanımlar: nesneler (object) ve diziler (array). Bir nesne, bir ad-değer (veya anahtar-değer (key-value)) çiftleri kümesidir (Python'da sözlüğe (dictionary) karşılık gelir). Sol ({) ve sağ (}) ayraçlar içinde bir nesne tanımlanır. Her ad-değer çifti, adla başlar, ardından iki nokta üst üste ve ardından değer gelir. Ad-değer çiftleri virgülle ayrılır. Bir dizi, bir değerler listesidir. Sol ([) ve sağ (]) parantez içinde bir dizi tanımlanır. Dizideki öğeler virgülle ayrılır. Bir dizideki her öğe, başka bir dizi veya bir nesne dahil olmak üzere farklı bir türde olabilir.

JSON yedi değer türü tanımlar: dizgi (string), sayı (number), nesne (object), dizi (array), doğru (true), yanlış (false) ve boş (null). Python nesneleri ve bunların JSON'a eşdeğer dönüşümü aşağıda verilmiştir:

|    **Python**   | **JSON Karşılığı** |
|:-----------:|:---------------:|
|     dict    |      object     |
| list, tuple |      array      |
|     str     |      string     |
|  int, float |      number     |
|     True    |       true      |
|    False    |      false      |
|     None    |       null      |

Aşağıdaki örnek, ad-değer çiftlerini içeren bir örnek nesne için JSON verilerini gösterir. `phoneNumbers` adının değeri, öğeleri iki nesne olan bir dizidir.

```JSON
{
   "firstName": "Duke",
   "lastName": "Java",
   "age": 18,
   "streetAddress": "100 Internet Dr",
   "city": "JavaTown",
   "state": "JA",
   "postalCode": "12345",
   "phoneNumbers": [
      { "Mobile": "111-111-1111" },
      { "Home": "222-222-2222" }
   ]
}
```

Nesneler ve diziler başka nesneler veya diziler içerdiğinde, veriler ağaç benzeri (tree-like) bir yapıya sahiptir.

JSON verileriyle çalışmak için Python, `json` adlı yerleşik bir pakete sahiptir.
