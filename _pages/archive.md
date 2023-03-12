---
layout: post
title: "Arşiv"
author: MMA
social: true
comments: false
permalink: /archive/
years:
- 2023
- 2022
- 2021
- 2020
---

{% for year in page.years %}
{% assign y1 = year | plus: 0 %}
# {{ y1 }}
<ul>
{% for post in site.posts %}
{% assign y2 = post.date | date: '%Y' | plus: 0 %}
{% if y1 == y2 %}
<li style="line-height:1.5em">{% assign m = post.date | date: "%-m" %}
{{ post.date | date: "%-d" }}
{% case m %}
  {% when '1' %}Ocak
  {% when '2' %}Şubat
  {% when '3' %}Mart
  {% when '4' %}Nisan
  {% when '5' %}Mayıs
  {% when '6' %}Haziran
  {% when '7' %}Temmuz
  {% when '8' %}Ağustos
  {% when '9' %}Eylül
  {% when '10' %}Ekim
  {% when '11' %}Kasım
  {% when '12' %}Aralık
{% endcase %} &middot; <a href="{{ post.url| prepend: site.baseurl }}" target="_blank">{{ post.title }}</a></li>
{% endif %}
{% endfor %}
</ul>
{% endfor %}
