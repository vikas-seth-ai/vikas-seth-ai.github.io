---
layout: page
title: Notes
permalink: /notes/
---

Technical write-ups, shipped every 2–3 weeks.

{% if site.posts.size > 0 %}
<ul class="entry-list">
  {% for post in site.posts %}
  <li class="entry">
    <a href="{{ post.url | relative_url }}">
      <span class="entry-date">{{ post.date | date: "%b %-d, %Y" }}</span>
      <span class="entry-title">{{ post.title }}</span>
    </a>
  </li>
  {% endfor %}
</ul>
{% else %}
<p class="empty-state">Nothing published yet.</p>
{% endif %}
