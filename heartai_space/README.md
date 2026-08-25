---
title: HeartAI Space
emoji: 🫀
colorFrom: red
colorTo: black
sdk: docker
app_port: 7860
pinned: false
---

Бэкенд HeartAI (demorg) — FastAPI-сервер, отдаёт ответы модели по `/v1/ask`.
Веса подтягиваются с `renderru/heartai-demorg` при старте контейнера.

Секреты Space: задай `MAIN_API_KEY`, иначе используется дефолтный ключ из кода.
