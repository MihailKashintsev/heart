"""Локальный тест обученной модели: python3 chat.py

Загружает minigpt_v3.pt (если файла нет — создаётся необученная модель,
это нормально для проверки, что скрипт работает) и даёт пообщаться
в терминале тем же способом, что использует heartai_space/app.py.
"""
from train import model, tokenizer, generate

print("HeartAI — Ctrl+C или пустая строка для выхода\n")
while True:
    try:
        msg = input("Ты: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not msg:
        break
    print("demorg:", generate(model, tokenizer, msg), "\n")
