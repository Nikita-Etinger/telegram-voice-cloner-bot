# Telegram Voice Cloner Bot

Простой Telegram-бот для **клонирования голоса** на базе **CosyVoice 3** (FunAudioLLM).  
Достаточно одного голосового сообщения - и бот озвучивает любой текст вашим голосом.

## Возможности

- Zero-shot клонирование голоса (5-12 секунд референса достаточно)
- Полная предобработка аудио:  
  - моно + 16 кГц  
  - обрезка шума с конца  
  - подъём тона (+0.5 полутонов) - убирает «грубость»  
  - нормализация громкости
- Разбиение длинных текстов на предложения  
- Чистые стыки: fade-in/out + короткие паузы между частями
- Стабильная генерация без обрезки начала и артефактов

**Лучшее качество на русском** достигается с референсным текстом:  Съешь ещё этих мягких французских булок, да выпей чаю.


## Установка и запуск

1. Клонируйте официальный репозиторий CosyVoice  
   ```bash
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   cd CosyVoice
   pip install -r requirements.txt
2. Установите aiogram  
   ```bash
   pip install aiogram
3. Скачайте модель Fun-CosyVoice3-0.5B-2512
   ```bash
   Через Hugging Face: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
   Или используйте скрипт download_model.py из проекта
4. Вставьте токен вашего бота в файл thBotVoiceCloner.py:  
   ```bash
   BOT_TOKEN = "YOUR_BOT_TOKEN"

5. Запустите бота   
   ```bash
    thBotVoiceCloner.py
   
## Тестирование без бота

1. Выполните пункт 1,3.
2. Создайте аудиофаил с референсным голосом reference_voice.wav
3. Запустите test_cosyvoice.py
4. Результат будет сохранен как clone_voice_result.wav
