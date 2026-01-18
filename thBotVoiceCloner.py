import asyncio
import logging
from pathlib import Path
import sys
import re

import torch
import torchaudio
from torchaudio.functional import pitch_shift  # ← только нужная функция, без конфликта

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile

# CosyVoice
sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel

# ──────────────────────────────────────────────
# НАСТРОЙКИ
# ──────────────────────────────────────────────

BOT_TOKEN = "YOUR_BOT_TOKEN"
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
SAMPLE_RATE_OUT = 22050
REFERENCE_SR = 16000

USERS_DIR = Path("users")
USERS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

cosyvoice = None

# ──────────────────────────────────────────────


def get_user_folder(user_id: int) -> Path:
    folder = USERS_DIR / str(user_id)
    folder.mkdir(exist_ok=True)
    return folder


async def load_model():
    global cosyvoice
    if cosyvoice is not None:
        return

    logger.info("Загружаю CosyVoice...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cosyvoice = AutoModel(
        model_dir=MODEL_DIR,
        fp16=(device == "cuda")
    )
    logger.info("CosyVoice загружен")


def preprocess_reference_audio(input_path: Path, output_path: Path, target_sr: int = REFERENCE_SR):
    TRIM_LAST_SEC = 0.5
    PITCH_SHIFT_SEMITONES = 0.3

    try:
        waveform, sr = torchaudio.load(str(input_path))

        # Моно
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ресемплинг
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Обрезка конца
        samples_to_trim = int(target_sr * TRIM_LAST_SEC)
        if waveform.shape[1] > samples_to_trim:
            waveform = waveform[:, :-samples_to_trim]

        # DC offset
        waveform = waveform - waveform.mean()

        # Подъём тона (без лишних аргументов)
        if PITCH_SHIFT_SEMITONES != 0:
            waveform = pitch_shift(
                waveform=waveform,
                sample_rate=target_sr,
                n_steps=PITCH_SHIFT_SEMITONES,
                bins_per_octave=12
            )
            logger.info(f"Тон поднят на {PITCH_SHIFT_SEMITONES} полутонов")

        # Нормализация
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform * (0.98 / max_val)

        # Сохранение
        torchaudio.save(
            str(output_path),
            waveform,
            target_sr,
            encoding='PCM_S',
            bits_per_sample=16
        )

        duration_sec = waveform.shape[1] / target_sr
        logger.info(f"Референс готов: {duration_sec:.2f} сек, тон +{PITCH_SHIFT_SEMITONES}")

        return True

    except Exception as e:
        logger.error(f"Ошибка предобработки: {e}")
        return False


def split_text_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\»)\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= 10]


def generate_audio_chunk(tts_text: str, prompt_text: str, prompt_wav: str) -> torch.Tensor:
    if len(tts_text.strip()) < 5:
        return torch.zeros((1, int(SAMPLE_RATE_OUT * 0.3)))

    result = cosyvoice.inference_zero_shot(
        tts_text=tts_text,
        prompt_text=prompt_text,
        prompt_wav=prompt_wav,
        stream=False
    )

    audio = next((item["tts_speech"] for item in result), None)
    if audio is None or audio.numel() == 0:
        return torch.zeros((1, int(SAMPLE_RATE_OUT * 1.0)))

    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio * (0.98 / max_val)

    return audio


router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Пришли голосовое — это будет образец.\n\n"
        "Лучше всего: «Съешь ещё этих мягких французских булок, да выпей чаю.» + пауза 0.4 сек"
    )


@router.message(F.voice)
async def handle_voice(message: Message, bot: Bot):
    user_id = message.from_user.id
    folder = get_user_folder(user_id)

    temp_ogg = folder / "temp.ogg"
    input_wav = folder / "input_audio.wav"

    try:
        file = await bot.get_file(message.voice.file_id)
        await bot.download_file(file.file_path, temp_ogg)

        success = preprocess_reference_audio(temp_ogg, input_wav)

        temp_ogg.unlink(missing_ok=True)

        if success:
            await message.answer("Голос сохранён ✓\nТеперь пиши текст")
        else:
            await message.answer("Не получилось обработать. Запиши заново.")

    except Exception as e:
        logger.exception("Ошибка голосового")
        await message.answer("Ошибка при обработке голосового")


@router.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    folder = get_user_folder(user_id)

    text = message.text.strip()
    if not text:
        await message.answer("Напиши хоть что-нибудь...")
        return

    text_path = folder / "text.txt"
    input_path = folder / "input_audio.wav"
    output_path = folder / "output_audio.wav"

    text_path.write_text(text, encoding="utf-8")

    if not input_path.exists():
        await message.answer("Сначала пришли голосовое сообщение.")
        return

    await message.answer("Генерирую... (по частям для качества)")

    try:
        sentences = split_text_into_sentences(text)

        audio_chunks = []
        prompt_text = "Съешь ещё этих мягких французских булок, да выпей чаю. <|endofprompt|>"  # Съешь ещё этих мягких французских булок, да выпей чаю! <|endofprompt|> или твой топовый промпт, e.g. "Съешь ещё этих мягких французских булок, да выпей чаю.<|endofprompt|>"  # ← самый стабильный вариант

        for i, sentence in enumerate(sentences):
            logger.info(f"Кусок {i+1}/{len(sentences)}: '{sentence[:50]}...'")
            chunk = generate_audio_chunk(sentence, prompt_text, str(input_path))

            # Fade для чистых стыков
            fade = torchaudio.transforms.Fade(
                fade_in_len=int(SAMPLE_RATE_OUT * 0.04),
                fade_out_len=int(SAMPLE_RATE_OUT * 0.08),
                fade_shape='linear'
            )
            chunk = fade(chunk)

            audio_chunks.append(chunk)

        # Паузы между предложениями
        silence = torch.zeros((1, int(SAMPLE_RATE_OUT * 0.15)), dtype=torch.float32)
        full_parts = []
        for i, chunk in enumerate(audio_chunks):
            full_parts.append(chunk)
            if i < len(audio_chunks) - 1:
                full_parts.append(silence)

        full_audio = torch.cat(full_parts, dim=-1)

        # Финальная нормализация
        max_val = full_audio.abs().max()
        if max_val > 0:
            full_audio = full_audio * (0.98 / max_val)

        torchaudio.save(str(output_path), full_audio, SAMPLE_RATE_OUT)

        await message.answer_voice(
            FSInputFile(output_path),
            caption=f"«{text[:70]}…»"
        )

    except Exception as e:
        logger.exception("Ошибка синтеза")
        await message.answer(f"Ошибка:\n{str(e)[:150]}")


async def main():
    bot = Bot(token=BOT_TOKEN)
    await load_model()
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())