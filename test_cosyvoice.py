import sys
import torch
import torchaudio

sys.path.append("third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
PROMPT_WAV = "reference_voice.wav"

cosyvoice = AutoModel(
    model_dir=MODEL_DIR,
    fp16=torch.cuda.is_available()
)

text = (
    ""

)

prompt_text = (
    "[RU]<|endofprompt|>" # [RU] - универсально для русского языка
)

for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        prompt_text,
        text,
        PROMPT_WAV,
        stream=False
    )
):
    torchaudio.save(
        "clone_voice_result.wav",
        j["tts_speech"],
        cosyvoice.sample_rate
    )
    break

print("DONE")