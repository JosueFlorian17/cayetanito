import torch

# 🚨 Importa TODAS las clases que XTTS necesita para cargar el modelo correctamente en PyTorch 2.6+
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

# ✅ Habilita todas como "safe" para torch.load
torch.serialization.add_safe_globals([
    BaseDatasetConfig,
    XttsConfig,
    XttsAudioConfig,
    XttsArgs
])


from TTS.api import TTS
import os
import time
from pydub import AudioSegment

# === 1. Parámetros ===
texto = "¡Hola! Soy un robot peruano, listo para ayudarte con tu alimentación saludable."
speaker_wav_path = "voz_latina.wav"  # Graba tu voz con acento latino y ponle este nombre
archivo_original = "voz_generada.wav"
archivo_final = "voz_robotica.wav"

# === 2. Verificar que el archivo de voz exista ===
if not os.path.exists(speaker_wav_path):
    raise FileNotFoundError(f"No se encontró el archivo de voz: {speaker_wav_path}")

# === 3. Cargar modelo XTTS ===
print("\n⏳ Cargando modelo XTTS...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

# === 4. Sintetizar el texto con tu voz ===
print("\n🔊 Sintetizando voz personalizada...")
start = time.time()
tts.tts_to_file(
    text=texto,
    file_path=archivo_original,
    speaker_wav=speaker_wav_path,
    language="es"
)
end = time.time()
print(f"✅ Voz generada: {archivo_original}")
print(f"🕒 Tiempo de inferencia: {end - start:.2f} segundos")

# === 5. Efectos robóticos con pydub ===
print("\n🎛️ Aplicando efectos robot...")

# Cargar audio original
audio = AudioSegment.from_wav(archivo_original)

# Aumentar pitch y aplicar eco
pitch_factor = 1.3
speed_factor = 1.0

# Cambiar pitch
audio_mod = audio._spawn(audio.raw_data, overrides={
    "frame_rate": int(audio.frame_rate * pitch_factor)
}).set_frame_rate(audio.frame_rate)

# Acelerar (si se desea)
audio_mod = audio_mod._spawn(audio_mod.raw_data, overrides={
    "frame_rate": int(audio_mod.frame_rate * speed_factor)
}).set_frame_rate(audio.frame_rate)

# Aplicar eco corto y metálico
eco = audio_mod - 6
audio_robot = eco.overlay(audio_mod, position=25)

# Exportar resultado
audio_robot.export(archivo_final, format="wav")
print(f"✅ Voz robotica guardada como: {archivo_final}")
