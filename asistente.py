import whisper
import sounddevice as sd
import scipy.io.wavfile
import subprocess
import simpleaudio as sa
from gtts import gTTS
from pydub import AudioSegment
import numpy as np
import asyncio
import websockets
import threading
import keyboard

# === Configuración ===
ARCHIVO_ENTRADA = "entrada.wav"
ARCHIVO_MP3 = "respuesta.mp3"
ARCHIVO_WAV = "respuesta.wav"

# === WebSocket ===
clients = set()

async def websocket_handler(websocket):
    print("🟢 Cliente WebSocket conectado")
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        print("🔴 Cliente WebSocket desconectado")
        clients.remove(websocket)

async def iniciar_websocket():
    async with websockets.serve(websocket_handler, "localhost", 8765):
        print("✅ WebSocket escuchando en ws://localhost:8765")
        await asyncio.Future()  # Mantener vivo

def iniciar_websocket_en_hilo():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_until_complete, args=(iniciar_websocket(),), daemon=True).start()

def enviar_estado(estado):
    msg = f'{{"estado": "{estado}"}}'
    print(f"📤 Enviando estado: {estado} a {len(clients)} cliente(s)")
    asyncio.run(_broadcast(msg))

async def _broadcast(message):
    for client in clients.copy():
        try:
            await client.send(message)
            print("✅ Estado enviado a cliente")
        except Exception as e:
            print("⚠️ Error al enviar a un cliente. Eliminando cliente.")
            print("  ➤ Error:", e)
            clients.remove(client)

# === Evento para interrupciones ===
interrupcion_pendiente = threading.Event()

# === 1. Grabar voz (con tecla CTRL para detener) ===
def grabar_audio_con_teclas(nombre_archivo=ARCHIVO_ENTRADA, fs=16000):
    print("🎙 Grabando... Presiona CTRL para detener.")
    sd.default.samplerate = fs
    sd.default.channels = 1

    audio_frames = []

    def callback(indata, frames, time, status):
        if status:
            print("⚠️", status)
        audio_frames.append(indata.copy())
        if interrupcion_pendiente.is_set():
            raise sd.CallbackStop

    with sd.InputStream(callback=callback):
        while not interrupcion_pendiente.is_set():
            sd.sleep(100)

    if len(audio_frames) == 0:
        print("❌ No se capturó audio.")
        return False

    audio_np = np.concatenate(audio_frames, axis=0)
    scipy.io.wavfile.write(nombre_archivo, fs, audio_np)
    print("✅ Grabación terminada")
    return True

# === 2. Transcribir ===
def transcribir(nombre_archivo=ARCHIVO_ENTRADA):
    print("🧠 Transcribiendo audio con Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(nombre_archivo)
    print("📄 Transcripción:", result["text"])
    return result["text"].strip()

# === 3. Consultar a Ollama ===
def preguntar_ollama(prompt_usuario):
    prompt_modificado = (
        "Responde en una sola frase de no más de 50 palabras. "
        "Sé claro, directo y en español.\n\n"
        "Toma el rol de un profesor especializado en educación física y alimentaria hablandole a niños. "
        "Cada pregunta es hecha por un niño, así que responde acorde. "
        "Recuerda que eres peruano, así que contesta acorde a la cultura peruana.\n"
        f"Pregunta: {prompt_usuario}"
    )
    print("💬 Consultando a Ollama...")
    result = subprocess.run(
        ["ollama", "run", "llama3.2:1b", prompt_modificado],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    if result.stderr:
        print("⚠️ Error desde Ollama:", result.stderr)
    return result.stdout.strip()

# === 4. Texto a voz ===
def limpiar_comillas(texto):
    return texto.replace('"', '').replace("“", "").replace("”", "").replace("'", "")

def texto_a_audio(texto, archivo_mp3=ARCHIVO_MP3, archivo_wav=ARCHIVO_WAV):
    print("🔊 Convirtiendo texto a voz...")
    texto_limpio = limpiar_comillas(texto)
    tts = gTTS(text=texto_limpio, lang='es')
    tts.save(archivo_mp3)
    audio = AudioSegment.from_mp3(archivo_mp3)
    audio.export(archivo_wav, format="wav")
    print("✅ Audio generado")

# === 5. Reproducir respuesta (interrumpible con ESPACIO) ===
def reproducir_audio(archivo=ARCHIVO_WAV):
    print("📢 Reproduciendo respuesta...")
    wave_obj = sa.WaveObject.from_wave_file(archivo)
    play_obj = wave_obj.play()
    while play_obj.is_playing():
        if interrupcion_pendiente.is_set():
            play_obj.stop()
            print("⏹ Audio interrumpido por usuario")
            return
        sd.sleep(100)
    print("✅ Reproducción terminada")

# === 6. Escuchar tecla ESPACIO para iniciar, CTRL para detener ===
def esperar_teclas():
    while True:
        keyboard.wait("space")
        interrupcion_pendiente.set()

# === 🚀 Conversación ===
if __name__ == "__main__":
    print("🔌 Iniciando WebSocket...")
    iniciar_websocket_en_hilo()

    print("⌨️ Escuchando tecla ESPACIO para iniciar...")
    threading.Thread(target=esperar_teclas, daemon=True).start()

    while True:
        print("🕒 Esperando ESPACIO para iniciar grabación...")
        interrupcion_pendiente.wait()
        interrupcion_pendiente.clear()

        enviar_estado("escuchar")

        print("⌨️ Presiona CTRL para detener grabación...")
        keyboard.add_hotkey("ctrl", lambda: interrupcion_pendiente.set())
        exito = grabar_audio_con_teclas()
        keyboard.remove_hotkey("ctrl")

        if not exito:
            continue

        enviar_estado("procesar")
        texto_usuario = transcribir()
        print("🧑 Tú dijiste:", texto_usuario)

        if texto_usuario.lower().strip() in ["salir", "terminar", "adiós"]:
            print("👋 Terminando la conversación.")
            break

        respuesta_llm = preguntar_ollama(texto_usuario)
        print("🤖 Asistente responde:", respuesta_llm)

        enviar_estado("hablar")
        texto_a_audio(respuesta_llm)

        interrupcion_pendiente.clear()
        print("⌨️ Puedes presionar ESPACIO para interrumpir el audio.")
        reproducir_audio()

    print("✅ Asistente finalizado.")
