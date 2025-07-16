import whisper
import sounddevice as sd
import scipy.io.wavfile
import subprocess
import simpleaudio as sa
from gtts import gTTS
from pydub import AudioSegment
import asyncio
import websockets
import threading

# === Configuración ===
DURACION_GRABACION = 5
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

# === 1. Grabar voz ===
def grabar_audio(nombre_archivo=ARCHIVO_ENTRADA, duracion=DURACION_GRABACION, fs=16000):
    print("🎙 Grabando...")
    sd.default.samplerate = fs
    sd.default.channels = 1
    audio = sd.rec(int(duracion * fs))
    sd.wait()
    scipy.io.wavfile.write(nombre_archivo, fs, audio)
    print("✅ Grabación terminada")

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

# === 4. Usar gTTS y convertir a WAV ===
def texto_a_audio(texto, archivo_mp3=ARCHIVO_MP3, archivo_wav=ARCHIVO_WAV):
    print("🔊 Convirtiendo texto a voz...")
    tts = gTTS(text=texto, lang='es')
    tts.save(archivo_mp3)
    audio = AudioSegment.from_mp3(archivo_mp3)
    audio.export(archivo_wav, format="wav")
    print("✅ Audio generado")

# === 5. Reproducir respuesta ===
def reproducir_audio(archivo=ARCHIVO_WAV):
    print("📢 Reproduciendo respuesta...")
    wave_obj = sa.WaveObject.from_wave_file(archivo)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    print("✅ Reproducción terminada")

# === 🚀 Conversación ===
if __name__ == "__main__":
    print("🔌 Iniciando WebSocket...")
    iniciar_websocket_en_hilo()
    print("🗣 Asistente iniciado. Di algo... (di 'salir' para terminar)\n")

    while True:
        enviar_estado("escuchar")
        grabar_audio()

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
        reproducir_audio()

    print("✅ Asistente finalizado.")
