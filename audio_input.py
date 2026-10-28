import sounddevice as sd
from scipy.io.wavfile import write
import io
import time
from tqdm import tqdm
from utils.load_model import cargar_llm
from langchain_core.messages import HumanMessage
from groq import Groq

# Graba 5 segundos de audio desde el micrófono
duration = 5  # seconds
sample_rate = 44100

print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
# Barra de progreso para la grabación
for _ in tqdm(range(duration * 10)):   # update 10× per second
    time.sleep(0.1)
sd.wait()
print("Done.")

# Reproducción de la grabación
print("Playing...")
sd.play(audio, samplerate=sample_rate)
sd.wait()
print("Done.")

# Guardar la grabación en un archivo WAV en memoria
buf = io.BytesIO()
write(buf, sample_rate, audio)
wav_bytes = buf.getvalue()

# Transcribir el audio usando el modelo de Groq
transcripcion = Groq().audio.transcriptions.create(
    file=("grabacion.wav", wav_bytes),
    model="whisper-large-v3-turbo",
    language="es",
)
print(f"Transcripción: {transcripcion.text}")

llm_model = cargar_llm(model_name="qwen/qwen3.8-27b", max_tokens=4096, reasoning_effort="low")

response = llm_model.invoke([
    HumanMessage(f"{transcripcion.text}")
])

print(response.content)