from datetime import date
import sounddevice as sd
from scipy.io.wavfile import write
import io
import time
from tqdm import tqdm
from dotenv import load_dotenv
from utils.load_model import cargar_llm
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from groq import Groq
from tavily import TavilyClient
from typing import Any, Dict

load_dotenv()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Busca en la web hechos actuales, cifras recientes y récords que puedan haber cambiado.

    Args:
        query: Pregunta concreta para buscar información y fuentes actuales.
    """
    return TavilyClient().search(query=query, include_answer="basic", max_results=5)

# Graba 20 segundos de audio desde el micrófono
duration = 20  # seconds
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

llm_model = cargar_llm(max_tokens=4096, reasoning_effort="low")
llm_model_con_busqueda = llm_model.bind_tools([web_search])

system_prompt = (
    f"Hoy es {date.today()}. Para preguntas sobre hechos actuales, cifras recientes "
    "o récords que puedan haber cambiado, usa siempre web_search antes de responder. "
    "Basa la respuesta en los resultados y menciona las fuentes disponibles. "
    "Si la búsqueda no aporta datos fiables o suficientes, dilo claramente."
)
mensajes = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=transcripcion.text),
]
response = llm_model_con_busqueda.invoke(mensajes)

while response.tool_calls:
    mensajes.append(response)
    for tool_call in response.tool_calls:
        if tool_call["name"] != web_search.name:
            raise RuntimeError(f"Herramienta no soportada: {tool_call['name']}")
        resultado = web_search.invoke(tool_call["args"])
        mensajes.append(
            ToolMessage(content=str(resultado), tool_call_id=tool_call["id"])
        )
    response = llm_model_con_busqueda.invoke(mensajes)

print(response.content)
