from datetime import date
from dotenv import load_dotenv
from utils.load_model import cargar_llm

load_dotenv()

llm_model = cargar_llm(max_tokens=4096, reasoning_effort="low")

#¿Qué tan actualizados están sus conocimientos?
#¿Quién es el presidente de Colombia?
#¿Quien ganó el balón de oro en 2026?
pregunta = "¿Quien ganó el balón de oro en 2026?"

respuesta = llm_model.invoke(pregunta)

print(f"Fecha de hoy: {date.today()}")
print(f"Pregunta: {pregunta}")
print("*" * 30)
print("Razonamiento del modelo:")
print(respuesta.additional_kwargs.get("reasoning_content", "(el modelo no devolvió razonamiento)"))
print("*" * 30)
print(f"Respuesta: {respuesta.content}")
print("*" * 30)
print(f"Motivo de finalización: {respuesta.response_metadata.get('finish_reason')}")
print(f"Tokens usados: {respuesta.usage_metadata}")
