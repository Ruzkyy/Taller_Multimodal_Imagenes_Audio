from datetime import date
from dotenv import load_dotenv
from utils.load_model import cargar_llm
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from tavily import TavilyClient
from typing import Dict, Any

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(
        query=query,
        include_answer='basic',
        max_results=5,
        start_date="2026-01-27",
        end_date="2026-10-27"
        )

# respuesta = web_search.invoke("¿Quién ganó el balón de oro en 2026?")

# print(f"Respuesta de la búsqueda web: {respuesta.get('answer')}")



llm_model = cargar_llm(max_tokens=4096, reasoning_effort="low").bind_tools([web_search])
#¿Qué tan actualizados están sus conocimientos?
#¿Quién es el presidente de Colombia?
#¿Quien ganó el balón de oro en 2026?
pregunta = "¿Quién es el presidente de Colombia?"

# El system prompt le da la fecha actual y le indica que busque en la web,
# porque el modelo "cree" saber la respuesta y no usaría la tool por sí solo
SYSTEM_PROMPT = (
    f"Hoy es {date.today()}. Tu conocimiento tiene una fecha de corte anterior, "
    "así que para cualquier pregunta sobre hechos actuales o recientes usa siempre "
    "la herramienta web_search antes de responder."
)

# 1. Enviar la pregunta: el LLM responde pidiendo la tool (tool_calls)
mensajes = [SystemMessage(SYSTEM_PROMPT), HumanMessage(pregunta)]
respuesta = llm_model.invoke(mensajes)
mensajes.append(respuesta)  # la petición del LLM debe quedar en el historial
print(f"Petición del LLM: {respuesta.tool_calls}")

# 2. Ejecutar cada tool pedida y devolver el resultado con un ToolMessage
for tool_call in respuesta.tool_calls:
    resultado = web_search.invoke(tool_call["args"])
    print(f"Resultado de la tool: {resultado.get('answer')}")
    mensajes.append(ToolMessage(content=str(resultado), tool_call_id=tool_call["id"]))

# 3. Volver a llamar al LLM con el resultado para que redacte la respuesta final
respuesta_final = llm_model.invoke(mensajes)

print(f"Fecha de hoy: {date.today()}")
print(f"Pregunta: {pregunta}")
print("*" * 30)
print("Razonamiento del modelo:")
print(respuesta_final.additional_kwargs.get("reasoning_content", "(el modelo no devolvió razonamiento)"))
print("*" * 30)
print(f"Respuesta: {respuesta_final.content}")
print("*" * 30)
print(f"Motivo de finalización: {respuesta_final.response_metadata.get('finish_reason')}")
print(f"Tokens usados: {respuesta_final.usage_metadata}")
