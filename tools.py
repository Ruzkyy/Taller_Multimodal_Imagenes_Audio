from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

from utils.load_model import cargar_llm

load_dotenv()

# Creación de una tool
@tool("nota_final_kl")
def nota_final_kl(parcial1: float, parcial2: float) -> float:
    """Calcula la nota final de un estudiante según la fórmula de la Konrad Lorenz:
    40% del primer parcial y 60% del segundo parcial.

    Args:
        parcial1: nota del primer parcial (de 0 a 50).
        parcial2: nota del segundo parcial (de 0 a 50).
    """
    return parcial1 * 0.4 + parcial2 * 0.6

# prueba de la tool
nota_final = nota_final_kl.invoke({"parcial1": 3.5, "parcial2": 4.0})

print(f"Nota final: {nota_final}")

llm_model = cargar_llm()

pregunta = "Responde en una frase: ¿qué es un sistema RAG?"

respuesta = llm_model.invoke(pregunta)
print(f"Respuesta: {respuesta.content}")

# Usar la tool desde el LLM
pregunta_tool = "Calcula la nota final de un estudiante que obtuvo 3.5 en el primer parcial y 4.0 en el segundo parcial."
llm_model_con_tools = cargar_llm().bind_tools([nota_final_kl])

# 1. Enviar la pregunta: el LLM responde pidiendo la tool (tool_calls)
mensajes = [HumanMessage(pregunta_tool)]
respuesta_tool = llm_model_con_tools.invoke(mensajes)
mensajes.append(respuesta_tool)  # la petición del LLM debe quedar en el historial
print(f"Petición del LLM: {respuesta_tool.tool_calls}")

# 2. Ejecutar cada tool pedida y devolver el resultado con un ToolMessage
for tool_call in respuesta_tool.tool_calls:
    resultado = nota_final_kl.invoke(tool_call["args"])
    print(f"Resultado de la tool: {resultado}")
    mensajes.append(ToolMessage(content=str(resultado), tool_call_id=tool_call["id"]))

# 3. Volver a llamar al LLM con el resultado para que redacte la respuesta final
respuesta_final = llm_model_con_tools.invoke(mensajes)
print(f"Respuesta final: {respuesta_final.content}")
