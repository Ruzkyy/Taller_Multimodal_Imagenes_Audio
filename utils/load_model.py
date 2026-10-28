import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODELO_LLM = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")


def cargar_llm(model_name=None, temperature=0, **kwargs):
    """Carga un modelo LLM de Groq a través de LangChain.

    Args:
        model_name: nombre del modelo en Groq (por defecto, LLM_MODEL del .env).
        temperature: 0 da respuestas más precisas y repetibles; valores altos, más creativas.
        **kwargs: parámetros adicionales de ChatGroq (p. ej. max_tokens, reasoning_effort).

    Returns:
        Objeto ChatGroq con el método invoke para enviar mensajes.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Falta la variable GROQ_API_KEY en el archivo .env")

    return ChatGroq(
        model=model_name or MODELO_LLM,
        temperature=temperature,
        **kwargs,
    )


if __name__ == "__main__":
    llm = cargar_llm()
    respuesta = llm.invoke("Responde en una frase: ¿qué es un sistema RAG?")
    print(respuesta.content)
