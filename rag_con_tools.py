import os
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from utils.load_documents import cargar_pdfs
from utils.chunking import crear_chunks
from utils.embeddings import cargar_modelo_embeddings
from utils.store_vector import crear_vector_store, PERSIST_DIR
from utils.retrieval import cargar_vector_store, recuperar_chunks
from utils.load_model import cargar_llm

SYSTEM_PROMPT = """Eres un asistente experto en responder preguntas basadas en los reglamentos institucionales.
Para responder preguntas sobre los reglamentos, usa la herramienta buscar_reglamentos. Puedes llamarla varias veces con consultas distintas si lo necesitas.
Responde usando ÚNICAMENTE la información que devuelva la herramienta. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
Si la respuesta no está en la información recuperada, indica exactamente: "No encontré información sobre esto en la base de conocimientos."
Si el usuario solo saluda o conversa, responde con normalidad sin usar la herramienta."""

MAX_ITERACIONES = 5  # límite de rondas de tools por pregunta, para evitar bucles infinitos


def context(resultados):
    contexto="\n\n".join([f"Fuente: {doc.metadata.get('source', '?')}, Pág. {doc.metadata.get('page', 0)+1}\n{doc.page_content}" for doc, _ in resultados])
    return contexto


def crear_tools(vector_store):
    """Crea las tools que el LLM puede usar, conectadas a la base vectorial."""

    @tool
    def buscar_reglamentos(consulta: str) -> str:
        """Busca en los reglamentos institucionales fragmentos relevantes para la consulta.
        Úsala siempre que la pregunta trate sobre normas, derechos, deberes, sanciones o procedimientos."""
        resultados = recuperar_chunks(vector_store, consulta, k=5, verbose=False)
        return context(resultados)

    return [buscar_reglamentos]


def responder(llm_con_tools, tools_por_nombre, pregunta):
    """Ejecuta el bucle de tool calling hasta que el LLM da una respuesta final."""
    mensajes = [SystemMessage(SYSTEM_PROMPT), HumanMessage(pregunta)]

    for _ in range(MAX_ITERACIONES):
        respuesta = llm_con_tools.invoke(mensajes)
        mensajes.append(respuesta)

        # Si el LLM no pide ninguna tool, esta es la respuesta final
        if not respuesta.tool_calls:
            return respuesta.content

        # Ejecutar cada tool que pidió el LLM y devolverle el resultado
        for tool_call in respuesta.tool_calls:
            print(f"  [tool] {tool_call['name']}({tool_call['args']})")
            herramienta = tools_por_nombre[tool_call["name"]]
            resultado = herramienta.invoke(tool_call["args"])
            mensajes.append(ToolMessage(content=resultado, tool_call_id=tool_call["id"]))

    return "No pude completar la respuesta: se alcanzó el límite de llamadas a herramientas."


if __name__ == "__main__":
    # Cargar modelo embeddings
    modelo_embeddings=cargar_modelo_embeddings()

    # Abrir la base vectorial existente o crearla si todavía no existe
    if os.path.isdir(PERSIST_DIR):
        vector_store=cargar_vector_store(modelo_embeddings)
    else:
        documentos=cargar_pdfs(verbose=False)
        chunks=crear_chunks(documentos, verbose=False)
        vector_store=crear_vector_store(chunks, modelo_embeddings, reset=True, verbose=False)

    # Cargar el LLM y conectarle las tools
    tools=crear_tools(vector_store)
    tools_por_nombre={t.name: t for t in tools}
    llm_con_tools=cargar_llm().bind_tools(tools)

    while True:
        pregunta=input("\nPregunta (o 'salir' para terminar): ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        respuesta=responder(llm_con_tools, tools_por_nombre, pregunta)
        print(f"\nRespuesta: {respuesta}")
