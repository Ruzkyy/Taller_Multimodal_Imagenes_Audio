from utils.load_documents import cargar_pdfs
from utils.chunking import crear_chunks
from utils.embeddings import cargar_modelo_embeddings
from utils.store_vector import crear_vector_store
from utils.retrieval import recuperar_chunks
from utils.load_model import cargar_llm

def context(resultados):
    contexto="\n\n".join([f"Fuente: {doc.metadata.get('source', '?')}, Pág. {doc.metadata.get('page', '?')}\n{doc.page_content}" for doc, _ in resultados])
    return contexto



if __name__ == "__main__":
    # Cargar documentos PDF
    documentos=cargar_pdfs(verbose=False)
    # Crear chunks de los documentos cargados
    chunks=crear_chunks(documentos, verbose=False, ejemplo=20)
    # Cargar modelo embeddings para los chunks
    modelo_embeddings=cargar_modelo_embeddings()
    # Crear y almacenar la base vectorial con los chunks y el modelo de embeddings
    vector_store=crear_vector_store(chunks, modelo_embeddings, reset=True, verbose=False)

    PROMPT_TEMPLATE = """Eres un asistente experto en responder preguntas basadas en los reglamentos institucionales.
    Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado. Al final de cada parte de la respuesta, incluye entre paréntesis la fuente y la página del fragmento de donde proviene la información, por ejemplo: (Fuente: NombreDocumento.pdf, Pág. X).
    Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

    Contexto recuperado del documento:
    {context}

    Pregunta del usuario: {question}

    Respuesta:"""

    llm_model=cargar_llm()


    while True:
        pregunta=input("\nPregunta (o 'salir' para terminar): ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        # Recuperar los chunks más parecidos a la pregunta
        resultados=recuperar_chunks(vector_store, pregunta, k=5, verbose=True)
        # Construir el contexto a partir de los chunks recuperados
        contexto=context(resultados)
        # Construir el prompt final para el modelo LLM
        prompt_final=PROMPT_TEMPLATE.format(context=contexto, question=pregunta)
        # Obtener la respuesta del modelo LLM
        respuesta=llm_model.invoke(prompt_final)
        print(f"\nRespuesta: {respuesta.content}")
