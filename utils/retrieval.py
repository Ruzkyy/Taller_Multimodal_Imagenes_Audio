import os
from langchain_chroma import Chroma
from utils.store_vector import PERSIST_DIR


def cargar_vector_store(embeddings_model, persist_dir=PERSIST_DIR, collection_name="mis_documentos"):
    """Abre una base vectorial ChromaDB ya creada, sin volver a indexar.

    Args:
        embeddings_model: el mismo modelo de embeddings usado al crear la base.
        persist_dir: carpeta donde está guardada la base vectorial.
        collection_name: nombre de la colección dentro de ChromaDB.

    Returns:
        Objeto Chroma listo para hacer búsquedas.
    """
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"No existe la base vectorial en \"{persist_dir}\". Créala con crear_vector_store.")

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings_model,
        collection_name=collection_name,
    )


def recuperar_chunks(vector_store, pregunta, k=4, verbose=True):
    """Recupera los k chunks más parecidos a la pregunta.

    Args:
        vector_store: base vectorial (salida de crear_vector_store o cargar_vector_store).
        pregunta: texto de la consulta.
        k: número de chunks a recuperar.
        verbose: si es True, imprime los chunks recuperados.

    Returns:
        Lista de tuplas (Document, distancia). Con distancia coseno,
        un valor más bajo significa más parecido (0 = idéntico).
    """
    resultados = vector_store.similarity_search_with_score(pregunta, k=k)

    if verbose:
        print(f"Pregunta: {pregunta}")
        print(f"Chunks recuperados: {len(resultados)}")
        for i, (doc, distancia) in enumerate(resultados, 1):
            print()
            print(f"[{i}] Distancia: {distancia:.4f}")
            print(f"    Fuente: {doc.metadata.get('source', '?')}")
            print(f"    Página: {doc.metadata.get('page', '?')}")
            print(f"    Contenido: {doc.page_content[:300]}")

    return resultados
