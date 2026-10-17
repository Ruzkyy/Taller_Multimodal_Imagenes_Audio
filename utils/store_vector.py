from langchain_chroma import Chroma
import shutil
import os

PERSIST_DIR = os.getenv("PERSIST_DIR", "./db")


def crear_vector_store(chunks, embeddings_model, persist_dir=PERSIST_DIR,
                       collection_name="mis_documentos", reset=True, verbose=True):
    """Genera los embeddings de los chunks y los guarda en una base vectorial ChromaDB.

    Args:
        chunks: lista de Document (por ejemplo, la salida de crear_chunks).
        embeddings_model: modelo de embeddings de LangChain (por ejemplo, cargar_modelo_embeddings()).
        persist_dir: carpeta donde se guarda la base vectorial.
        collection_name: nombre de la colección dentro de ChromaDB.
        reset: si es True, borra la base anterior antes de indexar.
        verbose: si es True, imprime el progreso y un resumen.

    Returns:
        Objeto Chroma listo para hacer búsquedas.
    """
    if reset and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        if verbose:
            print(f"Base de datos anterior eliminada: {persist_dir}")

    if verbose:
        print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")
        print("(Embeddings generados localmente con Sentence Transformers)")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if verbose:
        total = vector_store._collection.count()
        dimension = len(embeddings_model.embed_query("prueba"))
        modelo = getattr(embeddings_model, "model_name", type(embeddings_model).__name__)
        print()
        print("[OK] Base vectorial creada exitosamente!")
        print(f"  Ubicación:             {persist_dir}/")
        print(f"  Fragmentos indexados:  {total}")
        print(f"  Modelo embeddings:     {modelo}")
        print(f"  Dimensión de vectores: {dimension}")

    return vector_store
