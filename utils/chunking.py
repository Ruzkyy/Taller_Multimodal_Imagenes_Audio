from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.load_documents import cargar_pdfs

def crear_chunks(documents, chunk_size=500, chunk_overlap=50, verbose=True, ejemplo=None):
    """Divide una lista de Document en fragmentos (chunks) más pequeños.

    Args:
        documents: lista de Document (por ejemplo, la salida de cargar_pdfs).
        chunk_size: tamaño máximo de cada fragmento, en caracteres.
        chunk_overlap: caracteres compartidos entre fragmentos consecutivos.
        verbose: si es True, imprime estadísticas del chunking.
        ejemplo: índice de un fragmento a mostrar como ejemplo (opcional).

    Returns:
        Lista de Document, uno por fragmento, conservando los metadatos originales.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_documents(documents)

    if verbose:
        print(f"Documentos originales (páginas): {len(documents)}")
        print(f"Fragmentos generados:            {len(chunks)}")
        if documents:
            print(f"Factor de expansión:             {len(chunks)/len(documents):.1f}x")

        if ejemplo is not None and 0 <= ejemplo < len(chunks):
            chunk = chunks[ejemplo]
            print()
            print(f"Ejemplo — Fragmento #{ejemplo}:")
            print(f"  Fuente: {chunk.metadata.get('source', '?')}")
            print(f"  Página: {chunk.metadata.get('page', '?')}")
            print(f"  Longitud: {len(chunk.page_content)} caracteres")
            print(f"  Contenido: {chunk.page_content}")

    return chunks


if __name__ == "__main__":
    documents = cargar_pdfs()
    crear_chunks(documents, ejemplo=20)
