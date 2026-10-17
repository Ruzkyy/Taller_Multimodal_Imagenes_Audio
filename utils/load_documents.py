import os
from langchain_community.document_loaders import PyPDFLoader


def cargar_pdfs(pdf_dir="pdfs", verbose=True):
    """Carga todas las páginas de los PDFs de un directorio.

    Args:
        pdf_dir: ruta del directorio que contiene los PDFs.
        verbose: si es True, imprime el progreso de la carga.

    Returns:
        Lista de objetos Document (una entrada por página).
    """
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"No existe el directorio \"{pdf_dir}\"")

    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))

    if verbose:
        print(f"PDFs disponibles en \"{pdf_dir}/\" ({len(pdf_files)} archivos):")
        for i, f in enumerate(pdf_files, 1):
            size_kb = os.path.getsize(os.path.join(pdf_dir, f)) // 1024
            print(f"  [{i}] {f}  ({size_kb} KB)")

    documents = []

    for pdf_file in pdf_files:
        path = os.path.join(pdf_dir, pdf_file)
        loader = PyPDFLoader(path)
        pages = loader.load()
        documents.extend(pages)
        if verbose:
            print(f"  {pdf_file}: {len(pages)} páginas cargadas")

    if verbose:
        print(f"\nTotal de páginas cargadas: {len(documents)}")

    return documents
