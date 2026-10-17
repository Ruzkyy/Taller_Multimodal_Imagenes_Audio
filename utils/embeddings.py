from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from transformers.utils import logging as hf_logging
import os
import numpy as np
# Ocultar la barra "Loading weights" y los avisos de transformers
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()

load_dotenv()


MODELO_EMBEDDINGS = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")

def cargar_modelo_embeddings(model_name=None, device="cpu"):
    """Carga un modelo de embeddings de Hugging Face a través de LangChain.

    Args:
        model_name: nombre del modelo en Hugging Face.
        device: "cpu" o "cuda" si hay GPU disponible.

    Returns:
        Objeto HuggingFaceEmbeddings con los métodos embed_documents y embed_query.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name or MODELO_EMBEDDINGS,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # BGE se usa con vectores normalizados
    )


if __name__ == "__main__":
    model = cargar_modelo_embeddings(model_name=MODELO_EMBEDDINGS)

    sentences = [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day",
    ]

    embeddings = np.array(model.embed_documents(sentences))
    print(f"Embeddings:\n{embeddings.shape}")

    scores = embeddings @ embeddings.T
    print(f"Similarity scores:\n{scores}")
