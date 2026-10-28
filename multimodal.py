from pathlib import Path
from utils.load_model import cargar_llm
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import base64

load_dotenv()

CARPETA_IMAGENES = Path(__file__).parent / "imagenes"

imagenes = sorted(CARPETA_IMAGENES.glob("*.png"))
if not imagenes:
    raise FileNotFoundError(f"No hay imágenes .png en {CARPETA_IMAGENES}")

for i, imagen in enumerate(imagenes, start=1):
    print(f"{i}. {imagen.name}")

ruta = imagenes[int(input("Número de la imagen: ")) - 1]

imagen_bytes = ruta.read_bytes()

# Now base64 encode
img_b64 = base64.b64encode(imagen_bytes).decode("utf-8")


llm_model = cargar_llm(model_name="qwen/qwen3.8-27b",  max_tokens=4096, reasoning_effort="low")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "¿Que alimentos hay en esta imagen?"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = llm_model.invoke([multimodal_question])

print(response.content)
