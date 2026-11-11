"""Consulta imágenes o evalúa extracción estructurada de capturas de Switch 2."""

import argparse
import base64
import json
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import GroqError
from jsonschema import Draft202012Validator
from langchain_core.messages import HumanMessage

from utils.load_model import cargar_llm

CARPETA_IMAGENES = Path(__file__).parent / "imagenes"
ARCHIVO_VERDAD_TERRENO = Path(__file__).parent / "ground_truth_switch_2.json"
MODELO_VISION_PREDETERMINADO = "qwen/qwen3.8-27b"
MAX_IMAGENES = 3
MAX_BYTES_SOLICITUD = 20_000_000
COMPLETAR = "COMPLETAR"
CAMPO_RESUMEN = "resumen_breve"
CAMPO_ESTADO_ANALISIS = "estado_analisis"
CAMPO_MOTIVO_NO_ANALIZABLE = "motivo_no_analizable"
PREGUNTA_PANTALLA_CARGA = "¿Qué juego hay en la imagen?"
PREGUNTA_TIENDA = "¿Qué información del juego se puede ver en la imagen?"
CAMPOS_PANTALLA_CARGA = ("titulo_juego",)
CAMPOS_TIENDA = (
    "titulo_juego",
    "precio",
    "moneda",
    "clasificacion_edad",
    "tamano_descarga_gb",
    "desarrollador_o_editor",
)

# Los campos que no sean visibles se omiten de la respuesta. El resumen breve
# describe el contenido visible y no forma parte de la comparación de aciertos.
ESQUEMA_EXTRACCION: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "tipo_captura": {
            "type": "string",
            "enum": ["tienda", "perfil", "menu_consola", "menu_juego", "otra"],
        },
        "titulo_juego": {"type": ["string", "null"]},
        "precio": {"type": ["number", "null"]},
        "moneda": {"type": ["string", "null"]},
        "clasificacion_edad": {"type": ["string", "null"]},
        "tamano_descarga_gb": {"type": ["number", "null"]},
        "desarrollador_o_editor": {"type": ["string", "null"]},
        "nombre_perfil": {"type": ["string", "null"]},
        "estado_perfil": {"type": ["string", "null"]},
        "amigos_en_linea": {"type": ["integer", "null"]},
        "historial_juegos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "titulo": {"type": "string"},
                    "tiempo_jugado": {"type": ["string", "null"]},
                },
                "required": ["titulo", "tiempo_jugado"],
                "additionalProperties": False,
            },
        },
        CAMPO_ESTADO_ANALISIS: {
            "type": "string",
            "enum": ["analizable", "no_analizable"],
            "description": "Indica si se puede leer información suficiente en la captura.",
        },
        CAMPO_MOTIVO_NO_ANALIZABLE: {
            "type": "string",
            "description": "Motivo de ilegibilidad; cadena vacía si es analizable.",
        },
        CAMPO_RESUMEN: {
            "type": "string",
            "minLength": 1,
            "description": "Resumen conciso del contenido visible en la captura.",
        },
    },
    "additionalProperties": False,
}
Draft202012Validator.check_schema(ESQUEMA_EXTRACCION)


def crear_esquema(campos: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Construye el esquema JSON del caso usando los campos definidos arriba."""
    campos_respuesta = list(
        dict.fromkeys(
            [
                *campos,
                CAMPO_ESTADO_ANALISIS,
                CAMPO_MOTIVO_NO_ANALIZABLE,
                CAMPO_RESUMEN,
            ]
        )
    )
    return {
        "$schema": ESQUEMA_EXTRACCION["$schema"],
        "type": "object",
        "properties": {
            campo: ESQUEMA_EXTRACCION["properties"][campo]
            for campo in campos_respuesta
        },
        "required": [
            CAMPO_ESTADO_ANALISIS,
            CAMPO_MOTIVO_NO_ANALIZABLE,
            CAMPO_RESUMEN,
        ],
        "additionalProperties": False,
    }


VALIDADOR_EXTRACCION = Draft202012Validator(ESQUEMA_EXTRACCION)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pregunta sobre una imagen o compara capturas de Nintendo Switch 2."
    )
    parser.add_argument(
        "--extraer",
        action="store_true",
        help="Extraer datos de 2 o 3 capturas en una solicitud y compararlos con la verdad terreno.",
    )
    parser.add_argument(
        "--imagenes",
        nargs="+",
        type=Path,
        metavar="IMAGEN",
        help="Dos o tres rutas para --extraer; si se omite, podrás elegirlas de la lista.",
    )
    parser.add_argument(
        "--verdad-terreno",
        type=Path,
        default=ARCHIVO_VERDAD_TERRENO,
        help="Ruta del JSON con los valores reales leídos de cada captura.",
    )
    parser.add_argument(
        "--modelo",
        default=None,
        help="Modelo de visión; por defecto usa VISION_MODEL o qwen/qwen3.8-27b.",
    )
    return parser.parse_args()


def obtener_mime_type(ruta: Path) -> str:
    tipos = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    try:
        return tipos[ruta.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"Formato no admitido para {ruta.name}; usa PNG o JPEG.") from error


def validar_imagenes(rutas: list[Path], *, maximo: int = MAX_IMAGENES) -> int:
    """Comprueba formato, existencia, cantidad y peso antes de enviar imágenes."""
    if not rutas:
        raise ValueError("Indica al menos una imagen.")
    if len(rutas) > maximo:
        raise ValueError(f"Se permiten como máximo {maximo} imágenes por solicitud.")

    total_bytes = 0
    for ruta in rutas:
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró {ruta}. Copia ahí la imagen o indica una ruta válida."
            )
        obtener_mime_type(ruta)
        total_bytes += ruta.stat().st_size
    if total_bytes > MAX_BYTES_SOLICITUD:
        raise ValueError(
            f"Las imágenes pesan {total_bytes:,} bytes; el máximo por solicitud "
            "es 20 MB."
        )
    return total_bytes


def listar_imagenes_disponibles() -> list[Path]:
    """Lista las imágenes admitidas para selección o análisis interactivo."""
    extensiones = {".png", ".jpg", ".jpeg"}
    return sorted(
        (
            ruta
            for ruta in CARPETA_IMAGENES.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in extensiones
        ),
        key=lambda ruta: ruta.name.casefold(),
    ) if CARPETA_IMAGENES.is_dir() else []


def seleccionar_imagenes_para_extraccion() -> list[Path] | None:
    """Muestra la lista y permite escoger dos o tres imágenes distintas."""
    disponibles = listar_imagenes_disponibles()
    if len(disponibles) < 2:
        print(
            f"Se necesitan al menos 2 imágenes PNG o JPEG en "
            f"{CARPETA_IMAGENES}; solo se encontraron {len(disponibles)}.",
            file=sys.stderr,
        )
        return None

    print("Imágenes disponibles para comparar:")
    for indice, ruta in enumerate(disponibles, start=1):
        print(f"{indice}. {ruta.name}")

    while True:
        try:
            seleccion = input(
                "Escribe los números de 2 o 3 imágenes relacionadas, separados "
                "por comas (por ejemplo: 1, 3 o 1, 3, 4): "
            )
        except (EOFError, KeyboardInterrupt):
            print("\nSelección cancelada.", file=sys.stderr)
            return None

        try:
            indices = [int(valor.strip()) for valor in seleccion.split(",")]
        except ValueError:
            print("Ingresa únicamente números separados por comas.")
            continue
        if not 2 <= len(indices) <= MAX_IMAGENES:
            print(f"Debes seleccionar 2 o {MAX_IMAGENES} imágenes.")
            continue
        if len(set(indices)) != len(indices):
            print("Selecciona imágenes diferentes.")
            continue
        if any(indice < 1 or indice > len(disponibles) for indice in indices):
            print(f"Elige números entre 1 y {len(disponibles)}.")
            continue
        return [disponibles[indice - 1] for indice in indices]


def crear_modelo(nombre_modelo: str | None) -> Any:
    """Carga el modelo de visión siguiendo el helper del repositorio."""
    import os

    modelo = nombre_modelo or os.getenv(
        "VISION_MODEL", MODELO_VISION_PREDETERMINADO
    )
    return cargar_llm(
        model_name=modelo,
        max_tokens=800,
        reasoning_effort="low",
    )


def preguntar_por_imagen(
    ruta: Path,
    pregunta: str,
    modelo: Any,
) -> str:
    """Envía una imagen y una pregunta como en el ejemplo original del taller."""
    imagen_base64 = base64.b64encode(ruta.read_bytes()).decode("utf-8")
    mensaje = HumanMessage(
        content=[
            {"type": "text", "text": pregunta},
            {
                "type": "image",
                "base64": imagen_base64,
                "mime_type": obtener_mime_type(ruta),
            },
        ]
    )
    respuesta = modelo.invoke([mensaje])
    return str(respuesta.content)


def preguntar_por_imagenes(
    rutas: list[Path],
    pregunta: str,
    modelo: Any,
) -> str:
    """Envía varias imágenes relacionadas dentro de un único HumanMessage."""
    contenido: list[dict[str, str]] = [{"type": "text", "text": pregunta}]
    for ruta in rutas:
        contenido.extend(
            [
                {"type": "text", "text": f"Imagen: {ruta.name}"},
                {
                    "type": "image",
                    "base64": base64.b64encode(ruta.read_bytes()).decode("utf-8"),
                    "mime_type": obtener_mime_type(ruta),
                },
            ]
        )
    respuesta = modelo.invoke([HumanMessage(content=contenido)])
    return str(respuesta.content)


def limpiar_json(texto: str) -> str:
    """Retira bloques Markdown opcionales de una respuesta JSON."""
    coincidencia = re.fullmatch(
        r"\s*```(?:json)?\s*(.*?)\s*```\s*",
        texto,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return coincidencia.group(1).strip() if coincidencia else texto.strip()


def parsear_y_validar_json(
    texto: str,
    esquema: dict[str, Any] = ESQUEMA_EXTRACCION,
) -> tuple[Any, str | None]:
    """Devuelve los datos extraídos o un mensaje claro de error de formato/esquema."""
    if not texto.strip():
        return None, (
            "El modelo devolvió una respuesta vacía. Vuelve a ejecutar la extracción "
            "para esta imagen o revisa la disponibilidad del modelo."
        )
    try:
        datos = json.loads(limpiar_json(texto))
    except json.JSONDecodeError as error:
        return None, f"JSON inválido: {error.msg} (posición {error.pos})."

    validador = (
        VALIDADOR_EXTRACCION
        if esquema is ESQUEMA_EXTRACCION
        else Draft202012Validator(esquema)
    )
    errores = sorted(
        validador.iter_errors(datos),
        key=lambda error: tuple(str(parte) for parte in error.absolute_path),
    )
    if errores:
        detalle = "; ".join(
            f"{'.'.join(str(parte) for parte in error.absolute_path) or '(objeto)'}: "
            f"{error.message}"
            for error in errores
        )
        return None, f"El JSON no cumple el esquema: {detalle}"
    datos_visibles = {
        campo: valor
        for campo, valor in datos.items()
        if campo in {CAMPO_ESTADO_ANALISIS, CAMPO_MOTIVO_NO_ANALIZABLE}
        or (valor is not None and valor != "" and valor != [])
    }
    return datos_visibles, None


def extraer_datos(
    ruta: Path,
    modelo: Any,
    pregunta: str = PREGUNTA_TIENDA,
    campos: tuple[str, ...] | list[str] | None = None,
) -> tuple[Any, str | None]:
    """Pide al modelo campos visibles y valida el JSON devuelto."""
    esquema = crear_esquema(campos or list(ESQUEMA_EXTRACCION["properties"]))
    instruccion = (
        f"{pregunta}\n\n"
        "Responde únicamente con un objeto JSON válido conforme al esquema. "
        "Evalúa primero si la pantalla y los datos solicitados se distinguen "
        "con suficiente claridad. Si la consola o el texto aparecen demasiado "
        "lejos, borrosos, tapados o ilegibles, usa estado_analisis "
        '"no_analizable", explica concretamente la causa en '
        "motivo_no_analizable y resumen_breve, y no extraigas ningún campo de "
        "contenido. Si la información relevante sí es legible, usa estado_analisis "
        '"analizable" y deja motivo_no_analizable como cadena vacía. '
        "Usa solo información legible en la captura. No deduzcas datos que no "
        "aparezcan. Si un campo no se ve, está vacío o no corresponde a esta "
        "imagen, omite por completo la propiedad: no escribas null, texto "
        "sustituto ni una lista vacía. Conserva el símbolo de moneda tal como "
        "aparece; no infieras la moneda por el país. El precio debe ser numérico. "
        "El tamaño de descarga solo se incluye si es visible; si aparece en MB, "
        "conviértelo a GB. Incluye siempre resumen_breve con una síntesis corta, "
        "de una o dos frases y específica de lo que muestra la captura. Si la "
        "captura no corresponde a la pregunta, omite los datos y explica brevemente "
        "eso en resumen_breve.\n"
        "Esquema JSON:\n"
        f"{json.dumps(esquema, ensure_ascii=False)}"
    )
    respuesta = preguntar_por_imagen(ruta, instruccion, modelo)
    datos, error = parsear_y_validar_json(respuesta, esquema)
    if error:
        print(f"  Respuesta del modelo: {respuesta}")
    if not error and datos[CAMPO_ESTADO_ANALISIS] == "no_analizable":
        campos_contenido = set(campos or ESQUEMA_EXTRACCION["properties"])
        campos_contenido.difference_update(
            {
                CAMPO_ESTADO_ANALISIS,
                CAMPO_MOTIVO_NO_ANALIZABLE,
                CAMPO_RESUMEN,
            }
        )
        datos = {
            campo: valor
            for campo, valor in datos.items()
            if campo not in campos_contenido
        }
    return datos, error


def extraer_datos_multiples(
    rutas: list[Path],
    modelo: Any,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Extrae todas las imágenes y genera una comparación conjunta en una llamada."""
    campos = [
        campo
        for campo in ESQUEMA_EXTRACCION["properties"]
        if campo
        not in {
            CAMPO_RESUMEN,
            CAMPO_ESTADO_ANALISIS,
            CAMPO_MOTIVO_NO_ANALIZABLE,
        }
    ]
    esquema_imagen = crear_esquema(campos)
    esquema = {
        "$schema": ESQUEMA_EXTRACCION["$schema"],
        "type": "object",
        "properties": {
            ruta.name: esquema_imagen for ruta in rutas
        }
        | {
            "resumen_conjunto": {
                "type": "string",
                "minLength": 1,
                "description": "Comparación breve y conjunta de las imágenes.",
            }
        },
        "required": [ruta.name for ruta in rutas] + ["resumen_conjunto"],
        "additionalProperties": False,
    }
    nombres = ", ".join(json.dumps(ruta.name, ensure_ascii=False) for ruta in rutas)
    instruccion = (
        f"Compara estas capturas de Switch 2: {nombres}. Devuelve solo JSON: "
        "una clave por imagen y resumen_conjunto. Cada imagen incluye "
        "estado_analisis (analizable/no_analizable), motivo_no_analizable, "
        "resumen_breve y solo campos visibles: tipo_captura, titulo_juego, "
        "precio, moneda, clasificacion_edad, tamano_descarga_gb, "
        "desarrollador_o_editor, nombre_perfil, estado_perfil, amigos_en_linea, "
        "historial_juegos. No inventes; omite campos ausentes. Precio numérico, "
        "moneda literal. Si está lejos, borrosa, tapada o ilegible, marca "
        "no_analizable, explica por qué y omite datos. resumen_conjunto compara "
        "similitudes y diferencias visibles."
    )
    respuesta = preguntar_por_imagenes(rutas, instruccion, modelo)
    datos, error = parsear_y_validar_json(respuesta, esquema)
    if error:
        print(f"Respuesta del modelo: {respuesta}")
        return None, None, error

    resumen_conjunto = datos.pop("resumen_conjunto")
    for ruta in rutas:
        imagen_datos = datos[ruta.name]
        if imagen_datos[CAMPO_ESTADO_ANALISIS] == "no_analizable":
            campos_contenido = set(campos)
            imagen_datos = {
                campo: valor
                for campo, valor in imagen_datos.items()
                if campo not in campos_contenido
            }
            datos[ruta.name] = imagen_datos
    return datos, resumen_conjunto, None


def es_completar(valor: Any) -> bool:
    return isinstance(valor, str) and valor.strip().casefold() == COMPLETAR.casefold()


def normalizar_texto(valor: str) -> str:
    texto = unicodedata.normalize("NFKD", valor).casefold()
    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    return " ".join(texto.split())


def normalizar_valor(valor: Any) -> Any:
    """Normaliza recursivamente texto y números para compararlos."""
    if isinstance(valor, str):
        return normalizar_texto(valor)
    if isinstance(valor, list):
        return [normalizar_valor(elemento) for elemento in valor]
    if isinstance(valor, dict):
        return {
            clave: normalizar_valor(elemento)
            for clave, elemento in sorted(valor.items())
        }
    if isinstance(valor, (int, float, Decimal)) and not isinstance(valor, bool):
        try:
            numero = Decimal(str(valor)).normalize()
        except InvalidOperation:
            return valor
        return numero if numero.is_finite() else valor
    return valor


def cargar_verdad_terreno(
    ruta: Path,
    imagenes: list[Path],
) -> dict[str, dict[str, Any]]:
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"No se encontró {ruta}; crea o completa el archivo de verdad terreno."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON de verdad terreno inválido en línea {error.lineno}, "
            f"columna {error.colno}: {error.msg}."
        ) from error

    if not isinstance(datos, dict):
        raise ValueError("La verdad terreno debe ser un objeto JSON por nombre de imagen.")
    for imagen in imagenes:
        valores_reales = datos.get(imagen.name)
        if valores_reales is None:
            valores_reales = {}
        elif not isinstance(valores_reales, dict):
            raise ValueError(
                f"La entrada de {imagen.name!r} debe ser un objeto JSON."
            )
        valores_reales = dict(valores_reales)
        for campo in ESQUEMA_EXTRACCION["properties"]:
            if campo in {
                CAMPO_RESUMEN,
                CAMPO_ESTADO_ANALISIS,
                CAMPO_MOTIVO_NO_ANALIZABLE,
            }:
                continue
            valor = valores_reales.get(campo, COMPLETAR)
            valores_reales[campo] = valor
            if es_completar(valor):
                continue
            validador = Draft202012Validator(
                ESQUEMA_EXTRACCION["properties"][campo]
            )
            error = next(validador.iter_errors(valor), None)
            if error:
                raise ValueError(
                    f"El valor real {imagen.name}.{campo} no coincide con el tipo "
                    f"esperado: {error.message}."
                )
        datos[imagen.name] = valores_reales
    return datos


def mostrar_comparacion(filas: list[dict[str, Any]]) -> None:
    encabezados = ["Imagen", "Campo", "Valor real", "Extraído", "Resultado"]
    columnas = [
        [
            fila["imagen"],
            fila["campo"],
            json.dumps(fila["valor_real"], ensure_ascii=False),
            json.dumps(fila["valor_extraido"], ensure_ascii=False),
            fila["resultado"],
        ]
        for fila in filas
    ]
    anchos = [
        min(max(len(encabezados[indice]), *(len(fila[indice]) for fila in columnas)), 46)
        for indice in range(len(encabezados))
    ]

    def formatear(fila: list[str]) -> str:
        celdas = []
        for indice, valor in enumerate(fila):
            if len(valor) > anchos[indice]:
                valor = valor[: anchos[indice] - 1] + "…"
            celdas.append(valor.ljust(anchos[indice]))
        return " | ".join(celdas)

    print("\nComparación campo por campo:")
    print(formatear(encabezados))
    print("-+-".join("-" * ancho for ancho in anchos))
    for fila in columnas:
        print(formatear(fila))


def mostrar_resumenes(
    imagenes: list[Path],
    valores_extraidos: dict[str, Any],
) -> None:
    """Imprime al final los resúmenes generados para las imágenes."""
    resumenes = [
        (imagen.name, valores_extraidos[imagen.name][CAMPO_RESUMEN])
        for imagen in imagenes
        if isinstance(valores_extraidos.get(imagen.name), dict)
        and valores_extraidos[imagen.name].get(CAMPO_RESUMEN)
    ]
    print("\nEn resumen:")
    if not resumenes:
        print("No se pudo generar un resumen a partir de las imágenes.")
        return
    for nombre, resumen in resumenes:
        print(f"- {nombre}: {resumen}")


def ejecutar_extraccion(args: argparse.Namespace) -> int:
    imagenes = args.imagenes
    if imagenes is None:
        imagenes = seleccionar_imagenes_para_extraccion()
        if imagenes is None:
            return 2
    if not 2 <= len(imagenes) <= MAX_IMAGENES:
        print("Debes proporcionar 2 o 3 imágenes para la comparación conjunta.", file=sys.stderr)
        return 2
    if len({imagen.resolve() for imagen in imagenes}) != len(imagenes):
        print("Selecciona imágenes diferentes para la comparación.", file=sys.stderr)
        return 2
    if len({imagen.name.casefold() for imagen in imagenes}) != len(imagenes):
        print(
            "Las imágenes deben tener nombres de archivo diferentes para "
            "identificarlas en la respuesta JSON.",
            file=sys.stderr,
        )
        return 2
    try:
        bytes_totales = validar_imagenes(imagenes)
        verdad_terreno = cargar_verdad_terreno(args.verdad_terreno, imagenes)
        modelo = crear_modelo(args.modelo)
    except (FileNotFoundError, ValueError, GroqError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print("Imágenes seleccionadas: " + ", ".join(imagen.name for imagen in imagenes))
    print(
        f"Enviando las {len(imagenes)} imágenes relacionadas en una sola solicitud..."
    )
    try:
        datos_multiples, resumen_conjunto, error_solicitud = (
            extraer_datos_multiples(imagenes, modelo)
        )
    except (GroqError, OSError, ValueError) as excepcion:
        datos_multiples = None
        resumen_conjunto = None
        error_solicitud = f"Error al procesar la solicitud conjunta: {excepcion}"

    if error_solicitud:
        print(f"\n{error_solicitud}", file=sys.stderr)
        print(
            "No se pudo completar la extracción; no se generó una tabla de "
            "aciertos. Prueba de nuevo cuando se restablezca el límite o "
            "selecciona 2 imágenes."
        )
        return 1

    comparaciones: list[dict[str, Any]] = []
    valores_extraidos: dict[str, Any] = {}
    metricas_pendientes: list[str] = []
    campos = [
        campo
        for campo in ESQUEMA_EXTRACCION["properties"]
        if campo
        not in {
            CAMPO_RESUMEN,
            CAMPO_ESTADO_ANALISIS,
            CAMPO_MOTIVO_NO_ANALIZABLE,
        }
    ]
    imagenes_no_analizables: list[tuple[str, str]] = []

    for imagen in imagenes:
        print(f"\nResultado para: {imagen.name}")
        extraidos = datos_multiples.get(imagen.name) if datos_multiples else None
        error = None

        valores_extraidos[imagen.name] = extraidos
        if error:
            print(f"  {error}")
        else:
            print("  JSON válido.")

        if (
            not error
            and extraidos[CAMPO_ESTADO_ANALISIS] == "no_analizable"
        ):
            motivo = extraidos[CAMPO_MOTIVO_NO_ANALIZABLE]
            print(f"  No se puede analizar ni comparar: {motivo}")
            imagenes_no_analizables.append((imagen.name, motivo))
            continue

        reales = verdad_terreno[imagen.name]
        for campo in campos:
            real = reales[campo]
            extraido = extraidos.get(campo) if extraidos is not None else None
            if es_completar(real):
                resultado = "PENDIENTE"
                metricas_pendientes.append(f"{imagen.name}.{campo}")
            elif error:
                resultado = "ERROR"
            else:
                resultado = (
                    "CORRECTO"
                    if normalizar_valor(real) == normalizar_valor(extraido)
                    else "INCORRECTO"
                )
            comparaciones.append(
                {
                    "imagen": imagen.name,
                    "campo": campo,
                    "valor_real": real,
                    "valor_extraido": extraido,
                    "resultado": resultado,
                }
            )

    mostrar_comparacion(comparaciones)
    if metricas_pendientes:
        print(
            "\nPorcentaje no calculado. Completa estos campos con lo que realmente "
            "se ve en la imagen; usa null cuando el dato no aparece: "
            + ", ".join(metricas_pendientes)
        )
    elif comparaciones:
        aciertos = sum(fila["resultado"] == "CORRECTO" for fila in comparaciones)
        total = len(comparaciones)
        porcentaje = 100 * aciertos / total
        print(f"\nAciertos: {aciertos} de {total} campos = {porcentaje:.2f} %.")
    else:
        print("\nPorcentaje no calculado: ninguna imagen fue legible para comparar.")

    print(f"Tamaño total: {bytes_totales:,} bytes (límite: 20 MB).")
    if imagenes_no_analizables:
        print("\nImágenes excluidas de la comparación:")
        for nombre, motivo in imagenes_no_analizables:
            print(f"- {nombre}: {motivo}")
    mostrar_resumenes(imagenes, valores_extraidos)
    if resumen_conjunto:
        print(f"\nComparación conjunta:\n{resumen_conjunto}")
    return 0


def ejecutar_pregunta(args: argparse.Namespace) -> int:
    imagenes = listar_imagenes_disponibles()
    if not imagenes:
        print(
            f"No hay imágenes PNG o JPEG en {CARPETA_IMAGENES}.",
            file=sys.stderr,
        )
        return 2

    for indice, imagen in enumerate(imagenes, start=1):
        print(f"{indice}. {imagen.name}")
    try:
        seleccion = int(input("Número de la imagen: "))
        if not 1 <= seleccion <= len(imagenes):
            raise ValueError
        ruta = imagenes[seleccion - 1]
        print("\nSelecciona el tipo de captura:")
        print(f"1. Pantalla de carga: {PREGUNTA_PANTALLA_CARGA}")
        print(f"2. Tienda de juegos: {PREGUNTA_TIENDA}")
        tipo_captura = input("Opción: ").strip()
        if tipo_captura == "1":
            pregunta = PREGUNTA_PANTALLA_CARGA
            campos = CAMPOS_PANTALLA_CARGA
        elif tipo_captura == "2":
            pregunta = PREGUNTA_TIENDA
            campos = CAMPOS_TIENDA
        else:
            print("Selecciona 1 para pantalla de carga o 2 para tienda.", file=sys.stderr)
            return 2
        validar_imagenes([ruta])
        modelo = crear_modelo(args.modelo)
        datos, error = extraer_datos(ruta, modelo, pregunta, campos)
    except ValueError as error:
        print(f"Selección o configuración inválida: {error}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nSelección cancelada.", file=sys.stderr)
        return 2
    except (GroqError, OSError) as error:
        print(f"Error al procesar la imagen: {error}", file=sys.stderr)
        return 1

    print(f"\nImagen: {ruta}")
    print(f"Pregunta: {pregunta}")
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if datos[CAMPO_ESTADO_ANALISIS] == "no_analizable":
        print(
            "\nNo se puede analizar esta imagen porque "
            f"{datos[CAMPO_MOTIVO_NO_ANALIZABLE]}"
        )
        print(f"En resumen, {datos[CAMPO_RESUMEN]}")
        return 0

    resumen = datos.pop(CAMPO_RESUMEN, "")
    datos.pop(CAMPO_ESTADO_ANALISIS, None)
    datos.pop(CAMPO_MOTIVO_NO_ANALIZABLE, None)
    print("JSON:")
    print(json.dumps(datos, ensure_ascii=False, indent=2))
    if not datos:
        print("No se encontró información legible para esta pregunta.")
    if resumen:
        print(f"\nEn resumen, {resumen}")
    return 0


def main() -> int:
    load_dotenv()
    args = parsear_argumentos()
    if args.extraer:
        return ejecutar_extraccion(args)
    if args.imagenes:
        print("--imagenes solo se usa junto con --extraer.", file=sys.stderr)
        return 2
    return ejecutar_pregunta(args)


if __name__ == "__main__":
    raise SystemExit(main())
