# De las *tools* a un RAG con *tools*

Este proyecto enseña en tres pasos cómo un LLM puede usar herramientas (*tools*) y cómo aplicarlo a un asistente que responde preguntas sobre reglamentos institucionales en PDF con **RAG** (*Retrieval-Augmented Generation*):

| Paso | Archivo | Qué muestra |
|---|---|---|
| 1 | [tools.py](tools.py) | Qué es una *tool*, con un ejemplo sencillo: calcular una nota final. |
| 2 | [rag_basico.py](rag_basico.py) | RAG clásico: **siempre** busca en la base vectorial y mete el resultado en el prompt. |
| 3 | [rag_con_tools.py](rag_con_tools.py) | RAG con *tools*: la búsqueda es una **herramienta** y **el LLM decide** cuándo y cómo usarla. |

Después se amplía el uso del LLM en dos direcciones: información actualizada y entradas que no son texto.

| Paso | Archivo | Qué muestra |
|---|---|---|
| 4 | [web_search.py](web_search.py) | La **fecha de corte**: el modelo no conoce hechos recientes y responde con datos desactualizados. |
| 5 | [web_search_2.py](web_search_2.py) | La solución: una *tool* de **búsqueda web** (Tavily) para responder sobre hechos actuales. |
| 6 | [multimodal.py](multimodal.py) | Entrada **multimodal**: el LLM recibe una imagen y la describe. |
| 7 | [audio_input.py](audio_input.py) | Entrada de **audio**: se graba la voz, se transcribe con Whisper y el LLM responde. |

---

## 1. Estructura del proyecto

```
RAG/
├── pdfs/                  # Documentos fuente (reglamentos)
├── imagenes/              # Imágenes .png para multimodal.py
├── db/                    # Base vectorial ChromaDB (se genera automáticamente)
├── utils/
│   ├── load_documents.py  # cargar_pdfs()              → lee los PDFs (un Document por página)
│   ├── chunking.py        # crear_chunks()             → divide las páginas en fragmentos
│   ├── embeddings.py      # cargar_modelo_embeddings() → modelo de embeddings (Hugging Face)
│   ├── store_vector.py    # crear_vector_store()       → indexa los chunks en ChromaDB
│   ├── retrieval.py       # cargar_vector_store(), recuperar_chunks() → búsqueda semántica
│   └── load_model.py      # cargar_llm()               → LLM de Groq
├── tools.py               # Paso 1: introducción a las tools
├── rag_basico.py          # Paso 2: RAG sin tools
├── rag_con_tools.py       # Paso 3: RAG con tools
├── web_search.py          # Paso 4: limitación de la fecha de corte
├── web_search_2.py        # Paso 5: tool de búsqueda web
├── multimodal.py          # Paso 6: entrada de imagen
├── audio_input.py         # Paso 7: entrada de audio
└── .env                   # Configuración y claves (no se sube a git)
```

Los dos scripts de RAG comparten las mismas piezas de `utils/`. **Solo cambia la forma de conectar la búsqueda con el LLM.**

---

## 2. ¿Qué es una *tool*? Ejemplo con `tools.py`

Un LLM solo genera texto: no puede ejecutar código, hacer cálculos exactos ni consultar una base de datos. Una *tool* (herramienta) es una **función de Python que el LLM puede pedir que se ejecute**.

El reparto de trabajo es este:
- **El LLM decide** qué herramienta usar y con qué argumentos.
- **Tu programa la ejecuta** y le devuelve el resultado.

### 2.1 Definir la tool

Se crea con el decorador `@tool` de LangChain:

```python
from langchain_core.tools import tool

@tool("nota_final_kl")
def nota_final_kl(parcial1: float, parcial2: float) -> float:
    """Calcula la nota final de un estudiante según la fórmula de la Konrad Lorenz:
    40% del primer parcial y 60% del segundo parcial.

    Args:
        parcial1: nota del primer parcial (de 0 a 50).
        parcial2: nota del segundo parcial (de 0 a 50).
    """
    return parcial1 * 0.4 + parcial2 * 0.6
```

El LLM **nunca ve el código** de la función. Solo recibe esta descripción, que LangChain genera a partir del nombre, el docstring y los parámetros:

```json
{
  "title": "nota_final_kl",
  "description": "Calcula la nota final de un estudiante según la fórmula de la Konrad Lorenz: 40% del primer parcial y 60% del segundo parcial. ...",
  "properties": {
    "parcial1": {"type": "number"},
    "parcial2": {"type": "number"}
  },
  "required": ["parcial1", "parcial2"]
}
```

De ahí salen tres reglas para escribir buenas tools:

1. **El docstring es la instrucción de uso.** Es lo único que el modelo lee para decidir si la herramienta le sirve, así que debe decir con claridad **qué hace** y **cuándo usarla**.
2. **Los parámetros deben tener nombres descriptivos.** Con `parcial1` y `parcial2`, el LLM sabe dónde va cada nota. Con nombres como `x` e `y` tendría que adivinarlo.
3. **El nombre que ve el LLM es el del decorador**, no el de la variable de Python. Si se escribe `@tool("nota_final_kl")` sobre `def tool1(...)`, para el modelo la herramienta se llama `nota_final_kl` y `tool1` no existe. Pedirle *"usa la tool1"* lo confunde. Lo más sencillo es usar el mismo nombre en los dos sitios, o no nombrar la tool en la pregunta y dejar que el LLM la elija por su descripción.

### 2.2 Probar la tool sin LLM

Una tool sigue siendo una función normal. Antes de dársela al LLM, conviene probarla directamente con `invoke`:

```python
nota_final = nota_final_kl.invoke({"parcial1": 3.5, "parcial2": 4.0})
print(f"Nota final: {nota_final}")   # Nota final: 3.8
```

### 2.3 Dársela al LLM con `bind_tools`

`bind_tools` envía al LLM la descripción de las herramientas disponibles:

```python
llm_model_con_tools = cargar_llm().bind_tools([nota_final_kl])
```

### 2.4 El ciclo completo: petición, ejecución y respuesta

`tools.py` sigue tres pasos:

```python
# 1. Enviar la pregunta: el LLM responde pidiendo la tool (tool_calls)
mensajes = [HumanMessage(pregunta_tool)]
respuesta_tool = llm_model_con_tools.invoke(mensajes)
mensajes.append(respuesta_tool)  # la petición del LLM debe quedar en el historial

# 2. Ejecutar cada tool pedida y devolver el resultado con un ToolMessage
for tool_call in respuesta_tool.tool_calls:
    resultado = nota_final_kl.invoke(tool_call["args"])
    mensajes.append(ToolMessage(content=str(resultado), tool_call_id=tool_call["id"]))

# 3. Volver a llamar al LLM con el resultado para que redacte la respuesta final
respuesta_final = llm_model_con_tools.invoke(mensajes)
```

**Paso 1: el LLM pide la herramienta.** Cuando decide usar una tool, el LLM no responde con texto: `content` llega vacío y la petición viene en el atributo **`tool_calls`**, que es una lista:

```python
respuesta_tool.tool_calls
# [{'name': 'nota_final_kl',                        # qué tool quiere usar
#   'args': {'parcial1': 3.5, 'parcial2': 4},       # con qué argumentos
#   'id':   'fc_6854174e-...'}]                     # identificador de la petición
```

Fíjate en que **el LLM extrajo los argumentos de la pregunta**. El usuario escribió *"obtuvo 3.5 en el primer parcial y 4.0 en el segundo"*, y el modelo lo convirtió en `parcial1=3.5` y `parcial2=4`.

**Paso 2: el programa ejecuta la tool y devuelve el resultado.** El resultado se envía en un **`ToolMessage`**, con dos condiciones:
- Debe llevar **el mismo `id`** de la petición (`tool_call_id`), para que el LLM sepa a qué petición responde.
- Su contenido debe ser **texto**. Por eso el código usa `str(resultado)`.

**Paso 3: el LLM redacta la respuesta.** Ahora el historial contiene la pregunta, la petición y el resultado, y con eso el LLM escribe la respuesta final.

Salida real de `python tools.py`:

```
Petición del LLM:     [{'name': 'nota_final_kl', 'args': {'parcial1': 3.5, 'parcial2': 4}, 'id': 'fc_6854...'}]
Resultado de la tool: 3.8
Respuesta final:      Para calcular la nota final según la fórmula de la Konrad Lorenz
                      (40 % del primer parcial y 60 % del segundo parcial): ... = 1.4 + 2.4 = 3.8
```

> **¿Por qué no deja que el LLM haga la cuenta?** Los LLM pueden equivocarse en cálculos y no conocen las reglas propias de una institución. Con la tool, la fórmula la aplica el código, que siempre da el resultado exacto, y el LLM solo se encarga de entender la pregunta y redactar la respuesta.

### 2.5 La idea clave

```
pregunta ──► LLM ──► tool_calls ──► tu código ejecuta la tool ──► ToolMessage ──► LLM ──► respuesta
```

Este mismo patrón sirve para cualquier herramienta. En la sección 4 se aplica a la búsqueda en los reglamentos.

---

## 3. RAG sin tools: `rag_basico.py`

### Cómo funciona

Para **cada** pregunta, el programa sigue siempre los mismos pasos:

```
pregunta ──► recuperar_chunks(pregunta, k=5) ──► contexto ──► PROMPT_TEMPLATE ──► LLM ──► respuesta
```

```python
resultados   = recuperar_chunks(vector_store, pregunta, k=5)          # 1. buscar (siempre)
contexto     = context(resultados)                                    # 2. unir los chunks
prompt_final = PROMPT_TEMPLATE.format(context=contexto, question=pregunta)  # 3. armar el prompt
respuesta    = llm_model.invoke(prompt_final)                         # 4. preguntar al LLM
```

El LLM no toma ninguna decisión: recibe un texto ya armado y lo completa. **El flujo lo decide el programador, no el modelo.**

### Desventajas

1. **Busca siempre, aunque no haga falta.**
   Si el usuario escribe *"Hola"* o *"Gracias"*, igualmente se recuperan 5 chunks y se envían al LLM. Esto gasta *tokens* y tiempo, y el modelo puede responder un saludo con información de reglamentos que nadie pidió.

2. **La búsqueda usa la pregunta literal.**
   La consulta a la base vectorial es exactamente lo que escribió el usuario, con muletillas, errores o redacción ambigua incluidos. Por ejemplo, *"Oye, ¿qué me pasa si llego tarde varias veces?"* se busca tal cual, cuando una consulta como *"sanciones por retrasos reiterados"* encontraría mejores fragmentos.

3. **Una sola búsqueda por pregunta.**
   Si la pregunta tiene varias partes (*"¿Cuáles son los deberes del trabajador y qué sanciones hay si no los cumple?"*), los 5 chunks recuperados con una sola búsqueda tienen que cubrir los dos temas, y suelen quedarse cortos en uno de ellos.

4. **No hay segunda oportunidad.**
   Si los chunks recuperados no sirven, el LLM no puede pedir más información. Solo le queda responder con lo que tiene o decir que no encontró nada.

5. **Difícil de ampliar.**
   Para añadir otra capacidad, como calcular una nota con `nota_final_kl`, habría que programar a mano un `if` que decida cuándo usarla según la pregunta.

6. **Reindexa en cada ejecución.**
   Cada vez que se arranca, vuelve a leer todos los PDFs y recrea la base vectorial (`reset=True`), aunque nada haya cambiado. Esta desventaja no tiene que ver con las *tools*, pero también se corrige en la nueva versión.

---

## 4. Conectar las tools con el RAG: `rag_con_tools.py`

### La búsqueda se convierte en una tool

Es el mismo patrón de `nota_final_kl`. La diferencia es que, en lugar de una fórmula, la función busca en la base vectorial:

```python
@tool
def buscar_reglamentos(consulta: str) -> str:
    """Busca en los reglamentos institucionales fragmentos relevantes para la consulta.
    Úsala siempre que la pregunta trate sobre normas, derechos, deberes, sanciones o procedimientos."""
    resultados = recuperar_chunks(vector_store, consulta, k=5, verbose=False)
    return context(resultados)
```

| | `tools.py` | `rag_con_tools.py` |
|---|---|---|
| Tool | `nota_final_kl(parcial1, parcial2)` | `buscar_reglamentos(consulta)` |
| Qué hace | Aplica una fórmula | Busca chunks en ChromaDB |
| Argumentos que elige el LLM | Las notas, extraídas de la pregunta | El texto de búsqueda, redactado por el propio LLM |
| Resultado | Un número | Los fragmentos con su fuente y página |

Las funciones `recuperar_chunks()` y `context()` son **las mismas** que en `rag_basico.py`. La búsqueda no cambia; lo que cambia es **quién decide cuándo hacerla y con qué texto**.

Hay un detalle técnico: la tool se define dentro de la función `crear_tools(vector_store)`, para que tenga acceso a la base vectorial sin usar variables globales.

### Del ciclo de una ronda al bucle

`tools.py` hace **una sola ronda**: petición, ejecución y respuesta. En un RAG, el LLM puede necesitar varias rondas, por ejemplo para buscar dos temas por separado o volver a buscar si el primer resultado no le sirvió. Por eso `rag_con_tools.py` repite los mismos tres pasos dentro de un bucle, en la función `responder()`:

```python
mensajes = [SystemMessage(SYSTEM_PROMPT), HumanMessage(pregunta)]

for _ in range(MAX_ITERACIONES):
    respuesta = llm_con_tools.invoke(mensajes)
    mensajes.append(respuesta)                        # guardar la petición del LLM en el historial

    if not respuesta.tool_calls:                      # sin peticiones → respuesta final
        return respuesta.content

    for tool_call in respuesta.tool_calls:            # puede pedir varias tools en la misma ronda
        herramienta = tools_por_nombre[tool_call["name"]]
        resultado = herramienta.invoke(tool_call["args"])
        mensajes.append(ToolMessage(content=resultado, tool_call_id=tool_call["id"]))
```

```
pregunta ──► LLM ──┬──► ¿pide una tool? NO ──► respuesta final
                   │
                   └──► SÍ ──► el programa ejecuta la tool ──► ToolMessage ──► LLM ──► (se repite)
```

Diferencias con `tools.py`:

- **La condición de salida es `tool_calls` vacío.** Si el LLM no pide nada, su mensaje es la respuesta final. Así, un saludo se responde en la primera ronda sin buscar.
- **`tools_por_nombre`** es un diccionario `{nombre: tool}` para ejecutar la herramienta que pida el LLM. En `tools.py` había una sola y se llamaba directamente.
- **`MAX_ITERACIONES`** (5) permite varias rondas, pero evita bucles infinitos si el modelo se queda pidiendo herramientas.
- **Hay un `SystemMessage`** con las reglas del asistente: usar la herramienta para las preguntas sobre reglamentos, responder solo con lo recuperado y citar la fuente y la página. A diferencia de `rag_basico.py`, el prompt no lleva `{context}`, porque el contexto llega en el `ToolMessage` y solo cuando el LLM lo pide.

### Ejemplo real de ejecución

```
Pregunta (o 'salir' para terminar): Hola, ¿cómo estás?

Respuesta: ¡Hola! Estoy bien, gracias. ¿En qué puedo ayudarte hoy?

Pregunta (o 'salir' para terminar): ¿Cuáles son las sanciones por llegar tarde?
  [tool] buscar_reglamentos({'consulta': 'sanciones por llegar tarde'})

Respuesta: Las sanciones por llegar tarde (retrasos sin excusa suficiente) están
establecidas en el Reglamento Interno de Trabajo:
1. Primera falta – Llamado de atención por escrito con copia a la hoja de vida. (Fuente: ...)
2. Segunda falta – Suspensión en el trabajo hasta por dos días, sin remuneración. (Fuente: ...)
...
```

- En el saludo, el LLM **no usó la herramienta**.
- En la pregunta sobre sanciones, el LLM **redactó su propia consulta**: *"sanciones por llegar tarde"*.

La línea `[tool] ...` se imprime en consola para poder ver cuándo y con qué argumentos decide buscar el modelo.

---

## 5. Comparación: qué se corrige con las tools

| Desventaja de `rag_basico.py` | Cómo se resuelve en `rag_con_tools.py` |
|---|---|
| Busca siempre, incluso en un saludo. | El LLM decide si necesita buscar. Un *"Hola"* se responde sin consultar la base. |
| Busca con la pregunta literal del usuario. | El LLM **reformula** la consulta (el argumento `consulta`) para encontrar mejores fragmentos. |
| Una sola búsqueda por pregunta. | El LLM puede llamar a la herramienta **varias veces**, por ejemplo una por cada parte de una pregunta compuesta. |
| Sin segunda oportunidad si los chunks no sirven. | Si el resultado no le basta, el LLM puede **volver a buscar** con otras palabras (hasta `MAX_ITERACIONES`). |
| Ampliarlo exige programar `if` a mano. | Basta con escribir otra función `@tool` y añadirla a la lista de `crear_tools()`. El LLM decide cuándo usarla. |
| Reindexa los PDFs en cada ejecución. | Abre la base existente con `cargar_vector_store()` y solo la crea si la carpeta `db/` no existe. |

### Lo que cuesta usar tools

Las tools no son gratis. Conviene conocer sus inconvenientes:

- **Más llamadas al LLM.** Una pregunta que necesita búsqueda hace **al menos 2 llamadas**: una para pedir la tool y otra para responder. El RAG básico hace solo 1. Esto supone más latencia y más *tokens*.
- **El modelo puede equivocarse al decidir.** Podría no buscar cuando debería y responder con su conocimiento general. Por eso el `SYSTEM_PROMPT` insiste en responder **únicamente** con la información de la herramienta.
- **Necesita un modelo con soporte de *tool calling*.** No todos los LLM lo tienen. El modelo usado aquí, `openai/gpt-oss-20b` en Groq, sí lo tiene.

---

## 6. Cómo ampliarlo con otra tool

Como el bucle de `responder()` ejecuta cualquier herramienta por su nombre, añadir una nueva solo requiere definirla dentro de `crear_tools()` e incluirla en la lista. Por ejemplo, una herramienta para saber qué documentos hay en la base:

```python
@tool
def listar_documentos() -> str:
    """Devuelve la lista de reglamentos disponibles en la base de conocimientos.
    Úsala cuando el usuario pregunte qué documentos o reglamentos se pueden consultar."""
    fuentes = {m["source"] for m in vector_store.get(include=["metadatas"])["metadatas"]}
    return "\n".join(sorted(fuentes))

return [buscar_reglamentos, listar_documentos]
```

También se podría añadir `nota_final_kl` de `tools.py`. El asistente buscaría en los reglamentos cuando le pregunten por normas y calcularía la nota cuando le den las calificaciones de los parciales. Como el contenido de un `ToolMessage` debe ser texto, en ese caso la tool debería devolver `str` o el bucle convertir el resultado con `str(resultado)`.

---

## 7. La fecha de corte: `web_search.py`

Un LLM solo sabe lo que vio durante su entrenamiento. Esa información termina en una **fecha de corte** (*knowledge cutoff*). Si se le pregunta por algo posterior, el modelo no puede saberlo, pero casi nunca lo dice: responde con lo último que conoce como si fuera actual.

`web_search.py` hace una pregunta sobre un hecho actual y muestra el razonamiento del modelo, su respuesta y los tokens usados:

```python
llm_model = cargar_llm(max_tokens=4096, reasoning_effort="low")

respuesta = llm_model.invoke("¿Quién es el presidente de Colombia?")

print(respuesta.additional_kwargs.get("reasoning_content"))   # razonamiento del modelo
print(respuesta.content)                                        # respuesta final
print(respuesta.response_metadata.get("finish_reason"))         # por qué terminó
```

Salida real (27 de octubre de 2026):

```
Razonamiento del modelo:
Need current president. As of 2026, president is Gustavo Petro? Actually Petro was president
2022-2026. In 2026, new president likely Gustavo Petro still until 2026? ... So answer: Gustavo Petro.

Respuesta: El presidente de Colombia es **Gustavo Petro**.
```

En su propio razonamiento, el modelo reconoce que el mandato de Petro termina en 2026. Aun así responde con el último dato que conoce y no advierte que puede estar desactualizado.

### El razonamiento y los tokens

`openai/gpt-oss-20b` es un **modelo de razonamiento**: antes de responder, "piensa" en un texto aparte. LangChain lo guarda en `respuesta.additional_kwargs["reasoning_content"]`, no en `content`. `respuesta` es un `AIMessage`.

Ese razonamiento también consume *tokens* de salida.  Por eso `cargar_llm` recibe dos parámetros adicionales:

- `max_tokens=4096`: deja margen para la respuesta.
- `reasoning_effort="low"`: le pide al modelo razonar menos.

---

## 8. Búsqueda web como tool: `web_search_2.py`

La solución a la fecha de corte es darle al modelo una herramienta para consultar información actual. Es el mismo patrón de las secciones 2 y 4: una función `@tool`, `bind_tools` y el ciclo de petición, ejecución y respuesta.

### La tool

La búsqueda la hace [Tavily](https://tavily.com), un buscador pensado para LLM. Devuelve los resultados ya resumidos (`include_answer='basic'`) junto con las fuentes:

```python
from tavily import TavilyClient

tavily_client = TavilyClient()   # lee TAVILY_API_KEY del .env

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(
        query=query,
        include_answer='basic',
        max_results=5,
        start_date="2026-01-27",
        end_date="2026-10-27"
    )
```

Igual que con `nota_final_kl`, conviene probar la tool sola antes de dársela al LLM:

```python
respuesta = web_search.invoke("¿Quién ganó el balón de oro en 2026?")
print(respuesta.get("answer"))
```

### Dos errores comunes

**1. Solo usar `bind_tools`.** `bind_tools` le dice al modelo que la herramienta existe, pero no la ejecuta. Si el programa solo llama a `invoke` una vez, el LLM pide la búsqueda en `tool_calls` y nadie la hace. Hay que completar los tres pasos:

```python
# 1. Enviar la pregunta: el LLM responde pidiendo la tool (tool_calls)
mensajes = [SystemMessage(SYSTEM_PROMPT), HumanMessage(pregunta)]
respuesta = llm_model.invoke(mensajes)
mensajes.append(respuesta)

# 2. Ejecutar cada tool pedida y devolver el resultado con un ToolMessage
for tool_call in respuesta.tool_calls:
    resultado = web_search.invoke(tool_call["args"])
    mensajes.append(ToolMessage(content=str(resultado), tool_call_id=tool_call["id"]))

# 3. Volver a llamar al LLM con el resultado para que redacte la respuesta final
respuesta_final = llm_model.invoke(mensajes)
```

**2. No decirle al modelo que busque.** El modelo "cree" saber quién es el presidente, así que puede responder directamente sin usar la herramienta. El `SystemMessage` le da la fecha de hoy y le indica que use la búsqueda para hechos actuales:

```python
SYSTEM_PROMPT = (
    f"Hoy es {date.today()}. Tu conocimiento tiene una fecha de corte anterior, "
    "así que para cualquier pregunta sobre hechos actuales o recientes usa siempre "
    "la herramienta web_search antes de responder."
)
```

### Resultado

Salida real con la misma pregunta de la sección 7:

```
Petición del LLM: [{'name': 'web_search', 'args': {'query': 'presidente de Colombia 2026'}, ...}]
Resultado de la tool: According to the sources, Abelardo de la Espriella is the president of
                      Colombia as of 2026. He was inaugurated on August 7, 2026, ...

Respuesta: El presidente de Colombia es **Abelardo de la Espriella**.
Fue proclamado presidente electo el 24 de junio de 2026 y asumió el cargo el 7 de agosto de 2026,
sucediendo a Gustavo Petro. Su mandato cubre el período 2026-2030.
```

Fíjate en que el LLM **redactó su propia consulta** (*"presidente de Colombia 2026"*), igual que con `buscar_reglamentos`.

| | `web_search.py` (sin tool) | `web_search_2.py` (con tool) |
|---|---|---|
| Respuesta | Gustavo Petro (desactualizada) | Abelardo de la Espriella (correcta) |
| Llamadas al LLM | 1 | 2 |
| Tokens de entrada | 79 | 3577 |

La búsqueda da la respuesta correcta, pero cuesta más: el resultado completo de Tavily entra al contexto del modelo.

> **Buscar en los reglamentos y buscar en la web es el mismo patrón.** En la sección 4 la tool consulta una base vectorial con documentos propios; aquí consulta internet. En los dos casos, el LLM decide cuándo buscar y con qué texto.

---

## 9. Entrada multimodal (imagen): `multimodal.py`

Un modelo **multimodal** puede recibir otros tipos de datos además de texto. `multimodal.py` envía una imagen y le pide al modelo que la analice.

### Elegir la imagen

El script lista las imágenes `.png` de la carpeta `imagenes/` y pide el número de la que se quiere usar:

```python
CARPETA_IMAGENES = Path(__file__).parent / "imagenes"

imagenes = sorted(CARPETA_IMAGENES.glob("*.png"))
for i, imagen in enumerate(imagenes, start=1):
    print(f"{i}. {imagen.name}")

ruta = imagenes[int(input("Número de la imagen: ")) - 1]
imagen_bytes = ruta.read_bytes()
```


### Enviar la imagen

La imagen se convierte a **base64** (texto) y se envía en el mismo `HumanMessage` que la pregunta. En lugar de un texto, `content` es una **lista de bloques**: uno de texto y uno de imagen.

```python
img_b64 = base64.b64encode(imagen_bytes).decode("utf-8")

llm_model = cargar_llm(model_name="qwen/qwen3.8-27b", max_tokens=4096, reasoning_effort="low")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "¿Que alimentos hay en esta imagen?"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

response = llm_model.invoke([multimodal_question])
print(response.content)
```

Con una foto del interior de una nevera, el modelo responde:

```
La imagen muestra el interior de un **refrigerador (nevera) muy ordenado y bien surtido**, ...
```

### Errores comunes

- **No todos los modelos ven imágenes.** `openai/gpt-oss-20b` solo recibe texto. Por eso este script usa un modelo con visión, `qwen/qwen3.8-27b`, pasándolo en `model_name` sin cambiar el `.env`.

- **La respuesta es un `AIMessage`**: el texto se lee con `response.content`.

---

## 10. Entrada de audio: `audio_input.py`

`audio_input.py` graba la voz del usuario, la reproduce y le pide al LLM que responda a lo que se dijo.

### Grabar y reproducir

La librería `sounddevice` usa el micrófono y los altavoces del computador:

```python
duration = 5        # segundos
sample_rate = 44100 # muestras por segundo

audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()                                   # esperar a que termine la grabación

sd.play(audio, samplerate=sample_rate)      # reproducir lo grabado
sd.wait()
```

`sd.rec` devuelve el audio como un arreglo de NumPy. Para enviarlo a un modelo, se convierte a un archivo **WAV en memoria**, sin guardarlo en disco:

```python
buf = io.BytesIO()
write(buf, sample_rate, audio)
wav_bytes = buf.getvalue()
```

> En Windows, el permiso del micrófono debe estar activado en *Configuración → Privacidad y seguridad → Micrófono*, incluido *"Permitir que las aplicaciones de escritorio accedan al micrófono"*. Si no lo está, se graba silencio sin ningún error.

### Del audio al texto con Whisper

Los modelos de chat de Groq aceptan texto e imágenes, pero **no audio**. Enviar el audio como un bloque `"image"` da `invalid image data`, y como bloque `"audio"` Groq también lo rechaza.

La solución es encadenar dos modelos:

```
voz ──► micrófono ──► WAV ──► Whisper (voz a texto) ──► transcripción ──► LLM ──► respuesta
```

```python
from groq import Groq

transcripcion = Groq().audio.transcriptions.create(
    file=("grabacion.wav", wav_bytes),
    model="whisper-large-v3-turbo",
    language="es",
)
print(f"Transcripción: {transcripcion.text}")

response = llm_model.invoke([HumanMessage(f"{transcripcion.text}")])
print(response.content)
```

- **Whisper** es un modelo especializado en transcribir voz. Groq lo ofrece con la misma `GROQ_API_KEY`.
- La librería `groq` ya viene instalada como dependencia de `langchain-groq`.

> **"Multimodal" depende del modelo.** `qwen/qwen3.8-27b` acepta imágenes pero no audio. Cuando un modelo no admite un tipo de dato, se puede encadenar con otro modelo que lo convierta a algo que sí acepte: en este caso, voz a texto.

---

## 11. Cómo ejecutarlo

### Requisitos

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```


### Archivo `.env`

```
GROQ_API_KEY=gsk_...                       # clave de https://console.groq.com
LLM_MODEL=openai/gpt-oss-20b               # modelo de Groq (debe admitir tool calling)
TAVILY_API_KEY=tvly-...                    # clave de https://tavily.com (para web_search_2.py)
EMBEDDINGS_MODEL=BAAI/bge-base-en-v1.5     # modelo de embeddings de Hugging Face
PERSIST_DIR=./db                           # carpeta de la base vectorial
HF_TOKEN=hf_...                            # opcional: descargas más rápidas desde Hugging Face
```

El `.env` contiene claves privadas y está excluido en el `.gitignore`. **No lo subas al repositorio.**

### Ejecutar

```bash
python tools.py            # paso 1: introducción a las tools
python rag_basico.py       # paso 2: RAG sin tools
python rag_con_tools.py    # paso 3: RAG con tools
python web_search.py       # paso 4: limitación de la fecha de corte
python web_search_2.py     # paso 5: búsqueda web como tool
python multimodal.py       # paso 6: análisis de una imagen de imagenes/
python audio_input.py      # paso 7: grabar, transcribir y responder
```

En los dos scripts de RAG, escribe `salir` para terminar.

> **Si cambias los PDFs o el modelo de embeddings**, borra la carpeta `db/` para que `rag_con_tools.py` vuelva a crear la base vectorial. Una base creada con un modelo de embeddings no sirve para otro.

