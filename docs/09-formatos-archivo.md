# 09 — Formatos de archivo

Referencia técnica de cada archivo persistente. Esta página existe para
que un humano (o un script de migración) pueda leer/escribir cualquiera
de estos archivos sin levantar la app.

## `.lab.md` (fuente única del lab)

Markdown con frontmatter YAML + bloques `::::cell{...}`. Detalle
completo en [02-autoria-lab-md.md](02-autoria-lab-md.md).

Path: `_TPS/sources/Laboratorio_<id>.lab.md`.

Esquema mínimo:

```markdown
---
lab: "3a"
title: "..."
---

::::cell{#header type=markdown role=header}
contenido
::::

::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
codigo_completo()
```
::::
```

## `.ipynb` (notebook compilado)

Formato estándar Jupyter (`nbformat: 4`, `nbformat_minor: 5`). El
compilador `lab_build.py` emite:

```json
{
  "cells": [
    {
      "cell_type": "markdown" | "code",
      "id": "ej1-code",
      "metadata": {},
      "source": ["línea 1\n", "línea 2\n"],
      "execution_count": null,           ← solo en code
      "outputs": []                      ← solo en code, vacío al compilar
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {"name": "python"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

Notas:

- **`source`**: list of strings con line endings explícitos (`\n` al
  final de cada línea salvo la última). Es lo que define el formato
  Jupyter; la app lo normaliza a un string al leer.
- **`outputs`**: vacío al compilar. Para que la rúbrica autogen pueda
  inferir `graded_outputs`, hay que abrir el notebook de solución en
  Jupyter/Colab y "Reiniciar y ejecutar todo" antes de generar la
  rúbrica.

## `.rubric.yaml` (rúbrica del lab)

YAML con la estructura de la rúbrica. Detalle de schemas v1 y v2 en
[03-rubricas.md](03-rubricas.md).

**Path canónico**:
- Generada externamente: `_TPS/rubricas/Laboratorio_<id>.rubric.yaml`.
- Generada por la app: `<workdir>/.corrector/rubrica.yaml`.

**Schema v2** (target):

```yaml
title: "Laboratorio 3b — Transferencia de estilo rápida"
ejercicios:
  - id: ej1
    titulo: "Matriz de Gram"
    enunciado_cell: ej1-enunciado
    items:
      - kind: code            # o "analysis"
        key: ej1-code         # único dentro del lab
        code_cell: ej1-code   # solo kind=code
        graded_outputs: [text]
        rubric:
          expected: "..."
          common_errors:
            - "CRÍTICO: ..."
            - "..."
      - kind: analysis
        key: ej1-analisis
        pregunta_cell: ej1-pregunta   # opcional
        answer_cell:   ej1-respuesta
        rubric:
          expected: "..."
          common_errors: ["..."]
```

**Schema v1** (legacy, lo que está en disco hoy):

```yaml
title: "Laboratorio 2"
ejercicios:
  - id: ej1
    titulo: "..."
    enunciado_cell: ej1-enunciado
    code_cell: ej1-code
    pregunta_cell: ej1-pregunta
    answer_cell: ej1-respuesta
    graded_outputs: [text]
    rubric:
      expected: "..."
      common_errors: [...]
```

`app/rubric.py::load_rubrica` detecta la ausencia de `items` y
sintetiza la lista en memoria, compartiendo la `rubric` de nivel
ejercicio entre el item code y el item analysis.

## `.corrector/config.json`

Estado interno de un workdir. Lo gestiona `app/workdir.py`.

Path: `<workdir>/.corrector/config.json`.

```json
{
  "title": "Laboratorio 2 — Convolucionales",
  "notebook_enunciado": "/Users/.../Laboratorio_2.ipynb",
  "notebook_solucion": "/Users/.../Laboratorio_2_Solucion.ipynb",
  "rubrica": "/Users/.../Laboratorio_2.rubric.yaml"
}
```

Reglas:

- **Paths absolutos** en los tres notebooks. La app valida que existan
  al cargar y muestra error si alguno fue movido o borrado.
- **Title** es solo cosmético — aparece en la barra de título de la
  matriz.
- Para editar a mano: editar el JSON con cualquier editor; la app lo
  re-lee al refresh.

## `~/.lab_corrector/recent.json`

Registry global de workdirs recientes. Lo gestiona `app/recents.py`.

```json
[
  "/Users/javiervelez/.../2026/lab2",
  "/Users/javiervelez/.../2025/lab3a",
  "/Users/javiervelez/.../2025/lab1c"
]
```

- Lista de strings (paths absolutos), ordenada por recencia (más
  reciente primero).
- Limitado a `MAX_RECENTS=10`. Los excedentes se descartan.
- `list_recents()` filtra los paths que ya no existen en disco — la
  lista en disco no se reescribe automáticamente; los que dejan de
  existir simplemente se omiten al mostrar.

Para "olvidar" todo: borrar el archivo. La app crea uno nuevo en el
próximo `touch`.

## `<workdir>/grupo_NN/entrega.ipynb`

El notebook del alumno, copiado del zip de Moodle por
`app/intake.py`. **El nombre original del archivo se descarta** — todos
quedan como `entrega.ipynb` para que `app/grupos.notebook_path` los
encuentre por convención (`<grupo_NN>/*.ipynb` esperando un único
match).

Si por alguna razón el grupo tiene **dos o más** `.ipynb`, la app:

- En el intake: warning, toma el primero alfabéticamente, lo copia
  como `entrega.ipynb`. Los demás archivos del grupo se ignoran.
- En la app: `notebook_path` devuelve `None` si hay más de uno, lo
  que se traduce en columna roja con label "sin ipynb". Esto solo
  pasa si alguien movió archivos al directorio del grupo a mano.

## `<workdir>/grupo_NN/feedback/<item_key>.md`

Devolución validada por el docente para ese (ítem × grupo). Vive
adentro del grupo para facilitar backups por grupo. Lo gestiona
`app/state.py`.

Tres estados encodados en el filesystem:

### 1. Archivo ausente — pendiente

El grupo no fue corregido todavía en este ítem. La matriz lo pinta
gris.

### 2. Archivo con marker `<!-- sin-observaciones -->` — sin observaciones

```
<!-- sin-observaciones -->
```

El grupo cumplió la rúbrica. Cuenta como 1pt (bien). El txt final
**no** lo incluye.

### 3. Archivo con texto — con observación

Tres sub-formatos según el nivel:

```
<!-- nivel: bien -->
Texto de la observación
```

```
<!-- nivel: regular -->
Texto de la observación
```

```
<!-- nivel: mal -->
Texto de la observación
```

El marker es la primera línea. El cuerpo va a partir de la segunda
línea. Los markers definen el color de la celda en la matriz y el
puntaje:

| marker | color | puntaje |
|---|---|---|
| `<!-- nivel: bien -->` | verde | 1pt |
| `<!-- nivel: regular -->` | amarillo | 0.5pt |
| `<!-- nivel: mal -->` | rojo | 0pt |

### Observación legacy (sin marker)

```
Texto de la observación
```

Es feedback escrito antes de que existieran los niveles. La app lo
trata como **pendiente** para el score (porque no tiene puntaje
asignado), pero igual lo incluye en el txt si tiene texto. La matriz
lo pinta gris con label "abrir".

## `<workdir>/grupo_NN/feedback/<item_key>.draft.md`

Borrador IA para ese (ítem × grupo). Lo escribe el botón "Generar
borrador IA" o el batch.

Dos estados:

### 1. Texto

```
Texto del borrador generado por Claude.
```

Se muestra como textarea readonly en la sección "Borrador IA" de la
vista corrección. El botón "Trasladar a observación" copia su
contenido al textarea de observación.

### 2. Marker `<!-- ai-ok -->`

```
<!-- ai-ok -->
```

Significa que la IA devolvió la cadena `OK` (la entrega cumple la
rúbrica según Claude). La app no carga texto en el textarea —
muestra un hint "Si coincidís, marcá sin observaciones".

**El draft nunca se exporta al `grupo_NN.txt`**. Solo el feedback
validado se exporta.

## `<workdir>/grupo_NN/cell_overrides.json`

Mapeo `expected_id → actual_id` para casos donde el alumno borró la
celda con id estable y respondió en otra. Lo gestiona
`app/state.py::set_cell_override`.

```json
{
  "ej3-code": "8a3f2e1b-9c5d-4f0e-b2a1-7c4e8f6d3a90",
  "ej5-respuesta": "ej5-respuesta-2"
}
```

Reglas:

- Solo existe si hay al menos un override. Cuando se quita el último
  con `clear_cell_override`, el archivo se borra.
- La app **resuelve** todos los lookups de celda a través de
  `state.resolved_id(expected_id, overrides)` — eso incluye la celda
  principal del ítem y las `companion_code_cells` para análisis.
- El notebook del alumno **NUNCA** se modifica. Esto es lógico-only.

## `<workdir>/grupo_NN/grupo_NN.txt`

Devolución final para Moodle. Lo escribe `app/export.py::build_grupo_txt`
disparado desde el botón "txt (N)" de la matriz.

Formato:

```
Ej 1 (código):
Texto de la observación del ej1-code

Ej 1 (análisis):
Texto de la observación del ej1-analisis

Ej 3 (código):
Texto de la observación del ej3-code

Ej 5 (análisis):
Texto de la observación del ej5-analisis-2
```

Reglas:

- **Solo se incluyen observaciones con texto real**. "Sin
  observaciones" y los ítems pendientes se omiten.
- **Sin notas numéricas**, sin puntaje, sin texto introductorio.
  Directo al copy/paste.
- **Orden**: el orden de los ítems en la rúbrica.
- **Header de cada bloque**: `Ej <N> (<tipo>):` donde N es el número
  del ejercicio (sin `ej`) y tipo es "código" o "análisis".

Si no hay ninguna observación con texto, el txt queda vacío y el
botón se deshabilita (no tiene sentido escribir un archivo de cero
bytes).

## `_TPS/metadata/lab<id>/Laboratorio_<id>_Solucion.md`

Generado a mano corriendo el prompt de
[`_TPS/metadata/prompt.md`](../_TPS/metadata/prompt.md). Para cada
celda del notebook de solución:

- **Celdas de resolución (code)**: conserva el código en bloques
  ```` ```python ```` sin outputs.
- **Celdas de setup**: omite.
- **Enunciados (markdown que arranca con `### Ejercicio`)**:
  reemplaza el contenido por su `cell_id` literal.
- **Respuestas de análisis**: conserva íntegro.
- **Resto de markdown** (intros, secciones, footers): omite.

Hoy se aplica a mano (Lab 1a tiene este par generado). La skill
`lab-extract-metadata` está en el roadmap — ver
[12-decisiones-y-roadmap.md](12-decisiones-y-roadmap.md).

## `_TPS/metadata/lab<id>/Laboratorio_<id>_eliminados.md`

Lista de cell_ids descartados al generar el `_Solucion.md`. Formato:

```markdown
- header
- imports
- secA
- ej1-enunciado
- ej2-enunciado
- footer
```

Excluye explícitamente los IDs de las preguntas de análisis (esos no
son "eliminados", quedan en el `_Solucion.md`).

## `.streamlit/config.toml`

Configuración del tema de Streamlit:

```toml
[theme]
base = "light"
backgroundColor = "#F8F5EE"           # crema cálido
secondaryBackgroundColor = "#EFEAE0"  # sidebar e inputs
primaryColor = "#7B8B6F"              # acento sage
textColor = "#2B2B2B"
font = "sans serif"
```

Detalle visual en [11-instalacion.md](11-instalacion.md).

## `.claude/settings.json` y `.claude/settings.local.json`

Permisos del agente Claude Code. Solo afectan a sesiones de Claude
Code corriendo en el repo (cuando un docente trabaja con el agente
en VS Code o terminal). **No afecta a la app de corrección** — la
app usa `setting_sources=[]` explícitamente para ignorar estos
archivos.

`settings.json` (commit-eable) tiene una lista `permissions.allow`
con comandos pre-aprobados. `settings.local.json` (gitignore) sirve
para overrides locales del docente.
