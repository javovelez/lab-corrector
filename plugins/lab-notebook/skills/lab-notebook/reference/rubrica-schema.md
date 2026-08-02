# Schema de la rúbrica

La rúbrica es un YAML por laboratorio que define, para cada ítem
corregible, **qué se espera** y **qué errores son frecuentes**. La
consume la app de corrección para armar la matriz de corrección y para
pedirle borradores de devolución a Claude.

**No se escribe a mano.** La genera la app desde el par
enunciado/solución, con una llamada a Claude por ítem. Este documento
sirve para leerla, retocarla a mano, o entender qué produce tu notebook.

## Esquema conceptual

Una rúbrica tiene un `title` y una lista `ejercicios`. Cada ejercicio
agrupa uno o más **ítems corregibles**, que el docente puntúa por
separado. Hay dos `kind`:

- `code` — la celda de código que escribió el alumno.
- `analysis` — la respuesta en prosa debajo de una pregunta de análisis.

Un ejercicio puede tener 1 code + 1 analysis (lo más común), N codes
(parte A/B/C), N analysis (varias preguntas), o 0 codes + 1 analysis
(ejercicio solo-análisis, típico de una reflexión de cierre).

## Schema v2 (actual)

```yaml
title: "Laboratorio 3b — Transferencia de estilo rápida"
ejercicios:
  - id: ej1
    titulo: "Matriz de Gram"
    enunciado_cell: ej1-enunciado
    items:
      - kind: code
        key: ej1-code
        code_cell: ej1-code
        graded_outputs: [text]      # opcional, informativo
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

**Ejercicio**

- `id` — identificador interno (`ej1`, `ej2`, ...).
- `titulo` — título humano.
- `enunciado_cell` — `cell_id` del enunciado. La app lo muestra como
  referencia a la izquierda en la vista corrección.
- `items` — lista de ítems corregibles.

**Item (común)**

- `kind` — `code` o `analysis`.
- `key` — clave única del ítem dentro del lab. Es el nombre del archivo
  de feedback (`<workdir>/grupo_NN/feedback/<key>.md`) y el
  identificador en la URL. Si falta, la app la sintetiza como
  `<ej_id>-<kind>`.
- `rubric` — `expected` (1 a 3 oraciones) y `common_errors` (lista; los
  críticos arrancan con `"CRÍTICO: "`).

**Item `code`**

- `code_cell` — `cell_id` de la celda donde escribe el alumno.
- `graded_outputs` — lista informativa de tipos de output a mirar
  (`text`, `image`). Sale de los outputs guardados en la solución
  ejecutada.

**Item `analysis`**

- `pregunta_cell` (opcional) — si falta, la app cae al `enunciado_cell`
  del ejercicio. Ese es el caso del ejercicio solo-análisis.
- `answer_cell` — `cell_id` de la respuesta del alumno.

## Schema v1 (legacy)

Formato de las rúbricas publicadas antes del schema v2. Cada ejercicio
tiene `code_cell`, `pregunta_cell`, `answer_cell` y una sola `rubric` a
nivel de ejercicio, compartida entre el code y el analysis:

```yaml
title: "Laboratorio 2"
ejercicios:
  - id: ej1
    titulo: "Correlación cruzada manual"
    enunciado_cell: ej1-enunciado
    code_cell: ej1-code
    pregunta_cell: ej1-pregunta
    answer_cell: ej1-respuesta
    graded_outputs: [text]
    rubric:
      expected: "..."
      common_errors: ["..."]
```

`app/rubric.py::load_rubrica` detecta v1 por ausencia de `items` y
sintetiza la lista en memoria; el archivo en disco queda intacto. La
rúbrica de nivel ejercicio se baja idéntica a cada ítem sintetizado.

**Conviene migrar a v2 a mano** cuando el code y el analysis necesitan
rúbricas distintas, o cuando el ejercicio tiene N codes / N analysis con
sufijos — v1 no los soporta.
