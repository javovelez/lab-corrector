# 03 — Rúbricas

La rúbrica es el contrato entre el laboratorio y la app de corrección:
para cada ítem corregible, define **qué se espera** y **qué errores son
frecuentes**. La app se la pasa a Claude para que arme borradores de
observación, y la usa para sugerir el orden y los headers en la matriz
de corrección.

Este documento describe el schema (v1 y v2), el script de
auto-generación, el formato YAML en disco y el prompt extra para
metadata por lab.

## Esquema conceptual

Una rúbrica tiene un `title` y una lista `ejercicios`. Cada ejercicio
agrupa **uno o más ítems corregibles** que el docente puntúa por
separado en la app. Hay dos `kind` de ítem:

- `code` — la celda de código que el alumno escribió. Se evalúa
  contra el enunciado.
- `analysis` — la respuesta de prosa que el alumno escribió debajo
  de una pregunta de análisis. Se evalúa contra esa pregunta.

Un ejercicio puede tener:

- 1 code + 1 analysis (lo más común).
- N codes (parte A / parte B / parte C, p. ej. `ej5-code-a`,
  `ej5-code-b`).
- N analysis (varias preguntas de análisis con sufijos como
  `ej8-pregunta-2` / `ej8-respuesta-2`).
- 0 codes + 1 analysis (ejercicio "solo-análisis", típico de una
  reflexión de cierre como en Lab 3b ej8).

## Schema v2 (actual)

YAML con `items` flexibles:

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

Campos por nivel:

**Ejercicio**:
- `id`: identificador interno (`ej1`, `ej2`, ...).
- `titulo`: título humano del ejercicio.
- `enunciado_cell`: `cell_id` de la celda con el enunciado. La app la
  usa como referencia visual a la izquierda en la vista corrección.
- `items`: lista de ítems corregibles.

**Item (común)**:
- `kind`: `code` o `analysis`.
- `key`: clave única del ítem dentro del lab. Se usa como nombre del
  archivo de feedback (`<workdir>/grupo_NN/feedback/<key>.md`) y como
  identificador en la URL de la vista corrección. Si no se provee,
  `app/main.py::build_items` la sintetiza como `<ej_id>-<kind>`.
- `rubric`: dict con `expected` (string, 1-3 oraciones) y
  `common_errors` (lista de strings; los críticos arrancan con
  `"CRÍTICO: ..."`).

**Item code**:
- `code_cell`: `cell_id` de la celda donde el alumno escribe.
- `graded_outputs`: lista informativa con los tipos de output a mirar
  (`text`, `image`). La app no la usa hoy, pero la rúbrica auto-
  generada la completa para que un humano sepa qué evaluar.

**Item analysis**:
- `pregunta_cell` (opcional): `cell_id` de la pregunta. Si falta, la
  app cae al `enunciado_cell` del ejercicio (ese es el caso de los
  ejercicios solo-análisis).
- `answer_cell`: `cell_id` de la respuesta del alumno.

## Schema v1 (legacy)

Es el formato que está en disco hoy en todas las rúbricas publicadas
(Lab 1a, 1b, 1c, 2, 3a, 3b). En v1 cada ejercicio tiene `code_cell`,
`pregunta_cell`, `answer_cell` y una sola `rubric` a nivel de
ejercicio que comparten el code y el analysis:

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

[`app/rubric.py::load_rubrica`](../app/rubric.py) detecta v1 por
ausencia de `items` y sintetiza la lista en memoria — el disco queda
intacto. La rúbrica de nivel ejercicio se "baja" idéntica a cada item
sintetizado para preservar el comportamiento. Eso significa que si en
Lab 2 una observación dice "lo crítico es X", esa misma frase aparece
tanto cuando la app pide borrador del ítem code como del analysis.

`save_rubrica` siempre escribe en v2; pero como hoy nadie llama a
`save_rubrica` desde la app sobre rúbricas existentes, el disco
permanece en v1 y no hay riesgo de migración accidental.

### Cuándo conviene migrar a v2 a mano

Cuando un ejercicio necesita rúbricas distintas para code y analysis
(p. ej. el code tiene errores frecuentes muy técnicos y el analysis
tiene errores conceptuales). En v1 obligatoriamente comparten;
moviéndolo a v2 cada item lleva su propia rubric.

También si un ejercicio tiene N codes o N analysis con sufijos: v1 no
los soporta — hay que pasar a v2.

## Auto-generación con `rubric_gen`

[`app/rubric_gen.py`](../app/rubric_gen.py) levanta la rúbrica
escaneando los notebooks y llamando a Claude **una vez por ítem**. Los
puntos de entrada son:

- `scan_ejercicios(enunciado_nb, solucion_nb)` — devuelve la
  estructura v2 sin las rubrics. Útil para tests o para regenerar
  ítems sin pisar las rubrics existentes.
- `generate_rubrica(title, enunciado_nb_path, solucion_nb_path,
  progress=...)` — escanea + llama a Claude por ítem + ensambla la
  rúbrica completa en v2.

### Cómo escanea los notebooks

El regex `^ej(\d+)-(enunciado|code|pregunta|respuesta)(?:-(\w+))?$`
captura cell_ids del framework. Para cada ejercicio (agrupado por el
número `N`):

- Toma el primer `ejN-enunciado` que aparezca como `enunciado_cell`.
- Acumula todos los `ejN-code...` en orden de aparición — cada uno
  produce un item `kind: code`.
- Parea `ejN-pregunta...` con `ejN-respuesta...` por sufijo. Si una
  tiene sufijo `-a` y la otra no, no se parean — quedan en items
  separados con `pregunta_cell` o `answer_cell` solo.
- Si un item analysis no tiene `pregunta_cell` ni `answer_cell`, se
  descarta (no debería pasar por construcción).
- Inferir `graded_outputs` de la celda del code en el notebook de
  solución: si tiene imágenes guardadas → `["image"]`, si tiene
  prints/errors → `["text"]`, si tiene ambos → `["image", "text"]`.

### Cómo llama a Claude

Un prompt diferente para code y para analysis:

- **Code**: `EJERCICIO`, `ITEM=código`, `ENUNCIADO`, `CÓDIGO DE LA
  SOLUCIÓN OFICIAL` → pide JSON con `expected` y `common_errors`.
- **Analysis**: lo mismo + `PREGUNTA DE ANÁLISIS`, `RESPUESTA OFICIAL`
  y, como contexto, los códigos del mismo ejercicio en la solución
  oficial. Esto último es importante: muchas preguntas dicen "¿por qué
  usaste X?" o "¿qué muestra la curva?", y sin el contexto del código
  Claude evalúa al voleo.

`SYSTEM_PROMPT_RUBRIC` impone:
- Devolver SOLO el JSON (sin markdown fences).
- `expected` corto (1-3 oraciones), referenciando funciones y
  variables concretas.
- `common_errors`: 3-6 ítems, prioritando los que cambian el
  resultado. Los críticos (que rompen el ítem) van con prefijo
  `"CRÍTICO: "`.
- Español rioplatense con voseo, sin emoticones.

Ante fallo del SDK o JSON mal formado, el item queda con
`{"expected": "(generación automática falló — completar a mano)",
"common_errors": ["(error: ...)"]}` para que el docente vea qué
falló y lo complete.

## CLI: `tools/rubric_build.py`

Wrapper para regenerar la rúbrica de un lab desde la línea de
comandos. Asume el layout estándar `_TPS/`.

```bash
app/.venv/bin/python tools/rubric_build.py 3b
```

Lee `_TPS/Laboratorios/Laboratorio_3b.ipynb` y
`_TPS/Soluciones/Laboratorio_3b_Solucion.ipynb`, llama a
`generate_rubrica(...)` con un callback de progress que imprime cada
ítem, y escribe la rúbrica en
`_TPS/rubricas/Laboratorio_3b.rubric.yaml`.

**Pre-requisito:** la solución tiene que estar **ejecutada** (con
outputs guardados) — si no, `graded_outputs` queda vacío.

**Tarda**: una llamada a Claude por ítem corregible. Para Lab 3b (15
ítems) son ~1-2 minutos. La barra de progreso en la app o el print
del CLI muestran el avance.

## Edición a mano post-autogen

Casi siempre conviene revisar la rúbrica auto-generada antes de
correr la app. Cosas que la IA tiende a hacer mal:

- **`expected` demasiado largo o demasiado genérico.** Reescribir en
  1-2 oraciones que mencionen los nombres reales del código.
- **`common_errors` que no son errores sino "tips".** Quitarlos: la
  rúbrica es lo que justifica una observación, no consejos
  generales.
- **Confundir code con analysis** cuando el ejercicio es solo-análisis.
  Si el ejercicio no tiene `code_cell`, hay que verificar que el item
  generado sea `kind: analysis` y que la respuesta oficial figure como
  `answer_cell`.

Lab 3b ej8 fue editado a mano para sacar `code_cell` y `pregunta_cell`,
dejando solo `enunciado_cell` + `answer_cell` (es una reflexión de
cierre sin código y sin pregunta separada).

## Metadata extra: `_Solucion.md` y `_eliminados.md`

`_TPS/metadata/prompt.md` define un prompt orientado a un asistente
de programación para producir, por cada lab, dos archivos:

- `_TPS/metadata/lab<id>/Laboratorio_<id>_Solucion.md` — el contenido
  de las celdas de solución (código + respuestas) con los enunciados
  reemplazados por el cell_id y los setups eliminados.
- `_TPS/metadata/lab<id>/Laboratorio_<id>_eliminados.md` — la lista
  de cell_ids descartados (setup, intros teóricas, footers).

Esto está pensado para futuras integraciones (analytics, búsqueda
semántica, índice cross-lab). Hoy se aplica **a mano**: el prompt se
le pasa a Claude con los dos notebooks y se guarda la salida. Lab 1a
ya tiene este par generado en `_TPS/metadata/lab1a/`. La skill
`lab-extract-metadata` está en el roadmap pero no implementada — ver
[12-decisiones-y-roadmap.md](12-decisiones-y-roadmap.md).
