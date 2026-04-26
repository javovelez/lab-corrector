# CLAUDE.md — Materia Redes Neuronales Profundas (RNP), UTN FRM

Este archivo es el norte conceptual del proyecto. La documentación
operativa completa (autoría, rúbricas, app, formatos, troubleshooting,
instalación, decisiones) vive en [docs/](docs/README.md). La guía de
estilo de los notebooks vive en [_TPS/README.md](_TPS/README.md).

---

## Qué es este proyecto

Material didáctico de la materia **Redes Neuronales Profundas** (UTN FRM, Ing. Javier Vélez). Contiene:

- **Clases teórico-prácticas** en subcarpetas `01_*`, `02_*`, … (`01_fundamentos_deep_learning`, `02_redes_convolucionales`, `03_transferencia_de_conocimiento`, etc.). Cada una con notebooks, slides y material de referencia.
- **Trabajos prácticos** en [_TPS/](_TPS/) — enunciados en `_TPS/Laboratorios/`, soluciones en `_TPS/Soluciones/`, fuentes en `_TPS/sources/`, rúbricas en `_TPS/rubricas/`, metadatos en `_TPS/metadata/`.
- **App de corrección** en [app/](app/) — Streamlit + Claude SDK para corregir entregas de Moodle con borradores IA.
- **Notebooks en markdown** (`notebooks_md/`) — exportaciones legibles de clases teóricas, de lectura.

---

## Los tres pilares del framework

El sistema se compone de tres pilares acoplados solo por convención de
`cell_id`:

1. **Autoría** — fuente única `.lab.md` que se compila a dos `.ipynb` (enunciado + solución) con [`tools/lab_build.py`](tools/lab_build.py). Detalle: [docs/02-autoria-lab-md.md](docs/02-autoria-lab-md.md).
2. **Rúbrica** — YAML por lab que define qué se espera y qué errores son frecuentes para cada ítem corregible. Auto-generación con Claude vía [`tools/rubric_build.py`](tools/rubric_build.py). Detalle: [docs/03-rubricas.md](docs/03-rubricas.md).
3. **Corrección** — app Streamlit en [`app/`](app/) que opera sobre un *workdir* y produce un `grupo_NN.txt` por grupo. Detalle: [docs/04-app-overview.md](docs/04-app-overview.md) y siguientes.

Diagrama de flujo end-to-end: [docs/01-overview.md](docs/01-overview.md).

---

## Convenciones (resumen — el detalle está en [_TPS/README.md](_TPS/README.md))

- **Idioma:** español rioplatense, voseo ("creá", "usá", "respondé").
- **Sin emoticones.**
- **Tres celdas por ejercicio** es el patrón usual: enunciado, código, pregunta de análisis + respuesta. El framework también soporta **ejercicios solo-análisis** (sin celda de código — típico de una reflexión de cierre) y ejercicios con múltiples bloques de código o múltiples preguntas (ver sufijos en [docs/08-convenciones-cellids.md](docs/08-convenciones-cellids.md)).
- Las celdas donde el alumno escribe llevan placeholder exacto: `# Tu código aquí` o `*(Escribí tu respuesta acá)*`.
- Convención de `cell_id`: regex `^ej(\d+)-(enunciado|code|pregunta|respuesta)(?:-(\w+))?$`. Detalle: [docs/08-convenciones-cellids.md](docs/08-convenciones-cellids.md).

---

## Estado actual (abril 2026)

- **Publicados**: Lab 1a, 1b, 1c, 2, 3a, 3b (enunciados + soluciones). Lab 3a y 3b en formato `.lab.md` + rúbrica YAML autogenerada.
- **`tools/lab_build.py`**: compila `.lab.md` → dos `.ipynb`.
- **`tools/rubric_build.py`**: CLI para autogenerar rúbricas vía Claude SDK (requiere venv `app/.venv/`).
- **App de corrección**: en producción, fase de testeo intensivo. Lab 2 2026 en corrección activa (workdir en `2026/lab2/`); Lab 3a y 3b listos para cuando lleguen entregas. Schema de rúbrica v2 (items flexibles) con compat hacia atrás v1.
- **`_TPS/metadata/prompt.md`** define cómo extraer, por notebook, `_Solucion.md` (solo celdas de solución) y `_eliminados.md` (IDs descartados). Hoy se aplica a mano; Lab 1a ya tiene este par generado.
- Mejoras pendientes por lab: ver [_TPS/Laboratorios/mejoras.md](_TPS/Laboratorios/mejoras.md).
- Roadmap del framework: ver [docs/12-decisiones-y-roadmap.md](docs/12-decisiones-y-roadmap.md).

---

## Scripts y skills

Inventario completo. Detalle de cada uno en la página de docs correspondiente.

| Nombre | Tipo | Estado | Qué hace | Doc |
|---|---|---|---|---|
| `lab-build` | script CLI ([`tools/lab_build.py`](tools/lab_build.py)) | **Implementado** | Compila `.lab.md` → `.ipynb` (enunciado + solución). Uso: `python tools/lab_build.py <archivo.lab.md>` | [docs/02](docs/02-autoria-lab-md.md) |
| `rubric-build` | script CLI ([`tools/rubric_build.py`](tools/rubric_build.py)) | **Implementado** | Genera `_TPS/rubricas/Laboratorio_<id>.rubric.yaml` llamando a `app/rubric_gen.py`. Uso: `app/.venv/bin/python tools/rubric_build.py <lab_id>` | [docs/03](docs/03-rubricas.md) |
| `lab2-split-pregunta` | script CLI ([`tools/lab2_split_pregunta.py`](tools/lab2_split_pregunta.py)) | **Implementado (one-off)** | Parche para notebooks pre-framework: divide cada `ejN-pregunta` en pregunta + respuesta. Idempotente, crea `.bak`. | [docs/10](docs/10-troubleshooting.md) |
| App de corrección | Streamlit app ([`app/`](app/)) | **Implementado** | UI de corrección por grupo/ítem con borradores IA, export a `grupo_NN.txt`, niveles de puntaje, navegador de celdas. Uso: `app/.venv/bin/streamlit run app/main.py` | [docs/04](docs/04-app-overview.md) – [docs/07](docs/07-ia.md) |
| `metadata/prompt.md` | prompt para Claude ([`_TPS/metadata/prompt.md`](_TPS/metadata/prompt.md)) | **Aplicado a mano** | Genera `_Solucion.md` + `_eliminados.md` por lab. Hoy se ejecuta desde un agente Claude. | [docs/03](docs/03-rubricas.md), [docs/09](docs/09-formatos-archivo.md) |
| `lab-new` | skill | **Planificado** | Arrancaría un `.lab.md` desde un notebook de referencia o desde cero | [docs/12](docs/12-decisiones-y-roadmap.md) |
| `lab-extract-metadata` | script | **Planificado** | Re-ejecutaría la lógica de `_TPS/metadata/prompt.md` sobre un par enunciado/solución | [docs/12](docs/12-decisiones-y-roadmap.md) |

### Módulos de la app

| Archivo | Responsabilidad |
|---|---|
| [`app/main.py`](app/main.py) | UI Streamlit: landing, sidebar, matriz, vista corrección, navegador de celdas, panel IA |
| [`app/ai.py`](app/ai.py) | Borradores IA: prompt individual + batch por fila, system prompt, parser de JSON |
| [`app/rubric_gen.py`](app/rubric_gen.py) | Auto-generación de rúbrica (escaneo + llamadas a Claude por ítem) |
| [`app/rubric.py`](app/rubric.py) | Carga/normalización de rúbrica (v1 → v2 in-memory) |
| [`app/state.py`](app/state.py) | Persistencia de feedback (markers de nivel, drafts, cell overrides) |
| [`app/export.py`](app/export.py) | `grupo_NN.txt` + scoring por grupo |
| [`app/intake.py`](app/intake.py) | Intake del zip de Moodle |
| [`app/workdir.py`](app/workdir.py) | `WorkdirConfig` + I/O de `.corrector/config.json` |
| [`app/recents.py`](app/recents.py) | Registry global `~/.lab_corrector/recent.json` |
| [`app/grupos.py`](app/grupos.py) | Descubrimiento de grupos en el workdir |
| [`app/nbparse.py`](app/nbparse.py) | Parseo de notebooks Jupyter sin `nbformat` |

---

## Cómo colaborar con Claude en este repo

- **Antes de crear o modificar un laboratorio**, leé [_TPS/README.md](_TPS/README.md) (guía de estilo) y [docs/02-autoria-lab-md.md](docs/02-autoria-lab-md.md) (formato de la fuente).
- **No edites `.ipynb` a mano** una vez que exista su `.lab.md`. Editá la fuente y recompilá con `python tools/lab_build.py _TPS/sources/Laboratorio_X.lab.md`.
- **Toda celda que el alumno completa necesita `cell_id` que matchee la convención** del framework, para que la app de corrección la parsee. Detalle en [docs/08-convenciones-cellids.md](docs/08-convenciones-cellids.md).
- **Antes de tocar el código de la app**, leé [docs/04-app-overview.md](docs/04-app-overview.md) y [docs/09-formatos-archivo.md](docs/09-formatos-archivo.md) para entender el modelo de datos.
- Si una decisión no está clara, preguntá antes de codificar — mejor chico y correcto que grande y a reescribir.
