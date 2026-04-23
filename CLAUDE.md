# CLAUDE.md — Materia Redes Neuronales Profundas (RNP), UTN FRM

Este archivo es el norte del proyecto. Complementa (no reemplaza) a [_TPS/README.md](_TPS/README.md), que es la guía de estilo autoritativa para los laboratorios.

---

## Qué es este proyecto

Material didáctico de la materia **Redes Neuronales Profundas** (UTN FRM, Ing. Javier Vélez). Contiene:

- **Clases teórico-prácticas** en subcarpetas `01_*`, `02_*`, … (`01_fundamentos_deep_learning`, `02_redes_convolucionales`, `03_transferencia_de_conocimiento`, etc.). Cada una con notebooks, slides y material de referencia.
- **Trabajos prácticos** en [_TPS/](_TPS/) — enunciados en `_TPS/Laboratorios/`, soluciones en `_TPS/Soluciones/`, metadatos de corrección en `_TPS/metadata/`.
- **Notebooks en markdown** (`notebooks_md/`) — exportaciones legibles de clases teóricas, de lectura.

---

## Hacia dónde vamos: aplicación de corrección asistida

El objetivo a mediano plazo es una **app de corrección** con este flujo:

1. Cargar un grupo de notebooks entregados por los alumnos (uno por grupo).
2. Iterar ejercicio por ejercicio: la app muestra
   - el **enunciado**,
   - la **respuesta del grupo** (celda de código y/o celda de análisis),
   - la **solución oficial** del ejercicio,
   - una **devolución generada por IA** (p. ej. "Ejercicio 1: [explicación del error]") que el docente puede editar y validar.
3. Al terminar todos los grupos para un ejercicio, avanza al siguiente.
4. Salida: un `grupo_xx.txt` por grupo con las observaciones del docente.

Ese objetivo impone dos disciplinas sobre los laboratorios que creemos **desde ahora**:

- **Identificar celdas con precisión.** Toda celda que el alumno completa (código o respuesta a pregunta de análisis) debe tener `cell_id` estable y tag de rol. Lo mismo para las celdas cuya **salida** (imagen, texto no predecible) es parte de lo evaluable.
- **Tener una rúbrica por laboratorio.** Por cada celda del alumno, qué se espera, errores frecuentes, y qué gráficos/outputs se miran.

---

## Arquitectura de autoría propuesta (fuente en Markdown)

Para evitar editar JSON de notebooks y habilitar corrección automatizada, usamos:

```
_TPS/
├── sources/                         ← fuente de verdad (a crear)
│   ├── Laboratorio_1a.lab.md
│   ├── Laboratorio_2.lab.md
│   └── Laboratorio_3.lab.md
├── Laboratorios/                    ← .ipynb generados (enunciados)
├── Soluciones/                      ← .ipynb generados (soluciones)
├── rubricas/                        ← rúbricas generadas por lab (a crear)
└── metadata/                        ← _eliminados.md + _Solucion.md por lab
```

### Formato `.lab.md` (borrador, a fijar cuando hagamos el primero)

Un solo archivo que contiene enunciado + solución. Cada celda lleva frontmatter YAML con:

```yaml
---
id: ej4-code          # cell_id estable
type: code            # code | markdown
role: student-code    # student-code | student-answer | scaffolding | setup | section | enunciado | pregunta | respuesta | footer
student_writes: true  # true → en enunciado queda "# Tu código aquí"; en solución va el código
graded_output:        # qué de la salida se evalúa (opcional)
  - image             # imagen generada
  - numeric           # print de error < 0.001
rubric:               # solo en la fuente; no se inyecta en el .ipynb del alumno
  expected: "..."
  common_errors: ["...", "..."]
---
# contenido de la celda (markdown o código)
```

### Script `tools/lab_build.py` (implementado)

Uso: `python tools/lab_build.py _TPS/sources/Laboratorio_X.lab.md`

- Lee `.lab.md` → produce:
  - `Laboratorios/Laboratorio_X.ipynb` (enunciado, placeholders del alumno).
  - `Soluciones/Laboratorio_X_Solucion.ipynb` (con código completo y respuestas; al título le agrega `-- SOLUCION`).
- Cell IDs estables + cell tags por role.
- **Pendiente** en versiones futuras: generación de `rubricas/Laboratorio_X_rubrica.md` y de `metadata/labX/Laboratorio_X_Solucion.md` + `_eliminados.md` (la lógica vive en `_TPS/metadata/prompt.md` y por ahora se aplica a mano).

---

## Convenciones (resumen — el detalle está en [_TPS/README.md](_TPS/README.md))

- **Idioma:** español rioplatense, voseo ("creá", "usá", "respondé").
- **Sin emoticones.**
- **Tres celdas por ejercicio** es el patrón usual: enunciado, código, pregunta de análisis + respuesta. La app de corrección soporta también **ejercicios solo-análisis** (sin celda de código — típico de una reflexión de cierre) y ejercicios con múltiples bloques de código.
- Las celdas donde el alumno escribe llevan placeholder exacto: `# Tu código aquí` o `*(Escribí tu respuesta acá)*`.

---

## Estado actual (abril 2026)

- Publicados: Lab 1a, 1b, 1c, 2, 3a, 3b (enunciados + soluciones). Lab 3a y 3b en formato `.lab.md` + rúbrica YAML.
- `tools/lab_build.py` compila `.lab.md` → `.ipynb` (enunciado + solución).
- `tools/rubric_build.py` genera la rúbrica YAML de un lab llamando al generador vía Claude SDK (requiere venv `app/.venv/`).
- **App de corrección** en `app/` corriendo en producción: actualmente corrigiendo Lab 2 2026 (workdir en `2026/lab2/`). Framework agnóstico basado en workdir (ver memory `project_correction_app.md` para detalles).
- `_TPS/metadata/prompt.md` define cómo extraer, por notebook, `_Solucion.md` (solo celdas de solución) y `_eliminados.md` (IDs descartados). Lab 1a ya tiene este par generado.
- Mejoras pendientes por lab: ver [_TPS/Laboratorios/mejoras.md](_TPS/Laboratorios/mejoras.md).

---

## Skills / tools

| Nombre | Tipo | Estado | Qué hace |
|---|---|---|---|
| `lab-build` | script CLI (`tools/lab_build.py`) | **Implementado** | Compila `.lab.md` → `.ipynb` (enunciado + solución). Uso: `python3 tools/lab_build.py <archivo.lab.md>` |
| `rubric-build` | script CLI (`tools/rubric_build.py`) | **Implementado** | Genera `_TPS/rubricas/Laboratorio_<id>.rubric.yaml` llamando a `app/rubric_gen.py` (Claude SDK). Requiere solución ejecutada con outputs. Uso: `app/.venv/bin/python tools/rubric_build.py <lab_id>` |
| `corrector app` | Streamlit app (`app/`) | **Implementado** | UI de corrección por grupo/ítem con borradores IA, export a `grupo_NN.txt`. Uso: `app/.venv/bin/streamlit run app/main.py` |
| `lab-new` | skill | Planificado | Arranca un `.lab.md` desde un notebook de referencia o desde cero |
| `lab-extract-metadata` | script | Planificado | Re-ejecuta la lógica de `_TPS/metadata/prompt.md` sobre un par enunciado/solución |

---

## Cómo colaborar con Claude en este repo

- **Antes de crear o modificar un laboratorio,** leé [_TPS/README.md](_TPS/README.md) — es la guía de estilo vinculante.
- **No edites `.ipynb` a mano** una vez que exista su `.lab.md`. Editá la fuente y recompilá con `python tools/lab_build.py _TPS/sources/Laboratorio_X.lab.md`.
- **Toda celda que el alumno completa necesita `cell_id` y role**, para que la app de corrección pueda parsear el entregable.
- Si una decisión no está clara, preguntá antes de codificar — mejor chico y correcto que grande y a reescribir.
