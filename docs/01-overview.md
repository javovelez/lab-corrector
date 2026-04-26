# 01 — Overview del framework

## Por qué existe

La materia **Redes Neuronales Profundas** (UTN FRM, Ing. Javier Vélez)
publica entre seis y diez laboratorios por cuatrimestre, cada uno con un
notebook de enunciado, un notebook de solución y una tanda de
correcciones individualizadas para grupos de alumnos. Mantener todo eso
sincronizado a mano, agregar feedback grupo por grupo y producir una
devolución consistente para Moodle es donde se va el tiempo del docente.

El framework intenta resolver tres problemas que aparecen siempre:

1. **El enunciado y la solución se desincronizan.** Si el lab vive como
   dos notebooks separados, cualquier corrección sobre el enunciado hay
   que replicarla a mano. La fuente única `.lab.md` resuelve esto: un
   solo archivo que se compila a los dos `.ipynb`.
2. **Las rúbricas son tediosas de armar.** Sentarse a escribir
   "qué se espera" y "errores frecuentes" para 12 ejercicios es trabajo
   mecánico. La autogeneración con Claude produce un punto de partida
   editable que cubre el 80% del trabajo.
3. **Corregir grupo por grupo es lento.** La app de corrección rompe el
   problema en celdas (un ítem corregible × un grupo) y permite navegar
   por matriz, generar borradores con IA, y producir el txt final por
   grupo.

## Los tres pilares

```
┌──────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  AUTORÍA         │   │  RÚBRICA           │   │  CORRECCIÓN        │
│                  │   │                    │   │                    │
│  .lab.md  ───────┼──►│  rubric_gen        │   │  app Streamlit     │
│  (fuente única)  │   │  ─►  rubric.yaml   │──►│  ─►  grupo_NN.txt  │
│                  │   │                    │   │                    │
│  lab_build.py    │   │  rubric_build.py   │   │  Claude (drafts)   │
│  ─►  .ipynb ×2   │──►│  (CLI)             │   │                    │
└──────────────────┘   └────────────────────┘   └────────────────────┘
       │                       │                          │
       ▼                       ▼                          ▼
   Laboratorios/         _TPS/rubricas/              <workdir>/
   Soluciones/           *.rubric.yaml               grupo_NN/
   *.ipynb                                           grupo_NN.txt
```

Los tres pilares están **acoplados solo por convención de `cell_id`**.
Cualquier notebook que respete el regex `^ej(\d+)-(enunciado|code|pregunta|respuesta)(?:-(\w+))?$`
funciona con la rúbrica autogenerada y con la app, sin necesidad de
haber pasado por `lab_build.py`. Eso es deliberado: para Lab 2 (que se
escribió antes del framework) hubo que aplicar un parche de adaptación
([`tools/lab2_split_pregunta.py`](../tools/lab2_split_pregunta.py)) y
después la app lo corrige sin ningún tratamiento especial.

Detalle del contrato de cell_id: [08-convenciones-cellids.md](08-convenciones-cellids.md).

## Flujo end-to-end de un lab nuevo

1. **Escribir la fuente** en `_TPS/sources/Laboratorio_X.lab.md` —
   formato Pandoc-style con `::::cell{...}` y bloques fenced para
   marcar enunciado vs. solución. Detalle en
   [02-autoria-lab-md.md](02-autoria-lab-md.md).
2. **Compilar a notebooks**:
   ```
   python tools/lab_build.py _TPS/sources/Laboratorio_X.lab.md
   ```
   Produce `_TPS/Laboratorios/Laboratorio_X.ipynb` (enunciado del
   alumno) y `_TPS/Soluciones/Laboratorio_X_Solucion.ipynb` (con
   código y respuestas de análisis completas). El título de la
   solución lleva el sufijo `-- SOLUCION`.
3. **Ejecutar la solución** en Colab/Jupyter para que tenga outputs
   guardados — la rúbrica usa esos outputs para inferir
   `graded_outputs`.
4. **Generar la rúbrica**:
   ```
   app/.venv/bin/python tools/rubric_build.py X
   ```
   Llama a Claude (vía `claude-agent-sdk`, sin API key — usa la sesión
   local de Claude Code) una vez por ítem corregible. Devuelve
   `_TPS/rubricas/Laboratorio_X.rubric.yaml`. Detalle en
   [03-rubricas.md](03-rubricas.md).
5. **Editar la rúbrica a mano** donde haga falta — Claude saca un
   borrador, el docente afina los `expected` y los `common_errors`.
6. **Publicar el enunciado** (Colab vía `_TPS/Laboratorios/links.md`).

## Flujo end-to-end de una corrección

1. **Bajar el zip de Moodle** con todas las entregas de los grupos.
2. **Abrir la app**:
   ```
   app/.venv/bin/streamlit run app/main.py
   ```
3. **Crear un workdir** (carpeta vacía donde va a vivir la tanda) o
   abrir uno existente desde el landing. La app pide los paths del
   notebook de enunciado, del de solución, de la rúbrica y del zip.
4. **Importar el zip** — la app extrae cada `Grupo N_<id>_assignsubmission_file/`
   a `<workdir>/grupo_NN/entrega.ipynb`.
5. **Corregir** con la matriz: un ítem corregible por fila, un grupo
   por columna. Cada celda lleva al detalle de ese (ítem × grupo) con
   referencia a la izquierda (enunciado o pregunta + solución oficial)
   y la entrega del grupo a la derecha. Se puede generar borrador con
   IA (individual o por fila completa). Detalle de la UI en
   [06-app-ui-detallada.md](06-app-ui-detallada.md).
6. **Asignar puntaje** (bien/regular/mal) — el porcentaje del grupo
   se pinta en el header de la columna.
7. **Escribir el `grupo_NN.txt`** desde el botón de la matriz. Solo
   lleva las observaciones que el docente validó (las "sin
   observaciones" se omiten).
8. **Subir los txt a Moodle**.

Detalle del flujo en [05-app-flujo-correccion.md](05-app-flujo-correccion.md).

## Estado al 2026-04-26

- Publicados: Labs 1a, 1b, 1c, 2, 3a, 3b — enunciados y soluciones.
- Lab 3a y 3b en formato `.lab.md` + rúbrica YAML autogenerada.
- Lab 2 2026 en corrección activa (workdir en `2026/lab2/`).
- App de corrección en producción (versión sin issues abiertos, en
  fase de testeo intensivo por parte del docente).

## Decisiones explícitas

El framework deja afuera varias cosas a propósito — las listamos con
su motivo en [12-decisiones-y-roadmap.md](12-decisiones-y-roadmap.md).
Resumen:

- Sin multi-tenancy ni multi-user.
- Sin notas numéricas en el txt final (solo texto cualitativo cuando
  hay observación).
- Sin edición directa de los notebooks de los alumnos desde la app.
- Sin que la IA vea imágenes de outputs (solo código + texto).
