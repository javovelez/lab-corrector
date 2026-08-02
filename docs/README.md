# Wiki del framework RNP

Documentación del sistema de autoría, generación de rúbricas y corrección
asistida de los laboratorios de **Redes Neuronales Profundas** (UTN FRM).
Este wiki es la referencia operativa del framework — el `CLAUDE.md` de la
raíz es el norte conceptual y la guía de estilo de los notebooks vive en
[../_TPS/README.md](../_TPS/README.md).

## Índice

1. [Overview del framework](01-overview.md) — qué es, los tres pilares, diagrama de flujo de punta a punta.
2. [Autoría con `.lab.md`](../plugins/lab-notebook/skills/lab-notebook/reference/lab-md.md) — formato de la fuente única, `lab_build.py` (plugin), ciclo de edición.
3. [Rúbricas](03-rubricas.md) — schema v1 y v2, autogeneración con Claude, `app/rubric_build.py`, formato YAML en disco.
4. [Arquitectura de la app](04-app-overview.md) — workdir, módulos de `app/`, registry de recientes, ciclo de vida.
5. [Flujo de corrección end-to-end](05-app-flujo-correccion.md) — landing, intake del zip de Moodle, matriz, vista corrección, exportación.
6. [UI detallada](06-app-ui-detallada.md) — vista matriz, navegador de celdas, `cell_overrides`, niveles de puntaje, sidebar.
7. [Uso de IA](07-ia.md) — borradores individuales, batch por ítem, contexto de código en análisis, system prompts.
8. [Convenciones de `cell_id`](../plugins/lab-notebook/skills/lab-notebook/reference/cell-ids.md) — regex canónico, sufijos, ejercicios solo-análisis y multi-código.
9. [Formatos de archivo](09-formatos-archivo.md) — `.lab.md`, rúbrica YAML, `config.json`, archivos de feedback, `cell_overrides.json`, `recent.json`.
10. [Troubleshooting](10-troubleshooting.md) — Lab 2 con celdas pegadas, alumnos que rompen `cell_id`, entregas faltantes, fallas de IA.
11. [Instalación](11-instalacion.md) — venv, dependencias, requisitos, levantar la app, configuración de Streamlit.
12. [Decisiones y roadmap](12-decisiones-y-roadmap.md) — qué se descartó deliberadamente y qué queda pendiente.

## Inventario rápido de scripts y skills

Cubierto en detalle en las páginas correspondientes. Atajo:

| Componente | Path | Página |
|---|---|---|
| Compilador de fuente única `.lab.md` → `.ipynb` | [`lab_build.py`](../plugins/lab-notebook/scripts/lab_build.py) | [02](../plugins/lab-notebook/skills/lab-notebook/reference/lab-md.md) |
| Generador CLI de rúbrica YAML | [`app/rubric_build.py`](../app/rubric_build.py) | [03](03-rubricas.md) |
| Parche one-off para Lab 2 (split pregunta+respuesta) | [`lab2_split_pregunta.py`](../plugins/lab-notebook/scripts/lab2_split_pregunta.py) | [10](10-troubleshooting.md) |
| App Streamlit (corrección) | [`app/main.py`](../app/main.py) | [04](04-app-overview.md) – [07](07-ia.md) |
| Generación de rúbrica embebida en la app | [`app/rubric_gen.py`](../app/rubric_gen.py) | [03](03-rubricas.md) |
| Borradores IA + batch | [`app/ai.py`](../app/ai.py) | [07](07-ia.md) |
| Persistencia de feedback / overrides | [`app/state.py`](../app/state.py) | [09](09-formatos-archivo.md) |
| Exportación a `grupo_NN.txt` + scoring | [`app/export.py`](../app/export.py) | [05](05-app-flujo-correccion.md) |
| Carga/normalización de rúbrica | [`app/rubric.py`](../app/rubric.py) | [03](03-rubricas.md) |
| Intake de zip de Moodle | [`app/intake.py`](../app/intake.py) | [05](05-app-flujo-correccion.md) |
| Configuración del workdir | [`app/workdir.py`](../app/workdir.py) | [04](04-app-overview.md) |
| Registry de workdirs recientes | [`app/recents.py`](../app/recents.py) | [04](04-app-overview.md) |
| Descubrimiento de grupos | [`app/grupos.py`](../app/grupos.py) | [04](04-app-overview.md) |
| Parseo de notebooks Jupyter | [`app/nbparse.py`](../app/nbparse.py) | [04](04-app-overview.md) |
| Prompt para extraer metadata por lab | [`_TPS/metadata/prompt.md`](../_TPS/metadata/prompt.md) | [03](03-rubricas.md) |
| Tema visual de la app | [`.streamlit/config.toml`](../.streamlit/config.toml) | [11](11-instalacion.md) |
| Permisos del agente Claude Code | [`.claude/settings.json`](../.claude/settings.json) | [11](11-instalacion.md) |

## Orden de lectura sugerido

- **Si vas a autorear un lab nuevo:** [01](01-overview.md) → [02](../plugins/lab-notebook/skills/lab-notebook/reference/lab-md.md) → [08](../plugins/lab-notebook/skills/lab-notebook/reference/cell-ids.md) → [03](03-rubricas.md).
- **Si vas a corregir una tanda de entregas:** [01](01-overview.md) → [11](11-instalacion.md) → [05](05-app-flujo-correccion.md) → [06](06-app-ui-detallada.md) → [07](07-ia.md).
- **Si algo se rompe:** [10](10-troubleshooting.md) primero, después la página del componente afectado.
- **Si vas a tocar el código de la app:** [04](04-app-overview.md) → [09](09-formatos-archivo.md) → [12](12-decisiones-y-roadmap.md).
