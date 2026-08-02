#!/usr/bin/env python3
"""
Contrato de notebooks corregibles — fuente única de verdad.

Este módulo define qué hace que un notebook sea compatible con la app de
corrección: el regex de `cell_id`, los roles válidos, los placeholders
del alumno y el algoritmo de agrupación en ítems corregibles.

Lo importan dos consumidores, y esa es la razón de que exista:

  - `lab_validate.py` (este mismo plugin), que le avisa al docente si un
    notebook va a fallar, en tiempo de autoría.
  - `app/rubric_gen.py` (la app de corrección), que escanea el notebook
    para armar la rúbrica.

Si los dos tuvieran su propia copia del regex, derivarían. La spec en
prosa (`skills/lab-notebook/reference/cell-ids.md`) explica y justifica;
este archivo es la autoridad.

Stdlib pura y sin estado: se puede importar desde cualquier lado.
"""
from __future__ import annotations

import re

# Versión del contrato. Subirla cuando cambie el regex, los roles, los
# placeholders o el algoritmo de agrupación. El registro de materias
# (`materias/registry.yaml` en el repo lab-corrector) compara contra esto
# para avisar qué materias quedaron desalineadas.
CONTRACT_VERSION = "1.0.0"

ROLES = ("enunciado", "code", "pregunta", "respuesta")

# Acepta `ej{N}-{rol}` con sufijo opcional `-{\w+}`.
EJ_ID_RE = re.compile(
    r"^ej(\d+)-(enunciado|code|pregunta|respuesta)(?:-(\w+))?$"
)

# Tipo de celda Jupyter que corresponde a cada rol.
EXPECTED_CELL_TYPE = {
    "enunciado": "markdown",
    "code":      "code",
    "pregunta":  "markdown",
    "respuesta": "markdown",
}

PLACEHOLDER_CODE = "# Tu código aquí"
PLACEHOLDER_ANSWER = "*(Escribí tu respuesta acá)*"


def parse_cell_id(cell_id: str):
    """Devuelve (n, rol, sufijo) si el id es corregible, o None si no lo es.

    Un id que devuelve None no es un error: es una celda no evaluable
    (header, reglas, imports, secA, checklist, footer, ...). La app las
    ignora en silencio.
    """
    m = EJ_ID_RE.match(cell_id or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def analysis_key(ej_id: str, sufijo: str | None) -> str:
    """Clave del ítem de análisis. Debe coincidir con la que arma la app."""
    return f"{ej_id}-analisis" if sufijo is None else f"{ej_id}-analisis-{sufijo}"


def group_cells(cells: list[dict]) -> dict[int, dict]:
    """Agrupa las celdas corregibles por número de ejercicio.

    `cells` es una lista de dicts con al menos `id` y `cell_type`, o sea
    el formato nativo de un .ipynb. Devuelve, por número de ejercicio, un
    slot con el enunciado, los codes en orden de aparición y las
    preguntas/respuestas indexadas por sufijo.

    Es la misma agrupación que hace `rubric_gen.scan_ejercicios`; se
    mantiene acá para que el validador y la app no puedan diferir.
    """
    por_ej: dict[int, dict] = {}
    for cell in cells:
        parsed = parse_cell_id(cell.get("id", ""))
        if parsed is None:
            continue
        n, role, sufijo = parsed
        slot = por_ej.setdefault(n, {
            "id":             f"ej{n}",
            "enunciado_cell": None,
            "codes":          [],   # [(sufijo, cell_id)]
            "preguntas":      {},   # sufijo → cell_id
            "respuestas":     {},   # sufijo → cell_id
        })
        cid = cell.get("id", "")
        if role == "enunciado":
            if slot["enunciado_cell"] is None:
                slot["enunciado_cell"] = cid
        elif role == "code":
            slot["codes"].append((sufijo, cid))
        elif role == "pregunta":
            slot["preguntas"][sufijo] = cid
        elif role == "respuesta":
            slot["respuestas"][sufijo] = cid
    return por_ej
