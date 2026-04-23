"""
Carga de rúbricas.

Una rúbrica es un YAML con una lista `ejercicios` en la que cada entrada
referencia celdas del notebook por `cell_id` estables (los que genera
`lab_build.py` desde un `.lab.md`). La rúbrica es agnóstica al curso; el
workdir la apunta vía `config.json`.

Formato (schema v2 — items flexibles):

    title: "..."                       # opcional (si no está, el title lo trae config)
    ejercicios:
      - id: ej1
        titulo: "..."
        enunciado_cell: ej1-enunciado
        items:
          - kind: code
            key: ej1-code              # único dentro del ejercicio
            code_cell: ej1-code
            graded_outputs: [text]     # opcional, informativo
            rubric:
              expected: "..."
              common_errors: ["..."]
          - kind: analysis
            key: ej1-analisis
            pregunta_cell: ej1-pregunta   # opcional
            answer_cell:   ej1-respuesta
            rubric:
              expected: "..."
              common_errors: ["..."]

Schema v1 (legacy, sigue cargando):

    ejercicios:
      - id: ej1
        enunciado_cell: ej1-enunciado
        code_cell:      ej1-code
        pregunta_cell:  ej1-pregunta   # opcional
        answer_cell:    ej1-respuesta  # opcional
        graded_outputs: [...]          # informativo
        rubric: { expected, common_errors }

Al leer una rúbrica v1, `load_rubrica` la normaliza a v2 sintetizando `items`
para que el resto de la app vea un único shape. Guardamos siempre en v2.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _normalize_exercise(ej: dict) -> dict:
    """Si el ejercicio viene en schema v1 (sin `items`), lo convierte a v2.

    La rúbrica a nivel de ejercicio se "baja" a cada item sintetizado: en v1
    había una sola rúbrica por ejercicio que compartían código y análisis;
    acá se la damos a ambos items para preservar el comportamiento.
    """
    if "items" in ej:
        return ej

    items: list[dict] = []
    shared_rubric = ej.get("rubric", {}) or {}
    shared_graded = ej.get("graded_outputs", [])

    if ej.get("code_cell"):
        items.append({
            "kind":      "code",
            "key":       f"{ej['id']}-code",
            "code_cell": ej["code_cell"],
            "graded_outputs": shared_graded,
            "rubric":    dict(shared_rubric),
        })
    if ej.get("answer_cell"):
        item = {
            "kind":        "analysis",
            "key":         f"{ej['id']}-analisis",
            "answer_cell": ej["answer_cell"],
            "rubric":      dict(shared_rubric),
        }
        if ej.get("pregunta_cell"):
            item["pregunta_cell"] = ej["pregunta_cell"]
        items.append(item)

    return {
        "id":             ej["id"],
        "titulo":         ej.get("titulo", ""),
        "enunciado_cell": ej["enunciado_cell"],
        "items":          items,
    }


def _normalize_rubrica(rubrica: dict) -> dict:
    out = dict(rubrica)
    out["ejercicios"] = [_normalize_exercise(ej) for ej in rubrica.get("ejercicios", [])]
    return out


def load_rubrica(path: Path) -> dict:
    """Lee un YAML de rúbrica y devuelve el dict normalizado a schema v2."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _normalize_rubrica(raw)


def save_rubrica(path: Path, rubrica: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            rubrica, f,
            sort_keys=False, allow_unicode=True, width=100,
        )
