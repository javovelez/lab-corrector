"""
Parseo de notebooks Jupyter sin dependencia de `nbformat`.

La app interactúa con tres notebooks por lab:
  - enunciado (`_TPS/Laboratorios/Laboratorio_X.ipynb`)
  - solución  (`_TPS/Soluciones/Laboratorio_X_Solucion.ipynb`)
  - entrega   (`_correccion/labX/entregas/grupo_YY/entrega.ipynb`)

Todos comparten el mismo esquema de `cell_id` porque los dos primeros los
genera `tools/lab_build.py` y Colab preserva los IDs al re-guardar.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=32)
def load_notebook(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_cell(nb: dict, cell_id: str) -> dict | None:
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return cell
    return None


def cell_source(cell: dict | None) -> str:
    if cell is None:
        return ""
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def cell_outputs(cell: dict | None) -> list[dict]:
    """
    Normaliza `cell.outputs` a una lista de dicts con forma:
      - {"kind": "text",  "text": str}
      - {"kind": "image", "mime": "image/png"|"image/jpeg", "data_b64": str}
      - {"kind": "error", "text": str}
    Se preserva el orden del notebook.
    """
    if cell is None:
        return []
    result: list[dict] = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")

        if otype == "stream":
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            result.append({"kind": "text", "text": text})

        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            # Priorizamos imagen sobre texto cuando ambos están presentes.
            if "image/png" in data:
                img = data["image/png"]
                if isinstance(img, list):
                    img = "".join(img)
                result.append({"kind": "image", "mime": "image/png", "data_b64": img})
            elif "image/jpeg" in data:
                img = data["image/jpeg"]
                if isinstance(img, list):
                    img = "".join(img)
                result.append({"kind": "image", "mime": "image/jpeg", "data_b64": img})
            elif "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                result.append({"kind": "text", "text": text})

        elif otype == "error":
            tb = out.get("traceback", [])
            if isinstance(tb, list):
                tb = "\n".join(tb)
            result.append({"kind": "error", "text": tb})

    return result


def image_popout_html(data_b64: str, mime: str) -> str:
    """`<img>` envuelto en `<a target="_blank">`: click abre la imagen en pestaña nueva."""
    data_url = f"data:{mime};base64,{data_b64}"
    return (
        f'<a href="{data_url}" target="_blank" rel="noopener" '
        f'style="display:inline-block;margin:6px 0">'
        f'<img src="{data_url}" '
        f'style="max-width:100%;cursor:zoom-in;border:1px solid #ddd;border-radius:4px"/>'
        f'</a>'
    )
