"""
Armado del archivo `grupo_XX.txt` de devolución final.

Recorre los ítems del lab en orden y concatena SOLO las observaciones
(ignora pendientes y "sin observaciones"). Sin notas numéricas, sin
texto introductorio — directo al copy/paste en Moodle.
"""
from __future__ import annotations

from pathlib import Path

from state import STATUS_OBS, feedback_path, read_feedback


def build_grupo_txt(
    *,
    workdir: Path,
    items: list[dict],
    grupo: str,
) -> str:
    """Devuelve el contenido del txt. Si no hay observaciones, devuelve cadena vacía."""
    bloques: list[str] = []
    for item in items:
        fb = feedback_path(workdir, grupo, item["key"])
        status, content = read_feedback(fb)
        if status != STATUS_OBS:
            continue
        header = f"Ej {item['ej_id'].removeprefix('ej')} ({item['tipo']}):"
        bloques.append(f"{header}\n{content.strip()}")
    return "\n\n".join(bloques)


def count_observaciones(
    *,
    workdir: Path,
    items: list[dict],
    grupo: str,
) -> int:
    n = 0
    for item in items:
        fb = feedback_path(workdir, grupo, item["key"])
        status, _ = read_feedback(fb)
        if status == STATUS_OBS:
            n += 1
    return n
