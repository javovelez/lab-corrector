"""
Armado del archivo `grupo_XX.txt` de devolución final + scoring por grupo.

`build_grupo_txt` recorre los ítems del lab en orden y concatena SOLO las
observaciones (ignora pendientes y "sin observaciones"). Sin notas
numéricas, sin texto introductorio — directo al copy/paste en Moodle.

`compute_grupo_score` calcula el puntaje (en %) según el nivel de cada
observación: bien/sin_obs = 1pt, regular = 0.5pt, mal = 0pt. Una
observación legacy sin marker de nivel se trata como `regular`. Si la
entrega del grupo no existe, todos los ítems valen 0pt.
"""
from __future__ import annotations

from pathlib import Path

from grupos import notebook_path
from state import (
    LEVEL_BIEN,
    LEVEL_MAL,
    LEVEL_REGULAR,
    STATUS_OBS,
    STATUS_OK,
    STATUS_PENDIENTE,
    feedback_path,
    read_feedback,
)


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
        status, content, _level = read_feedback(fb)
        if status != STATUS_OBS:
            continue
        body = content.strip()
        if not body:
            # "Guardar como bien/regular/mal" sin texto: cuenta para el
            # puntaje pero no aparece en el txt final.
            continue
        header = f"Ej {item['ej_id'].removeprefix('ej')} ({item['tipo']}):"
        bloques.append(f"{header}\n{body}")
    return "\n\n".join(bloques)


def count_observaciones(
    *,
    workdir: Path,
    items: list[dict],
    grupo: str,
) -> int:
    """Cuenta observaciones con texto real (las que aparecen en el txt)."""
    n = 0
    for item in items:
        fb = feedback_path(workdir, grupo, item["key"])
        status, content, _ = read_feedback(fb)
        if status == STATUS_OBS and content.strip():
            n += 1
    return n


def compute_grupo_score(
    *,
    workdir: Path,
    items: list[dict],
    grupo: str,
) -> tuple[float, int, float | None, int]:
    """Devuelve `(puntos, total, porcentaje, pendientes)`.

    - `puntos`: suma de puntos obtenidos (1 por verde, 0.5 por amarillo, 0 por rojo).
    - `total`: cantidad de ítems del lab.
    - `porcentaje`: `(puntos/total)*100`. Es `None` cuando hay ítems
      pendientes y la entrega existe (corrección incompleta — todavía no
      tiene sentido reportar un porcentaje). Si la entrega no existe,
      todos los ítems valen 0 y el porcentaje es 0.0.
    - `pendientes`: cantidad de ítems sin corregir todavía.
    """
    nb = notebook_path(workdir, grupo)
    total = len(items)
    if nb is None:
        # Entrega faltante: todos los ítems se cuentan como "mal" (0pt).
        return 0.0, total, 0.0, 0

    puntos = 0.0
    pendientes = 0
    for item in items:
        fb = feedback_path(workdir, grupo, item["key"])
        status, _, level = read_feedback(fb)
        if status == STATUS_PENDIENTE:
            pendientes += 1
        elif status == STATUS_OK:
            puntos += 1.0
        elif status == STATUS_OBS and level == LEVEL_BIEN:
            puntos += 1.0
        elif status == STATUS_OBS and level == LEVEL_REGULAR:
            puntos += 0.5
        elif status == STATUS_OBS and level == LEVEL_MAL:
            puntos += 0.0
        else:
            # Observación legacy sin nivel: cuenta como pendiente, así el
            # porcentaje no se reporta hasta que el corrector la clasifique.
            pendientes += 1

    if pendientes > 0:
        return puntos, total, None, pendientes
    pct = (puntos / total) * 100 if total > 0 else 0.0
    return puntos, total, pct, 0
