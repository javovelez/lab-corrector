"""
Persistencia filesystem del feedback, escoped a un workdir.

Un archivo markdown por (ítem corregible × grupo), adentro del grupo:
    <workdir>/grupo_NN/feedback/{item_key}.md

Estados encodados en el filesystem:
    - archivo ausente                        → pendiente
    - archivo con marcador `SIN_OBS_MARKER`  → sin observaciones (el txt final lo omite)
    - archivo con contenido                  → con observación (el txt final lo incluye)
"""
from __future__ import annotations

from pathlib import Path

STATUS_PENDIENTE = "pendiente"
STATUS_OK        = "sin_observaciones"
STATUS_OBS       = "con_observacion"

SIN_OBS_MARKER = "<!-- sin-observaciones -->"

FEEDBACK_DIRNAME = "feedback"


def feedback_dir(workdir: Path, grupo: str) -> Path:
    return workdir / grupo / FEEDBACK_DIRNAME


def feedback_path(workdir: Path, grupo: str, item_key: str) -> Path:
    return feedback_dir(workdir, grupo) / f"{item_key}.md"


def read_feedback(path: Path) -> tuple[str, str]:
    """Devuelve `(status, content)`. Content queda vacío salvo para observaciones reales."""
    if not path.exists():
        return STATUS_PENDIENTE, ""
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if stripped == "" or stripped == SIN_OBS_MARKER:
        return STATUS_OK, ""
    return STATUS_OBS, raw


def save_observation(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = text.strip() + "\n"
    path.write_text(content, encoding="utf-8")


def save_sin_observaciones(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SIN_OBS_MARKER + "\n", encoding="utf-8")


def clear_feedback(path: Path) -> None:
    if path.exists():
        path.unlink()
