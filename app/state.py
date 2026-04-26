"""
Persistencia filesystem del feedback, escoped a un workdir.

Un archivo markdown por (ítem corregible × grupo), adentro del grupo:
    <workdir>/grupo_NN/feedback/{item_key}.md          ← devolución validada
    <workdir>/grupo_NN/feedback/{item_key}.draft.md    ← borrador IA (opcional)

Estados encodados en el filesystem (sobre el `.md` validado):
    - archivo ausente                        → pendiente
    - archivo con marcador `SIN_OBS_MARKER`  → sin observaciones (el txt final lo omite)
    - archivo con contenido                  → con observación (el txt final lo incluye)

El `.draft.md` es independiente: lo escribe la IA y lo lee la UI para
precargar el textarea cuando todavía no hay devolución validada. Nunca
se exporta; el `grupo_NN.txt` solo mira el `.md` validado.

Adicionalmente, cada grupo tiene un archivo de overrides de celda:
    <workdir>/grupo_NN/cell_overrides.json

Mapea `expected_id → actual_id` para casos en los que el alumno borró
la celda con id estable y rehizo la respuesta en otra celda (o mezcló
todo en una celda existente). El notebook del grupo NO se modifica;
toda la app resuelve el id real a través de `resolved_id`.
"""
from __future__ import annotations

import json
from pathlib import Path

STATUS_PENDIENTE = "pendiente"
STATUS_OK        = "sin_observaciones"
STATUS_OBS       = "con_observacion"

# Sub-niveles para STATUS_OBS — definen color (verde/amarillo/rojo) y
# puntaje (1 / 0.5 / 0). Una observación legacy sin marker se trata como
# `regular` para retrocompat.
LEVEL_BIEN    = "bien"
LEVEL_REGULAR = "regular"
LEVEL_MAL     = "mal"

SIN_OBS_MARKER = "<!-- sin-observaciones -->"
AI_OK_MARKER   = "<!-- ai-ok -->"

LEVEL_MARKERS: dict[str, str] = {
    LEVEL_BIEN:    "<!-- nivel: bien -->",
    LEVEL_REGULAR: "<!-- nivel: regular -->",
    LEVEL_MAL:     "<!-- nivel: mal -->",
}
LEVEL_BY_MARKER: dict[str, str] = {v: k for k, v in LEVEL_MARKERS.items()}

FEEDBACK_DIRNAME = "feedback"


def feedback_dir(workdir: Path, grupo: str) -> Path:
    return workdir / grupo / FEEDBACK_DIRNAME


def feedback_path(workdir: Path, grupo: str, item_key: str) -> Path:
    return feedback_dir(workdir, grupo) / f"{item_key}.md"


def draft_path(workdir: Path, grupo: str, item_key: str) -> Path:
    return feedback_dir(workdir, grupo) / f"{item_key}.draft.md"


def read_feedback(path: Path) -> tuple[str, str, str | None]:
    """Devuelve `(status, content, level)`.

    `level` solo aplica a `STATUS_OBS`: es uno de "bien"/"regular"/"mal"
    si el archivo tiene marker, o `None` para observaciones legacy
    (escritas antes de que existieran los niveles). En todos los demás
    estados es `None`.
    """
    if not path.exists():
        return STATUS_PENDIENTE, "", None
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if stripped == "" or stripped == SIN_OBS_MARKER:
        return STATUS_OK, "", None

    lines = stripped.splitlines()
    level: str | None = None
    body = stripped
    if lines and lines[0].strip() in LEVEL_BY_MARKER:
        level = LEVEL_BY_MARKER[lines[0].strip()]
        body = "\n".join(lines[1:]).strip()
    return STATUS_OBS, body, level


def save_observation(path: Path, text: str, level: str | None) -> None:
    """Persiste una observación, opcionalmente con su nivel.

    `level` debe ser uno de `LEVEL_BIEN`/`LEVEL_REGULAR`/`LEVEL_MAL` o
    `None`. Si es `None`, se escribe el cuerpo sin marker — esto
    permite guardar la observación sin haber elegido todavía un puntaje
    (el archivo queda como "sin clasificar"). El corrector puede
    asignarle el nivel después y se reescribe con el marker.
    """
    if level is not None and level not in LEVEL_MARKERS:
        raise ValueError(f"level inválido: {level!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = text.strip()
    if level is None:
        content = body + "\n"
    else:
        marker = LEVEL_MARKERS[level]
        content = f"{marker}\n{body}\n"
    path.write_text(content, encoding="utf-8")


def save_sin_observaciones(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SIN_OBS_MARKER + "\n", encoding="utf-8")


def clear_feedback(path: Path) -> None:
    if path.exists():
        path.unlink()


def read_draft(path: Path) -> tuple[bool, str]:
    """Devuelve `(ai_ok, content)`.

    - Si el archivo no existe, `(False, "")`.
    - Si la IA marcó el ítem como correcto, `(True, "")` — la UI muestra
      un hint en lugar de cargar texto en el textarea.
    - Si la IA escribió un borrador real, `(False, texto)`.
    """
    if not path.exists():
        return False, ""
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if stripped == AI_OK_MARKER:
        return True, ""
    return False, raw


def save_draft(path: Path, text: str) -> None:
    """Persiste un borrador IA. Si `text` es 'OK' lo guarda como marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stripped = (text or "").strip()
    if stripped == "OK":
        path.write_text(AI_OK_MARKER + "\n", encoding="utf-8")
    else:
        path.write_text(stripped + "\n", encoding="utf-8")


def clear_draft(path: Path) -> None:
    if path.exists():
        path.unlink()


# ─── Cell overrides ─────────────────────────────────────────────────────────

CELL_OVERRIDES_FILENAME = "cell_overrides.json"


def cell_overrides_path(workdir: Path, grupo: str) -> Path:
    return workdir / grupo / CELL_OVERRIDES_FILENAME


def read_cell_overrides(workdir: Path, grupo: str) -> dict[str, str]:
    """Devuelve el mapa `expected_id → actual_id`. Vacío si no existe."""
    p = cell_overrides_path(workdir, grupo)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def set_cell_override(workdir: Path, grupo: str, expected_id: str, actual_id: str) -> None:
    p = cell_overrides_path(workdir, grupo)
    p.parent.mkdir(parents=True, exist_ok=True)
    overrides = read_cell_overrides(workdir, grupo)
    overrides[expected_id] = actual_id
    p.write_text(json.dumps(overrides, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clear_cell_override(workdir: Path, grupo: str, expected_id: str) -> None:
    overrides = read_cell_overrides(workdir, grupo)
    if expected_id not in overrides:
        return
    overrides.pop(expected_id)
    p = cell_overrides_path(workdir, grupo)
    if overrides:
        p.write_text(json.dumps(overrides, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif p.exists():
        p.unlink()


def resolved_id(expected_id: str, overrides: dict[str, str]) -> str:
    """Devuelve el id real a usar (override si existe, expected si no)."""
    return overrides.get(expected_id, expected_id)
