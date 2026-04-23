"""
Registry global de workdirs recientes.

Vive en `~/.lab_corrector/recent.json` y mantiene una lista de paths de
workdirs ordenados por recencia (el más reciente primero). La app
ofrece este listado en el landing para retomar una corrección en curso.
"""
from __future__ import annotations

import json
from pathlib import Path

REGISTRY_DIR = Path.home() / ".lab_corrector"
REGISTRY_FILE = REGISTRY_DIR / "recent.json"
MAX_RECENTS = 10


def _read() -> list[str]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if isinstance(x, str)]


def _write(paths: list[str]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(
        json.dumps(paths, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def list_recents() -> list[Path]:
    """Devuelve los workdirs que siguen existiendo en disco, en orden de recencia."""
    return [Path(p) for p in _read() if Path(p).is_dir()]


def touch(workdir: Path) -> None:
    """Mueve el workdir al tope de la lista. Crea el archivo si no existe."""
    s = str(workdir.resolve())
    cur = [p for p in _read() if p != s]
    cur.insert(0, s)
    _write(cur[:MAX_RECENTS])


def forget(workdir: Path) -> None:
    s = str(workdir.resolve())
    _write([p for p in _read() if p != s])
