"""Descubrimiento de grupos en `_correccion/lab{X}/entregas/`."""
from __future__ import annotations

from pathlib import Path


def list_grupos(entregas_dir: Path) -> list[str]:
    """Lista los subdirectorios `grupo_*` en orden alfabético."""
    if not entregas_dir.exists():
        return []
    return sorted(
        p.name for p in entregas_dir.iterdir()
        if p.is_dir() and p.name.startswith("grupo_")
    )


def notebook_path(entregas_dir: Path, grupo: str) -> Path | None:
    """Devuelve el único `.ipynb` dentro de `grupo_XX/`, o None si no hay o hay más de uno."""
    grupo_dir = entregas_dir / grupo
    if not grupo_dir.exists():
        return None
    ipynbs = list(grupo_dir.glob("*.ipynb"))
    if len(ipynbs) != 1:
        return None
    return ipynbs[0]
