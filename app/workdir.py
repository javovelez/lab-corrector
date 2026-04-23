"""
Configuración de un workdir de corrección.

Cada workdir es una carpeta elegida por el docente donde vive una tanda
de corrección (el zip de Moodle, las entregas extraídas y renombradas,
el feedback y los txt finales). El estado interno de la app vive en
`<workdir>/.corrector/config.json`; la rúbrica puede venir de afuera o
haber sido auto-generada dentro de `<workdir>/.corrector/rubrica.yaml`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


CONFIG_DIRNAME = ".corrector"
CONFIG_FILENAME = "config.json"


@dataclass
class WorkdirConfig:
    title: str                       # p.ej. "Laboratorio 3 — Transferencia"
    notebook_enunciado: str          # path absoluto al .ipynb de enunciado
    notebook_solucion: str           # path absoluto al .ipynb de solución
    rubrica: str                     # path absoluto al .rubric.yaml

    def to_dict(self) -> dict:
        return asdict(self)


def config_dir(workdir: Path) -> Path:
    return workdir / CONFIG_DIRNAME


def config_path(workdir: Path) -> Path:
    return config_dir(workdir) / CONFIG_FILENAME


def has_config(workdir: Path) -> bool:
    return config_path(workdir).exists()


def load_config(workdir: Path) -> WorkdirConfig:
    data = json.loads(config_path(workdir).read_text(encoding="utf-8"))
    return WorkdirConfig(
        title=data["title"],
        notebook_enunciado=data["notebook_enunciado"],
        notebook_solucion=data["notebook_solucion"],
        rubrica=data["rubrica"],
    )


def save_config(workdir: Path, cfg: WorkdirConfig) -> None:
    config_dir(workdir).mkdir(parents=True, exist_ok=True)
    config_path(workdir).write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_config(cfg: WorkdirConfig) -> list[str]:
    """Devuelve lista de errores; vacía = config OK."""
    errors: list[str] = []
    for field_name in ("notebook_enunciado", "notebook_solucion", "rubrica"):
        p = Path(getattr(cfg, field_name))
        if not p.is_absolute():
            errors.append(f"`{field_name}` debe ser un path absoluto: {p}")
        elif not p.exists():
            errors.append(f"`{field_name}` no existe: {p}")
    return errors
