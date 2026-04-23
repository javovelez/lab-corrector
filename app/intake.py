"""
Intake de entregas desde un zip de Moodle.

Moodle empaqueta cada grupo en una carpeta con nombre
`Grupo <n>_<moodle_id>_assignsubmission_file/`, con el(los) archivo(s)
entregados adentro. Este módulo toma el zip descargado y normaliza la
estructura adentro del workdir:

    <workdir>/
        grupo_01/entrega.ipynb
        grupo_02/entrega.ipynb
        ...

Archivos que no sean `.ipynb` se ignoran. El feedback existente (que
vive en `<workdir>/grupo_NN/feedback/`) se preserva aunque se re-importe
el mismo grupo.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


MOODLE_FOLDER_RE = re.compile(r"^Grupo\s+(\d+)_\d+_assignsubmission_file$")


@dataclass
class IntakeReport:
    imported: list[str] = field(default_factory=list)          # ["grupo_01", ...]
    skipped:  list[tuple[str, str]] = field(default_factory=list)  # (nombre_carpeta, motivo)
    warnings: list[tuple[str, str]] = field(default_factory=list)  # (grupo, mensaje)


def parse_moodle_group_number(folder_name: str) -> int | None:
    """Extrae el número de grupo de `Grupo N_<id>_assignsubmission_file`.

    Devuelve None si el nombre no matchea el patrón.
    """
    m = MOODLE_FOLDER_RE.match(folder_name.strip())
    if not m:
        return None
    return int(m.group(1))


def _iter_moodle_folders(root: Path):
    """Lista subdirectorios top-level que matchean el patrón de Moodle.

    Moodle a veces agrega un nivel intermedio (el nombre de la entrega).
    Si el zip contiene un solo directorio en la raíz, bajamos un nivel.
    """
    children = [p for p in root.iterdir() if p.is_dir()]
    candidates = [p for p in children if MOODLE_FOLDER_RE.match(p.name)]
    if candidates:
        return candidates
    if len(children) == 1:
        inner = children[0]
        return [p for p in inner.iterdir() if p.is_dir()]
    return children


def intake_zip(zip_path: Path, workdir: Path) -> IntakeReport:
    """Descomprime `zip_path` a un temp, copia el .ipynb de cada grupo al workdir.

    Escribe en `<workdir>/grupo_NN/entrega.ipynb`. Si ya existía
    `grupo_NN/entrega.ipynb` lo sobrescribe, pero el resto del contenido
    del grupo (en particular `grupo_NN/feedback/`) se preserva.
    """
    report = IntakeReport()
    if not zip_path.exists():
        report.skipped.append((zip_path.name, "el zip no existe"))
        return report
    if not zipfile.is_zipfile(zip_path):
        report.skipped.append((zip_path.name, "no es un zip válido"))
        return report

    workdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        folders = _iter_moodle_folders(tmp_path)
        for folder in sorted(folders):
            n = parse_moodle_group_number(folder.name)
            if n is None:
                report.skipped.append((folder.name, "nombre no matchea patrón de Moodle"))
                continue

            grupo_name = f"grupo_{n:02d}"
            ipynbs = sorted(folder.glob("*.ipynb"))
            if not ipynbs:
                report.skipped.append((grupo_name, f"sin .ipynb en `{folder.name}`"))
                continue
            if len(ipynbs) > 1:
                nombres = ", ".join(p.name for p in ipynbs)
                report.warnings.append((grupo_name, f"{len(ipynbs)} ipynb, tomé el primero ({nombres})"))

            target_dir = workdir / grupo_name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(ipynbs[0], target_dir / "entrega.ipynb")
            report.imported.append(grupo_name)

    report.imported.sort()
    return report
