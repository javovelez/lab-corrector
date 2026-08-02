#!/usr/bin/env python3
"""
Reporte de impacto: qué materias quedaron desalineadas del contrato.

Lee `materias/registry.yaml`, compara la `contract_version` de cada
materia contra la de `lab_contract.py`, y para las que están atrasadas
muestra el impacto declarado en el CHANGELOG de cada versión intermedia.

Con `--validar` corre además `lab_validate.py` sobre los notebooks de
cada materia y resume cuántos pasan.

Uso:
    python materias/check.py
    python materias/check.py --validar
    python materias/check.py --validar --materia rnp
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "plugins" / "lab-notebook" / "scripts"
CHANGELOG = (
    ROOT / "plugins" / "lab-notebook" / "skills" / "lab-notebook"
    / "reference" / "CHANGELOG.md"
)
REGISTRY = ROOT / "materias" / "registry.yaml"

sys.path.insert(0, str(SCRIPTS))
from lab_contract import CONTRACT_VERSION  # noqa: E402


PATHS_LOCAL = ROOT / "materias" / "paths.local.yaml"


def load_registry() -> list[dict]:
    """Registro versionado + overlay de paths locales (no versionado)."""
    try:
        import yaml
    except ImportError:
        print(
            "Falta pyyaml. Corré: app/.venv/bin/python materias/check.py",
            file=sys.stderr,
        )
        sys.exit(2)

    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    materias = data.get("materias", [])

    # Los paths locales viven aparte: el registro se versiona en un repo
    # público y cada máquina tiene los repos en otro lado.
    if PATHS_LOCAL.is_file():
        overlay = (yaml.safe_load(PATHS_LOCAL.read_text(encoding="utf-8")) or {})
        paths = overlay.get("paths") or {}
    else:
        paths = {}
        print(
            f"Aviso: no existe {PATHS_LOCAL.name}. Copiá "
            f"materias/paths.local.example.yaml y ajustá las rutas para poder "
            f"validar los notebooks.\n",
            file=sys.stderr,
        )

    for m in materias:
        p = paths.get(m.get("id"))
        m["path_local"] = str(Path(p).expanduser()) if p else ""
    return materias


def parse_version(v: str | None) -> tuple[int, ...]:
    if not v:
        return (0, 0, 0)
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def changelog_impactos() -> dict[str, str]:
    """Devuelve {version: texto del impacto} leído del CHANGELOG."""
    if not CHANGELOG.is_file():
        return {}
    texto = CHANGELOG.read_text(encoding="utf-8")
    impactos: dict[str, str] = {}
    bloques = re.split(r"^## ", texto, flags=re.MULTILINE)[1:]
    for bloque in bloques:
        version = bloque.split(None, 1)[0].strip()
        m = re.search(r"\*\*Impacto:\*\*\s*(.+?)(?=\n\n|\Z)", bloque, re.DOTALL)
        impactos[version] = (
            " ".join(m.group(1).split()) if m
            else "(esta versión no declara impacto en el CHANGELOG)"
        )
    return impactos


def notebooks_de(materia: dict) -> list[Path]:
    base = Path(materia.get("path_local", ""))
    layout = materia.get("layout") or {}
    sub = layout.get("enunciados", "Laboratorios")
    carpeta = base / sub
    if not carpeta.is_dir():
        return []
    return sorted(p for p in carpeta.glob("*.ipynb") if "checkpoint" not in p.name)


def validar(materia: dict) -> tuple[int, int, list[str]]:
    """Corre lab_validate sobre cada notebook. Devuelve (ok, fallan, detalle)."""
    ok = fallan = 0
    detalle: list[str] = []
    for nb in notebooks_de(materia):
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "lab_validate.py"), str(nb)],
            capture_output=True, text=True,
        )
        resumen = ""
        for linea in r.stdout.splitlines():
            if "errores," in linea:
                resumen = linea.strip()
        if r.returncode == 0:
            ok += 1
        else:
            fallan += 1
        detalle.append(f"      {nb.name}: {resumen or 'sin salida'}")
    return ok, fallan, detalle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validar", action="store_true",
                    help="correr lab_validate.py sobre los notebooks de cada materia")
    ap.add_argument("--materia", help="filtrar por id de materia")
    args = ap.parse_args()

    materias = load_registry()
    if args.materia:
        materias = [m for m in materias if m.get("id") == args.materia]
        if not materias:
            print(f"No hay materia con id `{args.materia}` en el registro",
                  file=sys.stderr)
            sys.exit(2)

    impactos = changelog_impactos()
    actual = parse_version(CONTRACT_VERSION)

    print(f"Contrato actual: v{CONTRACT_VERSION}\n")

    desalineadas = 0
    for m in materias:
        mv = m.get("contract_version")
        estado = m.get("estado", "?")
        print(f"── {m['nombre']}  [{m.get('id')}]  ({estado})")

        base = Path(m.get("path_local", ""))
        if not base.is_dir():
            print(f"   repo no encontrado en {base}")
            print()
            continue

        if estado == "sin-decidir":
            print("   sin decidir si adopta el contrato — ver `notas` en el registro")
        elif mv is None:
            desalineadas += 1
            print("   sin adoptar el contrato todavía")
            print(f"   pendiente: crear .labconfig.yaml y migrar los cell_id "
                  f"al formato ejN-rol")
        elif parse_version(mv) < actual:
            desalineadas += 1
            print(f"   ATRASADA: está en v{mv}, el contrato está en v{CONTRACT_VERSION}")
            for version in sorted(impactos, key=parse_version):
                if parse_version(mv) < parse_version(version) <= actual:
                    print(f"   v{version} → {impactos[version]}")
        else:
            print(f"   al día (v{mv})")

        if args.validar:
            nbs = notebooks_de(m)
            if not nbs:
                print("   sin notebooks para validar en la carpeta de enunciados")
            else:
                ok, fallan, detalle = validar(m)
                print(f"   validación: {ok} pasan, {fallan} con errores")
                for linea in detalle:
                    print(linea)
        print()

    print(f"{len(materias)} materias, {desalineadas} desalineadas")
    sys.exit(1 if desalineadas else 0)


if __name__ == "__main__":
    main()
