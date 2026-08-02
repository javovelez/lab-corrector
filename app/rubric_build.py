#!/usr/bin/env python3
"""
Generador CLI de rúbrica a partir de enunciado + solución ya ejecutada.

Uso:
    app/.venv/bin/python tools/rubric_build.py <lab_id>

donde <lab_id> es el número/sufijo del laboratorio (ej. "3b"). Asume el
layout estándar _TPS/ y escribe la rúbrica en:
    _TPS/rubricas/Laboratorio_<lab_id>.rubric.yaml

Requiere las dependencias del venv de la app (claude-agent-sdk, pyyaml).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from rubric import save_rubrica        # noqa: E402
from rubric_gen import generate_rubrica  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Uso: python tools/rubric_build.py <lab_id>", file=sys.stderr)
        print("  ej: python tools/rubric_build.py 3b", file=sys.stderr)
        sys.exit(1)

    lab_id = sys.argv[1]
    tps = ROOT / "_TPS"
    enun = tps / "Laboratorios" / f"Laboratorio_{lab_id}.ipynb"
    sol  = tps / "Soluciones"   / f"Laboratorio_{lab_id}_Solucion.ipynb"
    out  = tps / "rubricas"     / f"Laboratorio_{lab_id}.rubric.yaml"

    for p in (enun, sol):
        if not p.exists():
            print(f"No existe: {p}", file=sys.stderr)
            sys.exit(1)

    title = f"Laboratorio n° {lab_id}"

    def prog(i, total, msg):
        print(f"  ({i}/{total}) {msg}")

    print(f"Generando rúbrica para Lab {lab_id}...")
    rubrica = generate_rubrica(
        title=title,
        enunciado_nb_path=enun,
        solucion_nb_path=sol,
        progress=prog,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_rubrica(out, rubrica)
    print(f"\nGenerado: {out}")
    print(f"Ejercicios: {len(rubrica.get('ejercicios', []))}")


if __name__ == "__main__":
    main()
