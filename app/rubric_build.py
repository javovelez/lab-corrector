#!/usr/bin/env python3
"""
Generador CLI de rúbrica a partir de enunciado + solución ya ejecutada.

Es el mismo trabajo que hace el botón de auto-generación de la app; sirve
para cuando querés la rúbrica sin abrir la UI.

Uso:
    app/.venv/bin/python app/rubric_build.py <enunciado.ipynb> <solucion.ipynb> [-o salida.yaml]

Si no pasás `-o`, escribe al lado del enunciado con extensión
`.rubric.yaml`.

Los paths son explícitos a propósito: la app es agnóstica de la materia y
este script también. El layout de cada repo de materia está en su
`.labconfig.yaml`, no acá.

La solución tiene que estar **ejecutada y guardada con sus outputs**: de
ahí sale `graded_outputs`.

Requiere las dependencias del venv de la app (claude-agent-sdk, pyyaml).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rubric import save_rubrica          # noqa: E402
from rubric_gen import generate_rubrica  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Genera un .rubric.yaml desde un par enunciado/solución."
    )
    ap.add_argument("enunciado", type=Path, help="notebook de enunciado (.ipynb)")
    ap.add_argument("solucion", type=Path, help="notebook de solución ejecutado (.ipynb)")
    ap.add_argument("-o", "--out", type=Path, help="destino del .rubric.yaml")
    ap.add_argument("-t", "--title", help="título de la rúbrica")
    args = ap.parse_args()

    for p in (args.enunciado, args.solucion):
        if not p.is_file():
            print(f"No existe: {p}", file=sys.stderr)
            sys.exit(1)

    out = args.out or args.enunciado.with_suffix(".rubric.yaml")
    title = args.title or args.enunciado.stem.replace("_", " ")

    def prog(i, total, msg):
        print(f"  ({i}/{total}) {msg}")

    print(f"Generando rúbrica para {args.enunciado.name}...")
    rubrica = generate_rubrica(
        title=title,
        enunciado_nb_path=args.enunciado,
        solucion_nb_path=args.solucion,
        progress=prog,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_rubrica(out, rubrica)
    print(f"\nGenerado: {out}")
    print(f"Ejercicios: {len(rubrica.get('ejercicios', []))}")


if __name__ == "__main__":
    main()
