"""
Parche one-off para normalizar Lab 2 (y notebooks anteriores al framework) a
la convención `ejN-pregunta` + `ejN-respuesta`.

El Lab 2 se escribió antes de separar la pregunta de análisis de la celda del
alumno: cada `ejN-pregunta` contiene el texto de la pregunta + el placeholder
`*(Escribí tu respuesta acá)*` en la misma celda. La app de corrección exige
celdas separadas. Este script divide cada `ejN-pregunta` en dos:

    ejN-pregunta    ← texto de la pregunta (sin placeholder)
    ejN-respuesta   ← placeholder o respuesta del alumno

Solo toca celdas markdown con id `ejN-pregunta`. Las celdas de código y sus
outputs quedan intactos. Es idempotente: si ya existe `ejN-respuesta` para
ese N, no re-divide.

Modos de uso:

    # Split estándar (enunciado / solución — todavía tienen el placeholder):
    python tools/lab2_split_pregunta.py _TPS/Laboratorios/Laboratorio_2.ipynb
    python tools/lab2_split_pregunta.py _TPS/Soluciones/Laboratorio_2_Solucion.ipynb

    # Split de una entrega (el alumno puede haber reemplazado el placeholder):
    python tools/lab2_split_pregunta.py <workdir>/grupo_05/entrega.ipynb \
        --reference _TPS/Laboratorios/Laboratorio_2.ipynb

Crea un backup `.bak` al lado del notebook antes de sobrescribirlo.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

PLACEHOLDER = "*(Escribí tu respuesta acá)*"
PREGUNTA_RE = re.compile(r"^ej(\d+)-pregunta$")


def _source_to_str(source) -> str:
    if isinstance(source, list):
        return "".join(source)
    return source or ""


def _str_to_source(s: str) -> list[str]:
    """Convierte un string al formato `source` de ipynb (lista con line endings)."""
    if not s:
        return [""]
    return s.splitlines(keepends=True)


def _extract_question_prefix(enunciado_source: str) -> str:
    """Del enunciado, devuelve el texto de la pregunta (antes del placeholder)."""
    if PLACEHOLDER in enunciado_source:
        return enunciado_source.split(PLACEHOLDER, 1)[0].rstrip()
    return enunciado_source.rstrip()


def _build_reference_map(reference_nb: dict) -> dict[str, str]:
    """Extrae la pregunta canónica por cada `ejN-pregunta` del notebook de referencia."""
    ref: dict[str, str] = {}
    for cell in reference_nb.get("cells", []):
        cid = cell.get("id", "")
        if not PREGUNTA_RE.match(cid) or cell.get("cell_type") != "markdown":
            continue
        ref[cid] = _extract_question_prefix(_source_to_str(cell.get("source", "")))
    return ref


def _split_source(source: str, reference_question: str | None) -> tuple[str, str]:
    """Devuelve (pregunta, respuesta). Lógica de fallback en varios niveles."""
    # 1. Si hay referencia y la source arranca con la pregunta → split exacto.
    if reference_question:
        q = reference_question.strip()
        if source.strip().startswith(q):
            respuesta = source.strip()[len(q):].strip()
            return q, respuesta or PLACEHOLDER

    # 2. Si hay placeholder → split por placeholder.
    if PLACEHOLDER in source:
        idx = source.index(PLACEHOLDER)
        return source[:idx].rstrip(), source[idx:].strip()

    # 3. Ni referencia útil ni placeholder. Asumimos que el alumno borró la
    #    pregunta y escribió solo la respuesta. Usamos la referencia (si hay)
    #    como pregunta canónica; si no, dejamos la source como "pregunta" sin
    #    respuesta (no podemos inventar).
    if reference_question:
        return reference_question.strip(), source.strip()
    return source.rstrip(), ""


def process_notebook(nb_path: Path, reference_nb: dict | None = None,
                     backup: bool = True) -> int:
    """Procesa un notebook in place. Devuelve cantidad de celdas divididas."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Qué ejercicios ya tienen ejN-respuesta → saltamos.
    existing_respuestas = {
        c.get("id", "") for c in nb.get("cells", [])
        if c.get("id", "").endswith("-respuesta")
    }

    ref_map = _build_reference_map(reference_nb) if reference_nb else {}

    new_cells: list[dict] = []
    splits = 0
    for cell in nb["cells"]:
        cid = cell.get("id", "")
        m = PREGUNTA_RE.match(cid)
        if not m or cell.get("cell_type") != "markdown":
            new_cells.append(cell)
            continue

        n = m.group(1)
        respuesta_id = f"ej{n}-respuesta"
        if respuesta_id in existing_respuestas:
            # Ya procesado — mantenemos tal cual.
            new_cells.append(cell)
            continue

        src = _source_to_str(cell.get("source", ""))
        ref_q = ref_map.get(cid)
        pregunta_text, respuesta_text = _split_source(src, ref_q)

        pregunta_cell = dict(cell)
        pregunta_cell["source"] = _str_to_source(pregunta_text)

        respuesta_cell = {
            "cell_type": "markdown",
            "id": respuesta_id,
            "metadata": {},
            "source": _str_to_source(respuesta_text or PLACEHOLDER),
        }

        new_cells.append(pregunta_cell)
        new_cells.append(respuesta_cell)
        splits += 1

    if splits == 0:
        print(f"[skip] {nb_path}: nada que hacer.")
        return 0

    if backup:
        bak = nb_path.with_name(nb_path.name + ".bak")
        if not bak.exists():
            shutil.copy2(nb_path, bak)

    nb["cells"] = new_cells
    nb_path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] {nb_path}: {splits} celda(s) dividida(s).")
    return splits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("notebook", type=Path, help="Notebook a procesar (in place).")
    ap.add_argument(
        "--reference", type=Path, default=None,
        help="Notebook de enunciado (con `ejN-pregunta` originales) para split por referencia. "
             "Úsalo para entregas donde el alumno pudo haber reemplazado el placeholder.",
    )
    ap.add_argument(
        "--no-backup", action="store_true",
        help="No crear archivo .bak (por defecto sí).",
    )
    args = ap.parse_args()

    ref_nb = None
    if args.reference:
        ref_nb = json.loads(args.reference.read_text(encoding="utf-8"))

    process_notebook(args.notebook, reference_nb=ref_nb, backup=not args.no_backup)


if __name__ == "__main__":
    main()
