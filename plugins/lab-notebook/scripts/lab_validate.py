#!/usr/bin/env python3
"""
Validador de notebooks de laboratorio contra el contrato de la app.

Responde una sola pregunta: ¿la app de corrección va a poder corregir
este notebook? Aplica el mismo regex y la misma agrupación que usa
`rubric_gen` (los importa de `lab_contract.py`), así que un notebook que
pasa acá se corrige.

Uso:
    python lab_validate.py <enunciado.ipynb> [solucion.ipynb]
    python lab_validate.py <archivo.lab.md>

Salida: lista de hallazgos y un resumen de los ítems corregibles que va a
ver la app. Termina con código 1 si hay errores, 0 si solo hay avisos.

Un ERROR significa que la app va a fallar, o peor: que va a ignorar en
silencio algo que vos esperabas corregir. Un AVISO es una desviación de
la convención que no rompe nada.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab_contract import (  # noqa: E402
    CONTRACT_VERSION,
    EXPECTED_CELL_TYPE,
    PLACEHOLDER_ANSWER,
    PLACEHOLDER_CODE,
    ROLES,
    analysis_key,
    group_cells,
    parse_cell_id,
)

# Detección de typos en el rol. No alcanza con "empieza con ejN-": el contrato
# permite a propósito ids como `ej3-cierre` para celdas explicativas no
# evaluables, y marcarlos sería ruido. Solo marcamos cuando el id se vuelve
# válido al normalizarlo, o cuando el rol es una variante conocida.
RE_CASI = re.compile(r"^(ej|ejercicio)[\s_\-]?(\d+)[\s_\-](.+)$", re.IGNORECASE)

ROLES_TYPO = {
    "codigo": "code", "cod": "code", "código": "code", "codigos": "code",
    "coding": "code", "programa": "code",
    "question": "pregunta", "preg": "pregunta", "preguntas": "pregunta",
    "answer": "respuesta", "resp": "respuesta", "respuestas": "respuesta",
    "statement": "enunciado", "enun": "enunciado", "consigna": "enunciado",
}


def rol_sospechoso(cell_id: str):
    """Si el id parece un rol mal escrito, devuelve el rol que quiso ser."""
    m = RE_CASI.match(cell_id)
    if not m:
        return None
    n, resto = m.group(2), m.group(3)
    rol = re.split(r"[\s_\-]", resto)[0].lower()

    # Se vuelve válido al normalizar (mayúsculas, guion bajo, espacios).
    normalizado = f"ej{n}-" + re.sub(r"[\s_]+", "-", resto.lower())
    if parse_cell_id(normalizado) is not None:
        return normalizado

    if rol in ROLES_TYPO:
        sufijo = resto[len(rol):].lstrip(" _-")
        return f"ej{n}-{ROLES_TYPO[rol]}" + (f"-{sufijo}" if sufijo else "")
    return None

RE_EMOJI = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "]"
)


class Report:
    def __init__(self):
        self.errores: list[str] = []
        self.avisos: list[str] = []

    def error(self, msg: str):
        self.errores.append(msg)

    def aviso(self, msg: str):
        self.avisos.append(msg)


# ─── Carga ───────────────────────────────────────────────────────────────────
def load_cells_from_ipynb(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    return nb.get("cells", [])


def load_cells_from_labmd(path: str) -> list[dict]:
    """Compila el .lab.md en memoria y devuelve las celdas del enunciado."""
    import lab_build

    with open(path, encoding="utf-8") as f:
        text = f.read()
    _front, body = lab_build.parse_frontmatter(text)
    cells = []
    for c in lab_build.split_cells(body):
        if c["type"] == "code":
            enun, _sol = lab_build.split_code_cell(c["content"])
        else:
            enun, _sol = lab_build.split_markdown_cell(c["content"])
        cells.append({
            "id":        c["id"],
            "cell_type": c["type"],
            "source":    enun,
        })
    return cells


def cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


# ─── Chequeos ────────────────────────────────────────────────────────────────
def check_ids(cells: list[dict], rep: Report) -> None:
    """Ids duplicados y typos que hacen que una celda deje de ser corregible."""
    vistos: dict[str, int] = {}
    for i, cell in enumerate(cells):
        cid = cell.get("id", "")
        if not cid:
            rep.aviso(f"celda #{i} sin `id` — Jupyter le va a inventar uno al guardar")
            continue
        if cid in vistos:
            rep.error(
                f"`{cid}` duplicado (celdas #{vistos[cid]} y #{i}) — "
                f"la app corrige la primera e ignora la segunda en silencio"
            )
        vistos[cid] = i

        if parse_cell_id(cid) is None:
            quiso = rol_sospechoso(cid)
            if quiso:
                rep.error(
                    f"`{cid}` no matchea el contrato y la app lo va a ignorar "
                    f"en silencio. ¿Querías `{quiso}`? Roles válidos: "
                    f"{', '.join(ROLES)}, en minúscula y con guion medio"
                )


def check_tipos(cells: list[dict], rep: Report) -> None:
    """El rol del id tiene que coincidir con el tipo de celda de Jupyter."""
    for cell in cells:
        parsed = parse_cell_id(cell.get("id", ""))
        if parsed is None:
            continue
        _n, role, _sufijo = parsed
        esperado = EXPECTED_CELL_TYPE[role]
        real = cell.get("cell_type")
        if real != esperado:
            rep.error(
                f"`{cell['id']}` es una celda `{real}` pero el rol `{role}` "
                f"exige `{esperado}` — la app puede crashear al renderizarla"
            )


def check_estructura(por_ej: dict[int, dict], rep: Report) -> list[dict]:
    """Aplica la agrupación de la app y reporta lo que se descarta."""
    items_totales = []
    for n in sorted(por_ej):
        slot = por_ej[n]
        ej = slot["id"]

        if slot["enunciado_cell"] is None:
            rep.error(
                f"{ej}: hay celdas de ejercicio pero falta `{ej}-enunciado` — "
                f"la app descarta el ejercicio entero"
            )
            continue

        items = []
        for _sufijo, cid in slot["codes"]:
            items.append({"kind": "code", "key": cid})

        sufijos = sorted(
            set(slot["preguntas"]) | set(slot["respuestas"]),
            key=lambda s: ("" if s is None else s),
        )
        for sufijo in sufijos:
            preg = slot["preguntas"].get(sufijo)
            resp = slot["respuestas"].get(sufijo)
            if resp is None:
                rep.error(
                    f"{ej}: `{preg}` no tiene su celda de respuesta "
                    f"(`{ej}-respuesta{'-' + sufijo if sufijo else ''}`) — "
                    f"sin respuesta no hay nada que corregir y el ítem se descarta"
                )
                continue
            if preg is None:
                rep.aviso(
                    f"{ej}: `{resp}` no tiene pregunta propia — la app va a usar "
                    f"el enunciado como pregunta (correcto si es un ejercicio "
                    f"solo-análisis)"
                )
            items.append({"kind": "analysis", "key": analysis_key(ej, sufijo)})

        if not items:
            rep.error(
                f"{ej}: tiene enunciado pero ninguna celda corregible "
                f"(`{ej}-code` o `{ej}-respuesta`) — no va a aparecer en la rúbrica"
            )
            continue

        items_totales.extend([dict(it, ej=ej) for it in items])

    nums = sorted(por_ej)
    if nums:
        faltantes = [n for n in range(1, max(nums) + 1) if n not in por_ej]
        if faltantes:
            rep.aviso(
                "huecos en la numeración de ejercicios: "
                + ", ".join(f"ej{n}" for n in faltantes)
            )
    return items_totales


def check_placeholders(cells: list[dict], rep: Report) -> None:
    """En el enunciado, las celdas del alumno llevan el placeholder exacto."""
    for cell in cells:
        parsed = parse_cell_id(cell.get("id", ""))
        if parsed is None:
            continue
        _n, role, _sufijo = parsed
        texto = cell_text(cell)
        cid = cell["id"]

        if role == "code" and PLACEHOLDER_CODE not in texto:
            if texto.strip():
                rep.aviso(
                    f"`{cid}` no contiene el placeholder exacto "
                    f"`{PLACEHOLDER_CODE}` (¿andamiaje preescrito, o quedó "
                    f"código de la solución?)"
                )
            else:
                rep.aviso(f"`{cid}` está vacía — falta el placeholder `{PLACEHOLDER_CODE}`")

        if role == "respuesta" and PLACEHOLDER_ANSWER not in texto:
            rep.aviso(
                f"`{cid}` no contiene el placeholder exacto "
                f"`{PLACEHOLDER_ANSWER}`"
            )

        if role == "respuesta" and "?" in texto:
            rep.aviso(
                f"`{cid}` contiene un signo de pregunta — ¿quedó la pregunta "
                f"pegada en la celda de respuesta? Tienen que ir separadas"
            )


def check_estilo(cells: list[dict], rep: Report) -> None:
    for cell in cells:
        if RE_EMOJI.search(cell_text(cell)):
            rep.aviso(f"`{cell.get('id', '?')}` contiene emoticones (la convención los prohíbe)")


def check_solucion(cells_e: list[dict], cells_s: list[dict], rep: Report) -> None:
    """Comparación enunciado ↔ solución."""
    ids_e = {c.get("id") for c in cells_e if parse_cell_id(c.get("id", ""))}
    ids_s = {c.get("id") for c in cells_s if parse_cell_id(c.get("id", ""))}

    for cid in sorted(ids_e - ids_s):
        rep.error(f"`{cid}` está en el enunciado pero no en la solución")
    for cid in sorted(ids_s - ids_e):
        rep.error(f"`{cid}` está en la solución pero no en el enunciado")

    sin_outputs = 0
    for cell in cells_s:
        parsed = parse_cell_id(cell.get("id", ""))
        if parsed is None:
            continue
        _n, role, _sufijo = parsed
        texto = cell_text(cell)
        cid = cell["id"]

        if role == "code":
            if PLACEHOLDER_CODE in texto:
                rep.error(f"solución: `{cid}` todavía tiene `{PLACEHOLDER_CODE}` — sin resolver")
            if not cell.get("outputs"):
                sin_outputs += 1
        if role == "respuesta" and PLACEHOLDER_ANSWER in texto:
            rep.error(f"solución: `{cid}` todavía tiene el placeholder — sin responder")

    if sin_outputs:
        rep.aviso(
            f"solución: {sin_outputs} celda(s) de código sin outputs guardados. "
            f"Ejecutá el notebook entero y guardalo, o la rúbrica va a salir "
            f"sin `graded_outputs`"
        )


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    path_e = sys.argv[1]
    path_s = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isfile(path_e):
        print(f"No existe: {path_e}", file=sys.stderr)
        sys.exit(2)

    es_labmd = path_e.endswith(".lab.md")
    cells_e = load_cells_from_labmd(path_e) if es_labmd else load_cells_from_ipynb(path_e)

    rep = Report()
    check_ids(cells_e, rep)
    check_tipos(cells_e, rep)
    check_placeholders(cells_e, rep)
    check_estilo(cells_e, rep)
    items = check_estructura(group_cells(cells_e), rep)

    if path_s:
        if not os.path.isfile(path_s):
            print(f"No existe: {path_s}", file=sys.stderr)
            sys.exit(2)
        check_solucion(cells_e, load_cells_from_ipynb(path_s), rep)

    # ─── Reporte ───
    print(f"Contrato v{CONTRACT_VERSION} — {os.path.basename(path_e)}")
    if es_labmd:
        print("(validado sobre el enunciado compilado en memoria)")
    print()

    for msg in rep.errores:
        print(f"  ERROR  {msg}")
    for msg in rep.avisos:
        print(f"  aviso  {msg}")
    if rep.errores or rep.avisos:
        print()

    n_code = sum(1 for i in items if i["kind"] == "code")
    n_ana = sum(1 for i in items if i["kind"] == "analysis")
    n_ej = len({i["ej"] for i in items})
    print(f"La app va a ver: {n_ej} ejercicios, {len(items)} ítems corregibles "
          f"({n_code} de código, {n_ana} de análisis)")
    print(f"{len(rep.errores)} errores, {len(rep.avisos)} avisos")

    sys.exit(1 if rep.errores else 0)


if __name__ == "__main__":
    main()
