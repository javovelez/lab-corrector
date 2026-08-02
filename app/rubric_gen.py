"""
Auto-generación de rúbrica desde los notebooks de enunciado + solución.

Asume que los notebooks siguen la convención del framework `.lab.md`:
cell_ids estables de la forma `ej{N}-{rol}` con rol en
`{enunciado, code, pregunta, respuesta}` y un sufijo opcional
`-{algo}` para soportar múltiples ítems del mismo rol en un mismo
ejercicio. Ejemplos válidos:

    ej1-enunciado        ej1-code          ej1-pregunta         ej1-respuesta
    ej5-code-a           ej5-code-b        ej5-pregunta-2       ej5-respuesta-2
    ej8-enunciado        ej8-respuesta     (ejercicio solo-análisis sin pregunta)

Emite **schema v2** directamente: cada ejercicio lleva una lista
`items` con kind=code o kind=analysis. Pregunta y respuesta se parean
por sufijo; si un análisis no tiene pregunta explícita, la app cae al
enunciado del ejercicio.

Genera `rubric.expected` y `rubric.common_errors` con Claude (via
`claude-agent-sdk`) llamando **una vez por item**, no por ejercicio,
para que la rúbrica sea específica a cada pieza evaluable.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from textwrap import dedent

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    TextBlock,
    query,
)

from nbparse import cell_outputs, cell_source, find_cell, load_notebook

# El contrato de `cell_id` vive en el plugin `lab-notebook`, que es lo que se
# distribuye a los repos de materia. La app lo importa de ahí en vez de tener
# su propia copia: si derivaran, un notebook podría pasar el validador del
# docente y aun así no ser corregible acá.
sys.path.insert(0, str(
    Path(__file__).resolve().parent.parent
    / "plugins" / "lab-notebook" / "scripts"
))
from lab_contract import EJ_ID_RE, analysis_key as _contract_analysis_key  # noqa: E402


SYSTEM_PROMPT_RUBRIC = dedent("""
    Sos asistente experto en didáctica de programación. El docente te pasa
    un ítem corregible de un laboratorio (un bloque de código o una
    respuesta de análisis, con su enunciado y la solución oficial) y vos
    generás la parte evaluadora de la rúbrica: qué se espera y qué errores
    son frecuentes.

    Devolvé ESTRICTAMENTE un objeto JSON con esta forma, sin texto alrededor:

    {
      "expected": "...",
      "common_errors": ["...", "..."]
    }

    Reglas:
    - "expected" es un resumen muy conciso (1–3 oraciones) de qué debería
      hacer el código/respuesta del alumno, referenciando funciones y
      variables concretas de la solución oficial.
    - "common_errors" es una lista de 3 a 6 errores típicos que un alumno
      cometería en este ítem. Priorizá los que cambian el resultado
      (no cosméticos). Si hay un error crítico (que rompe el ítem),
      marcalo al inicio con "CRÍTICO: ...".
    - Español rioplatense con voseo. Sin emoticones.
    - NO incluyas comentarios fuera del JSON. NO uses markdown. NO envuelvas
      el JSON en fences.
""").strip()


# ─── Utilidades internas ──────────────────────────────────────────────────────

def _extract_title_from_enunciado(md: str) -> str:
    """Toma el primer heading del enunciado como título del ejercicio."""
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if "—" in title:
                title = title.split("—", 1)[1].strip()
            return title
    return ""


def _graded_outputs(sol_code_cell: dict | None) -> list[str]:
    """Infiere qué tipo de outputs evaluar a partir de la solución ejecutada."""
    kinds = set()
    for out in cell_outputs(sol_code_cell):
        if out["kind"] == "image":
            kinds.add("image")
        elif out["kind"] == "text":
            kinds.add("text")
    return sorted(kinds)


def _analysis_key(ej_id: str, sufijo: str | None) -> str:
    return _contract_analysis_key(ej_id, sufijo)


# ─── Scan: notebook → esqueleto v2 (sin rubrics aún) ──────────────────────────

def scan_ejercicios(enunciado_nb: dict, solucion_nb: dict) -> list[dict]:
    """Escanea cell_ids del enunciado y devuelve la estructura v2 sin rubric.

    Para cada ejercicio:
      - `enunciado_cell`
      - `items`: lista de kind=code y kind=analysis, en el orden en que
        aparecen en el notebook (primero los codes, después los análisis).
      - Rubric queda vacío hasta que `generate_rubrica` llame a Claude.
    """
    # Acumuladores por ejercicio.
    por_ej: dict[int, dict] = {}

    for cell in enunciado_nb.get("cells", []):
        cid = cell.get("id", "")
        m = EJ_ID_RE.match(cid)
        if not m:
            continue
        n_str, role, sufijo = m.group(1), m.group(2), m.group(3)
        n = int(n_str)
        slot = por_ej.setdefault(n, {
            "id": f"ej{n}",
            "enunciado_cell": None,
            "codes": [],           # [(sufijo, cell_id)] en orden de aparición
            "preguntas": {},       # sufijo → cell_id
            "respuestas": {},      # sufijo → cell_id
        })
        if role == "enunciado":
            # Solo tomamos el primero; sufijos en enunciado no se esperan.
            if slot["enunciado_cell"] is None:
                slot["enunciado_cell"] = cid
        elif role == "code":
            slot["codes"].append((sufijo, cid))
        elif role == "pregunta":
            slot["preguntas"][sufijo] = cid
        elif role == "respuesta":
            slot["respuestas"][sufijo] = cid

    ejercicios: list[dict] = []
    for n in sorted(por_ej):
        slot = por_ej[n]
        if slot["enunciado_cell"] is None:
            # Sin enunciado no podemos armar un ejercicio corregible.
            continue

        items: list[dict] = []

        # Items code — uno por cada cell_id de código.
        for sufijo, cid in slot["codes"]:
            item: dict = {
                "kind": "code",
                "key": cid,
                "code_cell": cid,
            }
            outs = _graded_outputs(find_cell(solucion_nb, cid))
            if outs:
                item["graded_outputs"] = outs
            items.append(item)

        # Items analysis — un item por cada sufijo presente en pregunta o respuesta.
        sufijos_analisis = sorted(
            set(slot["preguntas"]) | set(slot["respuestas"]),
            key=lambda s: ("" if s is None else s),
        )
        for sufijo in sufijos_analisis:
            preg_id = slot["preguntas"].get(sufijo)
            resp_id = slot["respuestas"].get(sufijo)
            if preg_id is None and resp_id is None:
                continue  # no debería pasar por construcción
            item = {
                "kind": "analysis",
                "key": _analysis_key(slot["id"], sufijo),
            }
            if preg_id:
                item["pregunta_cell"] = preg_id
            if resp_id:
                item["answer_cell"] = resp_id
            items.append(item)

        if not items:
            # Enunciado suelto sin code ni análisis: nada que corregir.
            continue

        enun_src = cell_source(find_cell(enunciado_nb, slot["enunciado_cell"]))
        ejercicios.append({
            "id":             slot["id"],
            "titulo":         _extract_title_from_enunciado(enun_src),
            "enunciado_cell": slot["enunciado_cell"],
            "items":          items,
        })

    return ejercicios


# ─── Prompt por item ──────────────────────────────────────────────────────────

def _build_prompt_code(
    *,
    titulo: str,
    enunciado: str,
    code_sol: str,
) -> str:
    sections = [
        ("EJERCICIO", titulo or "(sin título)"),
        ("ITEM", "código"),
        ("ENUNCIADO DEL EJERCICIO", enunciado.strip() or "(vacío)"),
        ("CÓDIGO DE LA SOLUCIÓN OFICIAL", code_sol.strip() or "(sin código)"),
    ]
    body = "\n\n".join(f"{t}\n{'=' * len(t)}\n{c}" for t, c in sections)
    return (
        body
        + "\n\nGenerá el JSON con `expected` y `common_errors` evaluando "
        "la implementación de código."
    )


def _build_prompt_analysis(
    *,
    titulo: str,
    enunciado: str,
    pregunta: str,
    answer_sol: str,
    codigo_contexto: str,
) -> str:
    sections = [
        ("EJERCICIO", titulo or "(sin título)"),
        ("ITEM", "respuesta de análisis"),
        ("ENUNCIADO DEL EJERCICIO", enunciado.strip() or "(vacío)"),
    ]
    if pregunta.strip():
        sections.append(("PREGUNTA DE ANÁLISIS", pregunta.strip()))
    else:
        sections.append((
            "PREGUNTA DE ANÁLISIS",
            "(la pregunta vive en el enunciado del ejercicio)",
        ))
    sections.append((
        "RESPUESTA OFICIAL",
        answer_sol.strip() or "(sin respuesta oficial)",
    ))
    if codigo_contexto.strip():
        sections.append((
            "CÓDIGO DEL EJERCICIO (contexto)",
            codigo_contexto.strip(),
        ))
    body = "\n\n".join(f"{t}\n{'=' * len(t)}\n{c}" for t, c in sections)
    return (
        body
        + "\n\nGenerá el JSON con `expected` y `common_errors` evaluando "
        "la calidad conceptual de la respuesta."
    )


# ─── Llamada a Claude ─────────────────────────────────────────────────────────

async def _ask_claude_rubric(prompt: str) -> dict:
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT_RUBRIC,
        tools=[],
        setting_sources=[],
        permission_mode="default",
        max_turns=1,
    )
    parts: list[str] = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    raw = "".join(parts).strip()
    # Toleramos fences aunque pidamos lo contrario.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    return json.loads(raw.strip())


# ─── Entry point ──────────────────────────────────────────────────────────────

def generate_rubrica(
    *,
    title: str,
    enunciado_nb_path: Path,
    solucion_nb_path: Path,
    progress=None,   # callback(i, total, msg) — opcional
) -> dict:
    """Genera la rúbrica completa en schema v2 llamando a la IA una vez por item."""
    enunciado_nb = load_notebook(enunciado_nb_path)
    solucion_nb  = load_notebook(solucion_nb_path)
    ejercicios = scan_ejercicios(enunciado_nb, solucion_nb)

    # Armamos lista plana [(ej, item)] para el progress y el loop.
    plan: list[tuple[dict, dict]] = [
        (ej, it) for ej in ejercicios for it in ej["items"]
    ]
    total = len(plan)

    for i, (ej, it) in enumerate(plan, start=1):
        titulo_ej = ej.get("titulo") or ej["id"]
        if progress:
            progress(i, total, f"{it['key']} — {titulo_ej}")

        enun_src = cell_source(find_cell(enunciado_nb, ej["enunciado_cell"]))

        if it["kind"] == "code":
            code_sol = cell_source(find_cell(solucion_nb, it["code_cell"]))
            prompt = _build_prompt_code(
                titulo=titulo_ej,
                enunciado=enun_src,
                code_sol=code_sol,
            )
        else:  # analysis
            preg_src = cell_source(
                find_cell(enunciado_nb, it.get("pregunta_cell") or "")
            )
            answer_sol = cell_source(
                find_cell(solucion_nb, it.get("answer_cell") or "")
            )
            # Códigos del mismo ejercicio como contexto (solución oficial).
            code_contexto_parts: list[str] = []
            for sibling in ej["items"]:
                if sibling.get("kind") != "code":
                    continue
                sib_src = cell_source(find_cell(solucion_nb, sibling["code_cell"]))
                if sib_src.strip():
                    code_contexto_parts.append(
                        f"# celda {sibling['code_cell']}\n{sib_src}"
                    )
            prompt = _build_prompt_analysis(
                titulo=titulo_ej,
                enunciado=enun_src,
                pregunta=preg_src,
                answer_sol=answer_sol,
                codigo_contexto="\n\n".join(code_contexto_parts),
            )

        try:
            rubric_part = asyncio.run(_ask_claude_rubric(prompt))
        except (ClaudeSDKError, json.JSONDecodeError) as e:
            rubric_part = {
                "expected": "(generación automática falló — completar a mano)",
                "common_errors": [f"(error: {e})"],
            }

        it["rubric"] = {
            "expected": rubric_part.get("expected", ""),
            "common_errors": rubric_part.get("common_errors", []),
        }

    return {"title": title, "ejercicios": ejercicios}
