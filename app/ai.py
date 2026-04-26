"""
Generación de borradores de feedback vía `claude-agent-sdk`.

Usa la sesión local de Claude Code del usuario (sin API key). Ejecuta una
query one-shot, sin tools y sin cargar settings del proyecto — solo
prompt → texto.
"""
from __future__ import annotations

import asyncio
import json
import re
from textwrap import dedent
from typing import TypedDict

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    TextBlock,
    query,
)


SYSTEM_PROMPT = dedent("""
    Sos asistente de corrección de un laboratorio universitario de
    redes neuronales profundas. El docente te pasa la entrega de un grupo
    para un ejercicio puntual y vos generás un borrador corto de
    observación que el docente va a editar y mandarle al alumno.

    La entrega es de un GRUPO de alumnos (no de una persona sola).
    Dirigite siempre al grupo en plural ("ustedes"): "no construyeron",
    "les falta", "tienen", "fíjense". Nunca en singular ("no
    construiste", "te falta", "fijate").

    Reglas de estilo de salida:
    - Español rioplatense, segunda persona del plural con "ustedes"
      ("usen", "fíjense", "tienen").
    - Sin emoticones. Sin fórmulas de cortesía.
    - Tono directo y técnico. Nada de felicitaciones.
    - Mencioná SOLO lo que esté mal o amerite señalar. Si hay varios
      puntos independientes, una línea por punto.
    - No des notas numéricas ni calificaciones.
    - Referite a funciones, variables o líneas por su nombre real
      cuando ayude a ubicar el error.
    - Entre 1 y 4 oraciones por lo general.
    - SOLO señalá el error o la falta. NO des instrucciones de cómo
      arreglarlo, ni sugerencias, ni "agreguen X", ni "deberían usar Y",
      ni pasos a seguir. El docente y los alumnos se encargan de la
      corrección; vos solamente identificás qué está mal o falta.

    Contrastá siempre QUÉ SE ESPERA contra la ENTREGA DEL GRUPO.
    Las secciones marcadas como "(contexto)" son solo informativas: te
    ayudan a entender la pregunta o la consigna, pero no son lo que
    estás corrigiendo y no deben figurar en la observación.

    Si la entrega resuelve correctamente lo pedido y no hay nada que
    señalar, respondé EXACTAMENTE la palabra: OK
    (sin punto, sin nada más).
""").strip()


def build_prompt(
    *,
    ej_titulo: str,
    item_tipo: str,
    expected: str,
    common_errors: list[str],
    enunciado_src: str,
    pregunta_src: str,
    solucion_src: str,
    entrega_src: str,
    entrega_text_outputs: str,
    codigo_ctx_solucion: str = "",
    codigo_ctx_entrega: str = "",
    codigo_ctx_outputs: str = "",
) -> str:
    """Ensambla el prompt que recibe la IA.

    `item_tipo` es "código" o "análisis" (lo que manda la rúbrica).
    `pregunta_src` solo se incluye si el ítem es de análisis.
    `entrega_text_outputs` son los stdout/errors de la celda del alumno
    concatenados (sin imágenes — esas las mira el docente).

    Para ítems de análisis, `codigo_ctx_*` (opcionales) llevan el código
    del mismo ejercicio como contexto: muchas preguntas de análisis se
    refieren a decisiones del código o a lo que el alumno observó al
    ejecutarlo. Sin este contexto, Claude evalúa la respuesta "a ciegas".
    """
    errores_block = "\n".join(f"- {e}" for e in common_errors) or "(sin lista)"

    # Código y análisis son dos actividades distintas: el ítem de código se evalúa
    # contra el enunciado del ejercicio; el de análisis se evalúa contra su
    # pregunta (que es autocontenida).
    if item_tipo == "análisis":
        enunciado_block = pregunta_src.strip() or enunciado_src.strip()
    else:
        enunciado_block = enunciado_src.strip()

    tipo_label = "código" if item_tipo == "código" else "respuesta de análisis"

    sections: list[tuple[str, str]] = [
        ("ENUNCIADO", enunciado_block),
        ("QUÉ SE ESPERA", expected.strip()),
        ("ERRORES FRECUENTES A TENER EN CUENTA", errores_block),
        ("SOLUCIÓN OFICIAL", solucion_src.strip() or "(no definida)"),
        ("ENTREGA DEL GRUPO", entrega_src.strip() or "(vacía)"),
    ]
    # Solo tiene sentido hablar de outputs para ítems de código. Para análisis,
    # la "salida" es el texto de la respuesta, que ya va en ENTREGA DEL GRUPO.
    if item_tipo == "código":
        outs = entrega_text_outputs.strip()
        if outs:
            sections.append(("OUTPUTS DE TEXTO DE LA ENTREGA", outs))
        else:
            sections.append((
                "OUTPUTS DE TEXTO DE LA ENTREGA",
                "(la celda no produjo outputs — no hay prints, ni errores, ni resultado)",
            ))

    # Contexto de código para preguntas de análisis: solo si el ítem es análisis
    # y existe una celda de código asociada al mismo ejercicio.
    if item_tipo == "análisis":
        if codigo_ctx_solucion.strip():
            sections.append((
                "CÓDIGO DEL EJERCICIO — SOLUCIÓN OFICIAL (contexto)",
                codigo_ctx_solucion.strip(),
            ))
        if codigo_ctx_entrega.strip():
            sections.append((
                "CÓDIGO DEL EJERCICIO — ENTREGA DEL GRUPO (contexto)",
                codigo_ctx_entrega.strip(),
            ))
        if codigo_ctx_outputs.strip():
            sections.append((
                "OUTPUTS DE TEXTO DEL CÓDIGO DEL GRUPO (contexto)",
                codigo_ctx_outputs.strip(),
            ))

    body = "\n\n".join(f"{title}\n{'=' * len(title)}\n{content}" for title, content in sections)

    header = f"Ejercicio: {ej_titulo}\nCorrigiendo: {tipo_label}"
    if item_tipo == "código":
        scope = (
            "Estás corrigiendo la CELDA DE CÓDIGO del grupo. Si QUÉ SE "
            "ESPERA describe pasos concretos (construir datos, aplicar un "
            "kernel, imprimir, graficar, comparar con un valor de "
            "referencia) y esos pasos no aparecen en la entrega ni en "
            "sus outputs, señalalo — aunque la parte que sí escribieron "
            "esté bien. Una celda sin outputs suele indicar que no la "
            "ejecutaron o que saltearon pasos pedidos."
        )
    else:
        scope = (
            "Estás corrigiendo SOLO la respuesta de análisis (prosa). El "
            "código que ves aparece como contexto para entender la "
            "pregunta; NO señales errores del código ni pasos faltantes "
            "del código salvo que la pregunta pida discutir algo del "
            "código. Evaluá si la respuesta cubre lo que pide la "
            "pregunta y si lo que afirma es correcto."
        )
    footer = (
        f"{scope} Redactá el borrador de observación. Si no hay nada que "
        "señalar, respondé solo: OK"
    )
    return f"{header}\n\n{body}\n\n{footer}"


async def _run_query(prompt: str) -> str:
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
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
    return "".join(parts).strip()


def generate_draft(prompt: str) -> str:
    """Ejecuta la query de forma síncrona. Lanza `ClaudeSDKError` ante fallos."""
    try:
        return asyncio.run(_run_query(prompt))
    except ClaudeSDKError:
        raise
    except Exception as e:
        raise ClaudeSDKError(f"Fallo inesperado al llamar al SDK: {e}") from e


# ─── Batch (todos los grupos para un mismo ítem en una sola llamada) ─────────

class GroupEntrega(TypedDict, total=False):
    """Datos de un grupo para un ítem dado, listos para meter en el batch.

    `grupo_id` se usa como clave en el JSON de salida. El resto depende del
    `item_tipo`: para "código" se usa `entrega_src` + `entrega_text_outputs`;
    para "análisis", la respuesta del alumno va en `entrega_src` y el código
    del mismo ejercicio del grupo (contexto) va en `codigo_ctx_entrega` +
    `codigo_ctx_outputs`.
    """
    grupo_id: str
    entrega_src: str
    entrega_text_outputs: str
    codigo_ctx_entrega: str
    codigo_ctx_outputs: str


def build_batch_prompt(
    *,
    ej_titulo: str,
    item_tipo: str,
    expected: str,
    common_errors: list[str],
    enunciado_src: str,
    pregunta_src: str,
    solucion_src: str,
    codigo_ctx_solucion: str = "",
    grupos: list[GroupEntrega],
) -> str:
    """Arma un prompt que evalúa N grupos contra la misma rúbrica.

    El contexto del ejercicio (enunciado, qué se espera, errores frecuentes,
    solución oficial) se manda UNA SOLA VEZ. Después siguen las entregas
    etiquetadas por `grupo_id`, y se pide salida JSON con una clave por
    grupo. La respuesta esperada para grupos correctos es exactamente "OK".
    """
    errores_block = "\n".join(f"- {e}" for e in common_errors) or "(sin lista)"

    if item_tipo == "análisis":
        enunciado_block = pregunta_src.strip() or enunciado_src.strip()
    else:
        enunciado_block = enunciado_src.strip()

    tipo_label = "código" if item_tipo == "código" else "respuesta de análisis"

    shared_sections: list[tuple[str, str]] = [
        ("ENUNCIADO", enunciado_block),
        ("QUÉ SE ESPERA", expected.strip()),
        ("ERRORES FRECUENTES A TENER EN CUENTA", errores_block),
        ("SOLUCIÓN OFICIAL", solucion_src.strip() or "(no definida)"),
    ]
    if item_tipo == "análisis" and codigo_ctx_solucion.strip():
        shared_sections.append((
            "CÓDIGO DEL EJERCICIO — SOLUCIÓN OFICIAL (contexto)",
            codigo_ctx_solucion.strip(),
        ))

    shared_block = "\n\n".join(
        f"{title}\n{'=' * len(title)}\n{content}" for title, content in shared_sections
    )

    # Bloque por grupo. Para cada uno repetimos solo lo que es propio del
    # grupo: la entrega y, según el tipo, sus outputs o el código de contexto.
    grupo_blocks: list[str] = []
    for g in grupos:
        gid = g["grupo_id"]
        parts = [f"--- {gid} ---"]
        entrega_src = (g.get("entrega_src") or "").strip()
        parts.append("ENTREGA:")
        parts.append(entrega_src or "(vacía)")
        if item_tipo == "código":
            outs = (g.get("entrega_text_outputs") or "").strip()
            parts.append("OUTPUTS DE TEXTO:")
            parts.append(
                outs if outs
                else "(la celda no produjo outputs — no hay prints, ni errores, ni resultado)"
            )
        else:
            ctx_e = (g.get("codigo_ctx_entrega") or "").strip()
            if ctx_e:
                parts.append("CÓDIGO DEL EJERCICIO — ENTREGA DEL GRUPO (contexto):")
                parts.append(ctx_e)
            ctx_o = (g.get("codigo_ctx_outputs") or "").strip()
            if ctx_o:
                parts.append("OUTPUTS DE TEXTO DEL CÓDIGO DEL GRUPO (contexto):")
                parts.append(ctx_o)
        grupo_blocks.append("\n".join(parts))

    grupos_block = "\n\n".join(grupo_blocks)
    grupo_ids = [g["grupo_id"] for g in grupos]
    ejemplo_keys = ", ".join(f'"{gid}": "..."' for gid in grupo_ids[:2]) or '"grupo_01": "..."'

    if item_tipo == "código":
        scope = (
            "Estás corrigiendo la CELDA DE CÓDIGO de cada grupo. Si QUÉ "
            "SE ESPERA describe pasos concretos (construir datos, aplicar "
            "un kernel, imprimir, graficar, comparar con un valor de "
            "referencia) y esos pasos no aparecen en la entrega ni en sus "
            "outputs, señalalo — aunque la parte que sí escribieron esté "
            "bien. Una celda sin outputs suele indicar que no la "
            "ejecutaron o que saltearon pasos pedidos."
        )
    else:
        scope = (
            "Estás corrigiendo SOLO la respuesta de análisis (prosa) de "
            "cada grupo. El código que ves aparece como contexto para "
            "entender la pregunta; NO señales errores del código ni pasos "
            "faltantes del código salvo que la pregunta pida discutir "
            "algo del código. Evaluá si la respuesta cubre lo que pide la "
            "pregunta y si lo que afirma es correcto."
        )

    independencia = (
        "EVALUÁ CADA ENTREGA DE FORMA INDEPENDIENTE contra la rúbrica. "
        "No compares las entregas entre sí: aunque varios grupos cometan "
        "el mismo error, señalalo en cada uno con el mismo nivel de "
        "detalle. El que muchos grupos estén bien no te baja el listón, "
        "y que muchos estén mal no te lo sube. La rúbrica es la única "
        "vara."
    )

    output_spec = (
        "Devolveme un único objeto JSON con una clave por grupo y como "
        "valor el borrador de observación (o 'OK' si la entrega cumple "
        "la rúbrica). Forma exacta:\n"
        f"{{ {ejemplo_keys}, ... }}\n\n"
        "Reglas de salida:\n"
        "- Si la entrega del grupo cumple lo pedido y no hay nada que "
        "señalar, el valor debe ser exactamente la cadena \"OK\" (sin "
        "punto, sin nada más).\n"
        "- Si hay algo que señalar, redactá el borrador siguiendo las "
        "reglas de estilo del system prompt (1 a 4 oraciones, dirigido "
        "al grupo en plural con \"ustedes\", solo señalando errores sin "
        "dar instrucciones de cómo arreglarlos).\n"
        "- Devolvé SOLO el JSON. Sin texto antes ni después, sin "
        "markdown fences, sin comentarios.\n"
        f"- Las claves del JSON deben ser exactamente: {', '.join(grupo_ids)}."
    )

    header = (
        f"Ejercicio: {ej_titulo}\nCorrigiendo: {tipo_label} de "
        f"{len(grupos)} grupos en una sola pasada.\n\n{independencia}"
    )
    footer = f"{scope}\n\n{output_spec}"

    return (
        f"{header}\n\n{shared_block}\n\nENTREGAS DE LOS GRUPOS\n"
        f"======================\n\n{grupos_block}\n\n{footer}"
    )


def _extract_json_object(raw: str) -> dict:
    """Extrae un dict JSON de una respuesta del modelo.

    Tolera markdown fences (```json ... ```), texto antes o después, y
    espacios. Lanza `ValueError` si no encuentra JSON parseable.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        # Recortamos al primer `{` y al último `}` para tolerar prosa.
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise ValueError("No encontré un objeto JSON en la respuesta.")
        text = text[first : last + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta no es un objeto JSON.")
    return parsed


def generate_batch_drafts(prompt: str, expected_keys: list[str]) -> dict[str, str]:
    """Ejecuta el batch y devuelve `{grupo_id: texto}`.

    `expected_keys` se usa para validar la salida y para tolerar el caso
    en que el modelo se saltee algún grupo: las claves faltantes quedan
    fuera del dict (el caller decide si reintenta esos grupos sueltos).
    Lanza `ClaudeSDKError` si el SDK falla o `ValueError` si la salida
    no es un JSON parseable.
    """
    raw = generate_draft(prompt)
    parsed = _extract_json_object(raw)
    out: dict[str, str] = {}
    for k in expected_keys:
        v = parsed.get(k)
        if isinstance(v, str):
            out[k] = v.strip()
    return out
