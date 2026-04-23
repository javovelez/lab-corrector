"""
Generación de borradores de feedback vía `claude-agent-sdk`.

Usa la sesión local de Claude Code del usuario (sin API key). Ejecuta una
query one-shot, sin tools y sin cargar settings del proyecto — solo
prompt → texto.
"""
from __future__ import annotations

import asyncio
from textwrap import dedent

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
