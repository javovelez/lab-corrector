"""
App de corrección — agnóstica al lab/asignatura.

La app opera sobre un *workdir*: una carpeta elegida por el docente en
la que vive una tanda de corrección (zip de Moodle, entregas extraídas
como `grupo_NN/`, feedback por grupo y `grupo_NN.txt` finales).

Estado interno: `<workdir>/.corrector/config.json` apunta a los notebooks
de enunciado/solución y a la rúbrica (cualquier path absoluto). Los
workdirs recientes se persisten en `~/.lab_corrector/recent.json` para
retomar correcciones sin tener que re-apuntar todo.

Ejecutar desde la raíz del proyecto (o cualquier directorio):
    app/.venv/bin/streamlit run app/main.py
"""
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import streamlit as st
import yaml
from claude_agent_sdk import ClaudeSDKError


def open_in_os(path: Path) -> bool:
    """Abre un archivo con la app por defecto del SO. Devuelve True si se lanzó bien.

    Funciona porque Streamlit corre local: `subprocess.run` se ejecuta en la
    misma máquina donde el docente está mirando el browser.
    """
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=True)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=True)
        elif sys.platform == "win32":
            import os as _os
            _os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            return False
        return True
    except Exception:
        return False


def _clean_path(raw: str) -> str:
    """Normaliza un path pegado: quita comillas externas y espacios."""
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _osascript_choose(kind: str, initial: str | None = None) -> str | None:
    """Abre el Finder vía osascript. `kind` = 'folder' | 'file'.

    Devuelve POSIX path elegido o None si se canceló.
    """
    base = "POSIX path of (choose folder" if kind == "folder" else "POSIX path of (choose file"
    if initial:
        safe = initial.replace('"', '\\"')
        base += f' default location POSIX file "{safe}"'
    script = base + ")"
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    if out and out != "/":
        out = out.rstrip("/")
    return out or None


def _initial_dir(current: str, kind: str) -> str | None:
    """Carpeta donde abrir el diálogo según el path actual."""
    if not current:
        return None
    p = Path(current).expanduser()
    if kind == "folder" and p.is_dir():
        return str(p)
    if p.parent.is_dir():
        return str(p.parent)
    return None


def _choose_path_button(label: str, key: str, kind: str) -> str:
    """Botón que abre el diálogo nativo + caption con el path elegido.

    Guarda el path en `st.session_state[key]` y lo devuelve.
    """
    if st.button(label, key=f"{key}-pick", use_container_width=True):
        cur = st.session_state.get(key, "")
        chosen = _osascript_choose(kind, _initial_dir(cur, kind))
        if chosen:
            st.session_state[key] = chosen
            st.rerun()
    cur = st.session_state.get(key, "")
    st.caption(f"`{cur}`" if cur else "_(sin elegir)_")
    return cur

import recents
from ai import (
    GroupEntrega,
    build_batch_prompt,
    build_prompt,
    generate_batch_drafts,
    generate_draft,
)
from export import build_grupo_txt, compute_grupo_score, count_observaciones
from grupos import list_grupos, notebook_path
from intake import intake_zip
from nbparse import cell_outputs, cell_source, find_cell, load_notebook
from rubric import load_rubrica, save_rubrica
from rubric_gen import generate_rubrica
from state import (
    LEVEL_BIEN,
    LEVEL_MAL,
    LEVEL_REGULAR,
    STATUS_OBS,
    STATUS_OK,
    STATUS_PENDIENTE,
    clear_cell_override,
    clear_feedback,
    clear_draft,
    draft_path,
    feedback_path,
    read_cell_overrides,
    read_draft,
    read_feedback,
    resolved_id,
    save_draft,
    save_observation,
    save_sin_observaciones,
    set_cell_override,
)
from workdir import (
    WorkdirConfig,
    config_path,
    has_config,
    load_config,
    save_config,
    validate_config,
)


COLOR_PENDIENTE = "#E0E0E0"  # gris — todavía no corregido
COLOR_OK        = "#A5D6A7"  # verde — sin observaciones o "con observación bien"
COLOR_REGULAR   = "#FFE082"  # amarillo — "con observación regular"
COLOR_MAL       = "#EF9A9A"  # rojo — "con observación mal" o entrega faltante


st.set_page_config(page_title="Corrector de notebooks", layout="wide")


# Streamlit no respeta `secondaryBackgroundColor` del theme para los botones
# secondary/tertiary (en 1.56 quedan blancos). Forzamos por CSS un tinte
# crema que combina con el resto de la app. Los botones con CSS específico
# (matriz IA con drafts, guardar bien/regular/mal, "usar esta para X") usan
# selectores `.st-key-*` más específicos y siguen pisando esto.
st.markdown(
    """
    <style>
    [data-testid="stButton"] button[kind="secondary"],
    [data-testid="stDownloadButton"] button[kind="secondary"],
    [data-testid="stFormSubmitButton"] button[kind="secondary"] {
        background-color: #EFEAE0 !important;
        border: 1px solid #DDD6C5 !important;
        color: #2B2B2B !important;
    }
    [data-testid="stButton"] button[kind="secondary"]:hover,
    [data-testid="stDownloadButton"] button[kind="secondary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="secondary"]:hover {
        background-color: #E5DFD0 !important;
        border-color: #C5BCA8 !important;
    }
    [data-testid="stButton"] button[kind="tertiary"] {
        background-color: #F2EFE7 !important;
        color: #2B2B2B !important;
    }
    [data-testid="stButton"] button[kind="tertiary"]:hover {
        background-color: #E8E3D5 !important;
    }
    [data-testid="stButton"] button:disabled {
        background-color: #F0EDE5 !important;
        color: #999 !important;
        border-color: #E0DBCE !important;
    }
    /* Textarea deshabilitada (borrador IA): texto en negro y negrita
       para que sea legible aunque el campo esté inactivo. */
    [data-testid="stTextArea"] textarea:disabled {
        color: #000 !important;
        -webkit-text-fill-color: #000 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helpers visuales ────────────────────────────────────────────────────────

def cell_div(color: str, label: str = "·") -> str:
    return (
        f"<div style='background:{color};height:28px;border-radius:4px;"
        f"display:flex;align-items:center;justify-content:center;"
        f"font-size:11px;color:#333'>{label}</div>"
    )


def cell_link(color: str, href: str, label: str = "abrir") -> str:
    return (
        f'<a href="{href}" target="_self" '
        f'style="display:block;background:{color};height:28px;'
        f'border-radius:4px;text-decoration:none;text-align:center;'
        f'line-height:28px;color:#222;font-size:12px;font-weight:500">{label}</a>'
    )


def build_items(rubrica: dict) -> list[dict]:
    """Expande cada ejercicio en sus ítems corregibles (schema v2).

    Cada ítem del YAML (kind=code o kind=analysis) produce una entrada en la lista.
    Los consumers leen `cell_id`, `code_cell`, `answer_cell`, `pregunta_cell`,
    `rubric` y `companion_code_cells` directamente del ítem — nunca del ejercicio.
    """
    result: list[dict] = []
    for ej in rubrica["ejercicios"]:
        items_ej = ej.get("items", [])
        # Códigos del mismo ejercicio: se pasan como contexto a los ítems de
        # análisis del mismo ejercicio (la pregunta suele depender del código).
        companion_codes = [
            it["code_cell"] for it in items_ej
            if it.get("kind") == "code" and it.get("code_cell")
        ]
        for it in items_ej:
            kind = it.get("kind")
            if kind == "code":
                tipo = "código"
                cell_id = it.get("code_cell")
            elif kind == "analysis":
                tipo = "análisis"
                cell_id = it.get("answer_cell")
            else:
                continue
            if cell_id is None:
                continue
            result.append({
                "key":     it.get("key") or f"{ej['id']}-{kind}",
                "ej_id":   ej["id"],
                "tipo":    tipo,
                "titulo":  ej.get("titulo", ""),
                "cell_id": cell_id,
                "code_cell":     it.get("code_cell"),
                "answer_cell":   it.get("answer_cell"),
                "pregunta_cell": it.get("pregunta_cell"),
                "rubric":        it.get("rubric", {}) or {},
                "companion_code_cells": companion_codes if kind == "analysis" else [],
            })
    return result


def text_outputs_concat(outputs: list[dict]) -> str:
    """Concatena stdout y tracebacks de los outputs de una celda — sin imágenes."""
    parts: list[str] = []
    for out in outputs:
        if out["kind"] in ("text", "error"):
            parts.append(out["text"])
    return "\n".join(parts)


def render_student_outputs(outputs: list[dict]) -> None:
    if not outputs:
        st.caption("_Sin outputs guardados._")
        return
    for out in outputs:
        if out["kind"] == "text":
            st.code(out["text"], language=None)
        elif out["kind"] == "image":
            st.image(base64.b64decode(out["data_b64"]))
        elif out["kind"] == "error":
            st.error(out["text"])


# ─── Landing: elegir o crear workdir ─────────────────────────────────────────

def landing() -> None:
    st.title("Corrector de notebooks")
    st.caption(
        "Elegí la carpeta (workdir) donde vive el zip de Moodle de una tanda "
        "de corrección. La app trabaja adentro de esa carpeta y retoma donde "
        "dejaste la próxima vez que la abras."
    )

    st.markdown("### Recientes")
    rec = recents.list_recents()
    if not rec:
        st.caption("_No hay workdirs recientes todavía._")
    else:
        for wd in rec:
            col_btn, col_forget = st.columns([5, 1])
            label = f"{wd.name}  ·  `{wd}`"
            if col_btn.button(label, key=f"open-{wd}", use_container_width=True):
                _enter_workdir(wd)
            if col_forget.button("quitar", key=f"forget-{wd}", help="Sacar de recientes"):
                recents.forget(wd)
                st.rerun()

    st.markdown("### Abrir otra carpeta")
    if st.button("Elegir carpeta del workdir…", key="landing-open-pick", use_container_width=True):
        chosen = _osascript_choose("folder")
        if chosen:
            p = Path(chosen).expanduser()
            if p.is_dir():
                _enter_workdir(p)
            else:
                st.error("Esa carpeta no existe.")

    st.markdown("### Nueva corrección")
    st.markdown("**Carpeta del workdir**")
    wd_str = _choose_path_button("Elegir carpeta…", "new-wd", "folder")
    title = st.text_input("Título del lab", key="new-title", placeholder="Laboratorio 3 — Transferencia")
    st.markdown("**Notebook de enunciado (.ipynb)**")
    nb_enun = _choose_path_button("Elegir notebook de enunciado…", "new-nb-enun", "file")
    st.markdown("**Notebook de solución (.ipynb)**")
    nb_sol = _choose_path_button("Elegir notebook de solución…", "new-nb-sol", "file")
    st.markdown("**Zip de Moodle** (opcional — podés importarlo después)")
    zip_path = _choose_path_button("Elegir zip…", "new-zip", "file")
    rubric_mode = st.radio(
        "Rúbrica",
        ["Generar automáticamente desde la solución (usa Claude)", "Ya tengo una (.rubric.yaml)"],
        index=0,
        key="new-rubric-mode",
    )
    rubric_existing = ""
    if rubric_mode.startswith("Ya tengo"):
        st.markdown("**Rúbrica existente (.rubric.yaml)**")
        rubric_existing = _choose_path_button("Elegir rúbrica…", "new-rubric-path", "file")

    if st.button("Crear", type="primary", use_container_width=True, key="new-create-btn"):
        _create_new_workdir(
            wd_str=wd_str,
            title=title,
            nb_enun_str=nb_enun,
            nb_sol_str=nb_sol,
            zip_str=zip_path,
            auto_rubric=(rubric_mode.startswith("Generar")),
            rubric_existing_str=rubric_existing,
        )


def _enter_workdir(wd: Path) -> None:
    if not has_config(wd):
        st.error(
            f"`{wd}` no tiene `.corrector/config.json`. "
            "Usá **Nueva corrección** para inicializarlo."
        )
        return
    st.session_state["workdir"] = str(wd.resolve())
    recents.touch(wd)
    st.rerun()


def _create_new_workdir(
    *,
    wd_str: str,
    title: str,
    nb_enun_str: str,
    nb_sol_str: str,
    zip_str: str,
    auto_rubric: bool,
    rubric_existing_str: str,
) -> None:
    wd_str = _clean_path(wd_str)
    if not wd_str:
        st.error("Indicá el path al workdir.")
        return
    wd_raw = Path(wd_str).expanduser()
    if not wd_raw.is_absolute():
        st.error(f"El path del workdir debe ser absoluto (empezar con `/`). Recibí: `{wd_str}`")
        return
    wd = wd_raw.resolve()
    wd.mkdir(parents=True, exist_ok=True)

    nb_enun = Path(_clean_path(nb_enun_str)).expanduser()
    nb_sol  = Path(_clean_path(nb_sol_str)).expanduser()
    if not nb_enun.is_absolute() or not nb_enun.exists():
        st.error(f"`notebook de enunciado` inválido: {nb_enun}")
        return
    if not nb_sol.is_absolute() or not nb_sol.exists():
        st.error(f"`notebook de solución` inválido: {nb_sol}")
        return

    # Rúbrica: existente o autogen.
    if auto_rubric:
        rubrica_path = wd / ".corrector" / "rubrica.yaml"
        rubrica_path.parent.mkdir(parents=True, exist_ok=True)
        progress_bar = st.progress(0.0, text="Generando rúbrica con Claude…")
        def prog(i, total, msg):
            progress_bar.progress(i / total, text=f"({i}/{total}) {msg}")
        try:
            rubrica = generate_rubrica(
                title=title or wd.name,
                enunciado_nb_path=nb_enun,
                solucion_nb_path=nb_sol,
                progress=prog,
            )
        except Exception as e:
            st.error(f"No pude generar la rúbrica: {e}")
            return
        save_rubrica(rubrica_path, rubrica)
    else:
        rub_str = _clean_path(rubric_existing_str)
        if not rub_str:
            st.error("Indicá el path a la rúbrica existente.")
            return
        rubrica_path = Path(rub_str).expanduser().resolve()
        if not rubrica_path.exists():
            st.error(f"La rúbrica no existe: {rubrica_path}")
            return

    cfg = WorkdirConfig(
        title=title or wd.name,
        notebook_enunciado=str(nb_enun.resolve()),
        notebook_solucion=str(nb_sol.resolve()),
        rubrica=str(rubrica_path.resolve()),
    )
    save_config(wd, cfg)

    # Intake opcional del zip.
    zp = _clean_path(zip_str)
    if zp:
        zip_p = Path(zp).expanduser()
        with st.spinner("Importando zip de Moodle…"):
            report = intake_zip(zip_p, wd)
        st.session_state["intake-last-report"] = {
            "imported": report.imported,
            "skipped":  report.skipped,
            "warnings": report.warnings,
        }

    recents.touch(wd)
    st.session_state["workdir"] = str(wd)
    st.success(f"Workdir listo en `{wd}`. Abriendo…")
    st.rerun()


# ─── Sidebar: workdir activo + acciones ──────────────────────────────────────

def sidebar(wd: Path, cfg: WorkdirConfig) -> None:
    st.sidebar.markdown(f"**{cfg.title}**")
    st.sidebar.caption(f"`{wd}`")
    if st.sidebar.button("Cambiar workdir", use_container_width=True):
        st.session_state.pop("workdir", None)
        st.query_params.clear()
        st.rerun()

    with st.sidebar.expander("Importar más entregas (Moodle)", expanded=False):
        st.caption(
            "El zip se descomprime en un temporal del sistema; los `.ipynb` "
            "se copian a `<workdir>/grupo_NN/entrega.ipynb`. El feedback "
            "existente de cada grupo se preserva."
        )
        st.markdown("**Zip de Moodle**")
        zip_str = _choose_path_button("Elegir zip…", "intake-zip", "file")
        if st.button("Importar", use_container_width=True, key="btn-intake"):
            clean = _clean_path(zip_str)
            zp = Path(clean).expanduser() if clean else None
            if zp is None:
                st.error("Elegí primero el zip.")
            else:
                with st.spinner("Importando…"):
                    report = intake_zip(zp, wd)
                st.session_state["intake-last-report"] = {
                    "imported": report.imported,
                    "skipped":  report.skipped,
                    "warnings": report.warnings,
                }
                st.rerun()

        rep = st.session_state.get("intake-last-report")
        if rep:
            if rep["imported"]:
                st.success(f"Importados: {len(rep['imported'])} — " + ", ".join(rep["imported"]))
            if rep["warnings"]:
                for grupo, msg in rep["warnings"]:
                    st.warning(f"{grupo}: {msg}")
            if rep["skipped"]:
                for nombre, motivo in rep["skipped"]:
                    st.info(f"Omitido `{nombre}`: {motivo}")

    with st.sidebar.expander("Config", expanded=False):
        st.caption(f"`{config_path(wd)}`")
        st.code(
            f"title: {cfg.title}\n"
            f"enunciado: {cfg.notebook_enunciado}\n"
            f"solución:  {cfg.notebook_solucion}\n"
            f"rúbrica:   {cfg.rubrica}",
            language="text",
        )


# ─── Helpers IA: armado de payload por ítem y por grupo ─────────────────────

def _shared_ai_context(
    *, item: dict, ej: dict, enunciado_nb, solucion_nb
) -> dict:
    """Datos del ítem que NO dependen del grupo: enunciado, solución, contexto.

    Se llama una vez por ítem (sea para el batch o para una llamada
    individual) y el resultado se mete tal cual en `build_prompt` o
    `build_batch_prompt`.
    """
    target_cell_id = item["code_cell"] if item["tipo"] == "código" else item["answer_cell"]
    solucion_src = cell_source(find_cell(solucion_nb, target_cell_id))
    pregunta_src = (
        cell_source(find_cell(enunciado_nb, item.get("pregunta_cell") or ""))
        if item["tipo"] == "análisis" else ""
    )
    enunciado_src = cell_source(find_cell(enunciado_nb, ej["enunciado_cell"]))

    codigo_ctx_solucion = ""
    if item["tipo"] == "análisis":
        companion = item.get("companion_code_cells", []) or []
        if companion:
            sol_parts: list[str] = []
            for code_id in companion:
                sol_src = cell_source(find_cell(solucion_nb, code_id))
                if sol_src.strip():
                    sol_parts.append(f"# celda {code_id}\n{sol_src}")
            codigo_ctx_solucion = "\n\n".join(sol_parts)

    return {
        "enunciado_src":       enunciado_src,
        "pregunta_src":        pregunta_src,
        "solucion_src":        solucion_src,
        "codigo_ctx_solucion": codigo_ctx_solucion,
    }


def _grupo_ai_payload(
    *, item: dict, entrega_nb, overrides: dict[str, str] | None = None
) -> GroupEntrega:
    """Datos del ítem ESPECÍFICOS al grupo (entrega + contexto del grupo).

    `overrides` se aplica a TODOS los lookups de celdas del entregable,
    tanto la celda principal del ítem como las companion_code_cells (que
    sirven de contexto en ítems de análisis). Así el prompt de la IA
    refleja exactamente lo que el corrector eligió en el navegador.
    """
    ov = overrides or {}
    target_cell_id = item["code_cell"] if item["tipo"] == "código" else item["answer_cell"]
    entrega_cell = find_cell(entrega_nb, resolved_id(target_cell_id, ov))
    entrega_src = cell_source(entrega_cell)

    if item["tipo"] == "código":
        entrega_outs = text_outputs_concat(cell_outputs(entrega_cell))
        return {
            "grupo_id": "",  # lo setea el caller
            "entrega_src": entrega_src,
            "entrega_text_outputs": entrega_outs,
            "codigo_ctx_entrega": "",
            "codigo_ctx_outputs": "",
        }

    # análisis: agregamos el código del mismo ejercicio del grupo como contexto
    ent_parts: list[str] = []
    out_parts: list[str] = []
    for code_id in item.get("companion_code_cells", []) or []:
        label = f"# celda {code_id}"
        ent_cell = find_cell(entrega_nb, resolved_id(code_id, ov))
        ent_src = cell_source(ent_cell)
        if ent_src.strip():
            ent_parts.append(f"{label}\n{ent_src}")
        out_text = text_outputs_concat(cell_outputs(ent_cell))
        if out_text.strip():
            out_parts.append(f"{label}\n{out_text}")
    return {
        "grupo_id": "",
        "entrega_src": entrega_src,
        "entrega_text_outputs": "",
        "codigo_ctx_entrega": "\n\n".join(ent_parts),
        "codigo_ctx_outputs": "\n\n".join(out_parts),
    }


# ─── Panel "Entrega" + navegador de celdas ──────────────────────────────────

def _render_entrega_panel(
    *, wd: Path, grupo: str, item: dict, item_key: str, entrega_nb, enunciado_nb
) -> None:
    """Renderiza la celda del entregable que corresponde al ítem.

    Siempre se muestra un mini-navegador (↑/↓) sobre la celda. Por
    defecto la celda mostrada es:
      1) la que el corrector eligió con ↑/↓ en esta sesión, o
      2) la del override persistido (`cell_overrides.json`), o
      3) la celda con el id esperado por la rúbrica.

    Si la celda mostrada es distinta de la esperada aparece el botón
    "Usar esta para `<id>`", que persiste el mapeo. Si ya existe un
    override, se muestra un banner con un botón para quitarlo. El
    notebook del grupo nunca se modifica.

    Cuando la celda esperada no existe en el entregable, ↑/↓ usan
    como ancla la posición canónica de esa celda en el enunciado
    oficial: ↓ va al primer candidato a/después de esa posición, ↑
    al último antes. Así el corrector aterriza cerca de donde la
    celda *debería* estar, no en los extremos del notebook.
    """
    expected_id = item["code_cell"] if item["tipo"] == "código" else item["answer_cell"]
    expected_kind = "code" if item["tipo"] == "código" else "markdown"

    overrides = read_cell_overrides(wd, grupo)
    nav_key = f"nav-cell-{grupo}-{item_key}"

    if nav_key in st.session_state:
        display_id = st.session_state[nav_key]
    elif expected_id in overrides:
        display_id = overrides[expected_id]
    else:
        display_id = expected_id

    candidates = [c for c in entrega_nb.get("cells", []) if c.get("cell_type") == expected_kind]
    cand_ids = [c.get("id") for c in candidates]
    try:
        cur_idx = cand_ids.index(display_id)
    except ValueError:
        cur_idx = None

    # Para el caso "celda esperada faltante": calcular el split del listado
    # de candidatos en "antes de la posición canónica" / "a partir de ella".
    # Lo hacemos comparando posiciones en `enunciado_nb`, que es el orden
    # canónico. Si una celda del entregable no aparece en el enunciado
    # (id random), va a quedar entre sus vecinos por orden de aparición
    # en el entregable, lo cual es razonable.
    canonical_ids = [c.get("id") for c in enunciado_nb.get("cells", [])]
    try:
        canon_anchor = canonical_ids.index(expected_id)
    except ValueError:
        canon_anchor = None

    def canonical_pos(cell_id: str) -> int | None:
        try:
            return canonical_ids.index(cell_id)
        except ValueError:
            return None

    # `split_idx`: cantidad de candidatos cuya posición canónica < anchor.
    # Si una celda candidata no está en el enunciado, heredamos la última
    # posición canónica vista para que las celdas con id random caigan
    # junto al bloque del que son vecinas.
    split_idx = 0
    if canon_anchor is not None:
        last_seen = -1
        for cand in candidates:
            cpos = canonical_pos(cand.get("id"))
            if cpos is not None:
                last_seen = cpos
            if last_seen < canon_anchor:
                split_idx += 1
            else:
                break

    pos_label = (
        f"{cur_idx + 1}/{len(candidates)}" if cur_idx is not None
        else f"—/{len(candidates)}"
    )
    nav_cols = st.columns([5, 1, 1, 4])
    nav_cols[0].markdown(
        f"<div style='padding-top:6px;font-size:13px'>"
        f"Mostrando: <code>{display_id}</code> &nbsp; "
        f"<span style='color:#888'>{pos_label}</span></div>",
        unsafe_allow_html=True,
    )

    no_candidates = len(candidates) == 0
    up_disabled = no_candidates or (cur_idx is not None and cur_idx == 0)
    down_disabled = no_candidates or (cur_idx is not None and cur_idx == len(candidates) - 1)

    if nav_cols[1].button(
        "↑", key=f"nav-up-{nav_key}", disabled=up_disabled, use_container_width=True,
        help="Celda anterior del mismo tipo en el notebook",
    ):
        if cur_idx is not None:
            new_idx = cur_idx - 1
        elif split_idx > 0:
            new_idx = split_idx - 1
        else:
            new_idx = len(candidates) - 1
        st.session_state[nav_key] = cand_ids[new_idx]
        st.rerun()

    if nav_cols[2].button(
        "↓", key=f"nav-down-{nav_key}", disabled=down_disabled, use_container_width=True,
        help="Celda siguiente del mismo tipo en el notebook",
    ):
        if cur_idx is not None:
            new_idx = cur_idx + 1
        elif split_idx < len(candidates):
            new_idx = split_idx
        else:
            new_idx = 0
        st.session_state[nav_key] = cand_ids[new_idx]
        st.rerun()

    if display_id != expected_id and cur_idx is not None:
        if nav_cols[3].button(
            f"Usar esta para `{expected_id}`",
            key=f"nav-use-{nav_key}",
            type="primary",
            use_container_width=True,
            help=(
                "Marca esta celda como la respuesta del grupo para "
                f"`{expected_id}`. No modifica el notebook del grupo: "
                "guarda un mapeo lógico que la app respeta de acá en "
                "más, también en el prompt de la IA."
            ),
        ):
            set_cell_override(wd, grupo, expected_id, display_id)
            st.session_state.pop(nav_key, None)
            st.rerun()

    if expected_id in overrides:
        actual = overrides[expected_id]
        col_info, col_clear = st.columns([6, 2])
        col_info.info(
            f"Mapeo activo: usando `{actual}` como `{expected_id}`. "
            "El notebook del grupo no se modifica."
        )
        if col_clear.button(
            "Quitar mapeo", key=f"nav-clear-{nav_key}", use_container_width=True
        ):
            clear_cell_override(wd, grupo, expected_id)
            st.session_state.pop(nav_key, None)
            st.rerun()

    if cur_idx is None:
        st.warning(
            f"El grupo no tiene la celda `{display_id}` en su entrega. "
            "Usá las flechas para buscarla en otra parte del notebook; "
            "cuando la encuentres, marcá **Usar esta para `"
            f"{expected_id}`**."
        )
        return

    student_cell = candidates[cur_idx]
    if item["tipo"] == "código":
        st.code(cell_source(student_cell), language="python")
        st.markdown("**Outputs:**")
        render_student_outputs(cell_outputs(student_cell))
    else:
        source = cell_source(student_cell).strip()
        if not source or source == "*(Escribí tu respuesta acá)*":
            st.caption("_El alumno no respondió._")
        else:
            st.markdown(source)


# ─── Vista matriz ────────────────────────────────────────────────────────────

def view_matriz(
    wd: Path, rubrica: dict, cfg: WorkdirConfig, items: list[dict], grupos: list[str]
) -> None:
    st.title(rubrica.get("title") or "Corrector")

    col_meta1, col_meta2 = st.columns([1, 3])
    col_meta1.metric("Grupos", len(grupos))
    col_meta2.metric("Ítems corregibles", len(items))

    st.markdown("### Estado de corrección")
    pill = "padding:3px 10px;border-radius:3px;font-size:13px;color:#222;font-weight:500"
    st.markdown(
        f"<div style='margin-bottom:12px'>"
        f"<span style='background:{COLOR_PENDIENTE};{pill}'>pendiente</span> &nbsp; "
        f"<span style='background:{COLOR_OK};{pill}'>sin obs / bien (1pt)</span> &nbsp; "
        f"<span style='background:{COLOR_REGULAR};{pill}'>regular (½pt)</span> &nbsp; "
        f"<span style='background:{COLOR_MAL};{pill}'>mal / entrega faltante (0pt)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not grupos:
        st.warning(
            "No hay grupos en este workdir. Usá **Importar más entregas (Moodle)** "
            "en el sidebar para traer un zip."
        )
        return

    header_cols = st.columns([3] + [1] * len(grupos))
    header_cols[0].markdown("**Ítem a corregir**")
    for i, g in enumerate(grupos):
        nb = notebook_path(wd, g)
        label = g.replace("grupo_", "G")
        if nb is None:
            # Entrega faltante: 0pt por definición, mostramos directamente.
            header_cols[i + 1].markdown(
                f"<div style='text-align:center'><b>{label}</b><br>"
                f"<span style='color:#c00;font-size:10px'>sin ipynb</span><br>"
                f"<span style='background:{COLOR_MAL};padding:1px 6px;"
                f"border-radius:3px;font-size:11px'>0%</span></div>",
                unsafe_allow_html=True,
            )
            continue

        if header_cols[i + 1].button(
            label,
            key=f"open-nb-{g}",
            use_container_width=True,
            type="tertiary",
            help=f"Abrir {nb.name} con la app por defecto del SO",
        ):
            if not open_in_os(nb):
                st.toast(f"No pude abrir {nb}", icon="⚠️")

        # Score badge: porcentaje cuando todos los ítems están corregidos,
        # cantidad de pendientes en gris si todavía falta corregir alguno.
        puntos, total, pct, pendientes = compute_grupo_score(
            workdir=wd, items=items, grupo=g
        )
        if pct is None:
            badge = (
                f"<span style='background:{COLOR_PENDIENTE};padding:1px 6px;"
                f"border-radius:3px;font-size:11px;color:#444'>"
                f"{pendientes} pend</span>"
            )
        else:
            # Color del badge según rango: ≥75% verde, ≥50% amarillo, resto rojo.
            badge_bg = (
                COLOR_OK if pct >= 75
                else COLOR_REGULAR if pct >= 50
                else COLOR_MAL
            )
            badge = (
                f"<span style='background:{badge_bg};padding:1px 6px;"
                f"border-radius:3px;font-size:11px'>{pct:.0f}%</span>"
            )
        header_cols[i + 1].markdown(
            f"<div style='text-align:center;margin-top:-4px'>{badge}</div>",
            unsafe_allow_html=True,
        )

        n_obs = count_observaciones(workdir=wd, items=items, grupo=g)
        txt = build_grupo_txt(workdir=wd, items=items, grupo=g)
        txt_path = wd / g / f"{g}.txt"
        already = txt_path.exists()
        if header_cols[i + 1].button(
            label=f"txt ({n_obs}){' ✓' if already else ''}",
            use_container_width=True,
            key=f"dl-{g}",
            help=(
                f"Escribir {txt_path.relative_to(wd)} en el workdir "
                "(solo observaciones, listo para Moodle)."
            ),
            disabled=not txt,
        ):
            txt_path.write_text(txt, encoding="utf-8")
            st.toast(f"Escrito {txt_path}", icon="💾")
            st.rerun()

    # Notebooks para el batch IA. Se cargan una vez y se reusan por fila.
    enunciado_nb = load_notebook(Path(cfg.notebook_enunciado))
    solucion_nb  = load_notebook(Path(cfg.notebook_solucion))

    # Filas que ya tienen al menos un borrador IA generado: el botón "IA" se
    # pinta verde con CSS apuntando a la clase `.st-key-batch-ia-{key}` que
    # Streamlit (≥1.36) coloca en el contenedor del widget.
    rows_with_drafts = [
        item["key"] for item in items
        if any(
            draft_path(wd, g, item["key"]).exists()
            for g in grupos
        )
    ]
    if rows_with_drafts:
        rules = "\n".join(
            f".st-key-batch-ia-{k} button {{"
            f" background:{COLOR_OK} !important;"
            f" color:#222 !important;"
            f" border-color:#81C784 !important; }}"
            for k in rows_with_drafts
        )
        st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)

    prev_ej = None
    for item in items:
        if prev_ej is not None and item["ej_id"] != prev_ej:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        prev_ej = item["ej_id"]

        row_cols = st.columns([2.2, 0.8] + [1] * len(grupos))
        if item["tipo"] == "código":
            label = f"**{item['ej_id']}** · código &nbsp; — &nbsp; {item['titulo']}"
        else:
            label = (
                f"<span style='color:#666'>&nbsp;&nbsp;&nbsp;&nbsp;↳ "
                f"<b>{item['ej_id']}</b> · análisis</span>"
            )
        row_cols[0].markdown(label, unsafe_allow_html=True)

        # Botón "IA": genera borradores en batch para los grupos pendientes
        # de esta fila (no pisa devoluciones ya validadas). Se pinta verde
        # cuando ya existe al menos un borrador en la fila (ver CSS arriba).
        has_drafts = item["key"] in rows_with_drafts
        if row_cols[1].button(
            "IA ✓" if has_drafts else "IA",
            key=f"batch-ia-{item['key']}",
            use_container_width=True,
            help=(
                "Regenerar borradores IA para los grupos pendientes."
                if has_drafts else
                "Generar borradores IA para los grupos pendientes "
                "de este ítem en una sola llamada. No pisa devoluciones "
                "ya validadas."
            ),
        ):
            _run_batch_for_item(
                wd=wd,
                item=item,
                rubrica=rubrica,
                grupos=grupos,
                enunciado_nb=enunciado_nb,
                solucion_nb=solucion_nb,
            )
            st.rerun()

        for i, g in enumerate(grupos):
            nb = notebook_path(wd, g)
            if nb is None:
                # Entrega faltante: cuenta como "mal" para el puntaje;
                # visualmente en rojo, sin link (no hay nada para corregir).
                content = cell_div(COLOR_MAL, "—")
            else:
                fb = feedback_path(wd, g, item["key"])
                status, _, level = read_feedback(fb)
                if status == STATUS_OK:
                    color, cell_label = COLOR_OK, "ok"
                elif status == STATUS_OBS and level == LEVEL_BIEN:
                    color, cell_label = COLOR_OK, "bien"
                elif status == STATUS_OBS and level == LEVEL_REGULAR:
                    color, cell_label = COLOR_REGULAR, "reg"
                elif status == STATUS_OBS and level == LEVEL_MAL:
                    color, cell_label = COLOR_MAL, "mal"
                else:
                    # Pendiente o observación legacy sin nivel: en ambos
                    # casos falta clasificación, se muestra "abrir" para
                    # que el corrector entre y le asigne puntaje.
                    color, cell_label = COLOR_PENDIENTE, "abrir"
                href = f"?view=corr&grupo={g}&item={item['key']}"
                content = cell_link(color, href, cell_label)
            row_cols[i + 2].markdown(content, unsafe_allow_html=True)


def _run_batch_for_item(
    *,
    wd: Path,
    item: dict,
    rubrica: dict,
    grupos: list[str],
    enunciado_nb,
    solucion_nb,
) -> None:
    """Genera borradores IA en batch para los grupos pendientes del ítem.

    Saltea grupos sin notebook y grupos con feedback ya validado (no
    pisamos lo que el docente aprobó). Persiste cada borrador en
    `<grupo>/feedback/<item_key>.draft.md`.
    """
    ej = next(e for e in rubrica["ejercicios"] if e["id"] == item["ej_id"])
    rubric = item.get("rubric", {}) or {}

    # Grupos a procesar: con notebook y sin feedback validado.
    targets: list[tuple[str, GroupEntrega]] = []
    for g in grupos:
        nb = notebook_path(wd, g)
        if nb is None:
            continue
        fb = feedback_path(wd, g, item["key"])
        status, _, _ = read_feedback(fb)
        if status != STATUS_PENDIENTE:
            continue
        try:
            entrega_nb = load_notebook(nb)
        except Exception as e:
            st.toast(f"{g}: no pude leer el notebook ({e})", icon="⚠️")
            continue
        overrides = read_cell_overrides(wd, g)
        payload = _grupo_ai_payload(item=item, entrega_nb=entrega_nb, overrides=overrides)
        payload["grupo_id"] = g
        targets.append((g, payload))

    if not targets:
        st.toast("No hay grupos pendientes en este ítem.", icon="ℹ️")
        return

    shared = _shared_ai_context(
        item=item, ej=ej, enunciado_nb=enunciado_nb, solucion_nb=solucion_nb
    )
    prompt = build_batch_prompt(
        ej_titulo=ej.get("titulo", ""),
        item_tipo=item["tipo"],
        expected=rubric.get("expected", ""),
        common_errors=rubric.get("common_errors", []) or [],
        enunciado_src=shared["enunciado_src"],
        pregunta_src=shared["pregunta_src"],
        solucion_src=shared["solucion_src"],
        codigo_ctx_solucion=shared["codigo_ctx_solucion"],
        grupos=[p for _, p in targets],
    )
    expected_keys = [g for g, _ in targets]

    with st.spinner(f"Generando borradores IA para {len(targets)} grupos…"):
        try:
            drafts = generate_batch_drafts(prompt, expected_keys)
        except ClaudeSDKError as e:
            st.error(f"No pude generar los borradores: {e}")
            return
        except ValueError as e:
            st.error(f"La IA devolvió una respuesta no parseable: {e}")
            return

    saved = 0
    missing: list[str] = []
    for g, _ in targets:
        text = drafts.get(g)
        if text is None:
            missing.append(g)
            continue
        save_draft(draft_path(wd, g, item["key"]), text)
        saved += 1

    if saved:
        st.toast(f"Borradores guardados: {saved}", icon="✅")
    if missing:
        st.toast(
            f"La IA no devolvió borrador para: {', '.join(missing)}",
            icon="⚠️",
        )


# ─── Vista corrección ────────────────────────────────────────────────────────

def view_correccion(
    wd: Path,
    rubrica: dict,
    cfg: WorkdirConfig,
    items: list[dict],
    grupos: list[str],
    item_key: str,
    grupo: str,
) -> None:
    item = next((i for i in items if i["key"] == item_key), None)
    if item is None:
        st.error(f"Ítem `{item_key}` no existe en la rúbrica.")
        if st.button("← Volver a la matriz"):
            st.query_params.clear(); st.rerun()
        return
    if grupo not in grupos:
        st.error(f"Grupo `{grupo}` no existe.")
        if st.button("← Volver a la matriz"):
            st.query_params.clear(); st.rerun()
        return

    ej = next(e for e in rubrica["ejercicios"] if e["id"] == item["ej_id"])

    enunciado_nb = load_notebook(Path(cfg.notebook_enunciado))
    solucion_nb  = load_notebook(Path(cfg.notebook_solucion))
    entrega_path = notebook_path(wd, grupo)
    if entrega_path is None:
        st.error(f"`{grupo}` no tiene un `.ipynb` válido.")
        if st.button("← Volver a la matriz"):
            st.query_params.clear(); st.rerun()
        return
    entrega_nb = load_notebook(entrega_path)

    # Breadcrumb + volver
    col_bc, col_grp, col_back = st.columns([3, 1, 1])
    col_bc.markdown(
        f"**{rubrica.get('title') or 'Lab'}** &nbsp; · &nbsp; "
        f"**{item['ej_id']}** · {item['tipo']}"
    )
    if col_grp.button(
        grupo,
        key="bc-open-grupo",
        use_container_width=True,
        type="tertiary",
        help=f"Abrir {entrega_path.name} con la app por defecto del SO",
    ):
        if not open_in_os(entrega_path):
            st.toast(f"No pude abrir {entrega_path}", icon="⚠️")
    if col_back.button("← Volver a la matriz", use_container_width=True):
        st.query_params.clear(); st.rerun()

    item_idx = items.index(item)
    grupo_idx = grupos.index(grupo)

    def render_nav(prefix: str) -> None:
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

        if item_idx > 0:
            prev_item = items[item_idx - 1]
            if col_nav1.button(
                f"← {prev_item['ej_id']} · {prev_item['tipo']}",
                use_container_width=True, key=f"{prefix}-prev-item",
            ):
                st.query_params.update({"view": "corr", "grupo": grupo, "item": prev_item["key"]})
                st.rerun()
        else:
            col_nav1.button("←", disabled=True, use_container_width=True, key=f"{prefix}-prev-item-dis")

        if item_idx < len(items) - 1:
            next_item = items[item_idx + 1]
            if col_nav2.button(
                f"{next_item['ej_id']} · {next_item['tipo']} →",
                use_container_width=True, key=f"{prefix}-next-item",
            ):
                st.query_params.update({"view": "corr", "grupo": grupo, "item": next_item["key"]})
                st.rerun()
        else:
            col_nav2.button("→", disabled=True, use_container_width=True, key=f"{prefix}-next-item-dis")

        if grupo_idx > 0:
            prev_grupo = grupos[grupo_idx - 1]
            if col_nav3.button(
                f"↑ {prev_grupo}",
                use_container_width=True, key=f"{prefix}-prev-grupo",
            ):
                st.query_params.update({"view": "corr", "grupo": prev_grupo, "item": item_key})
                st.rerun()
        else:
            col_nav3.button("↑", disabled=True, use_container_width=True, key=f"{prefix}-prev-grupo-dis")

        if grupo_idx < len(grupos) - 1:
            next_grupo = grupos[grupo_idx + 1]
            if col_nav4.button(
                f"↓ {next_grupo}",
                use_container_width=True, key=f"{prefix}-next-grupo",
            ):
                st.query_params.update({"view": "corr", "grupo": next_grupo, "item": item_key})
                st.rerun()
        else:
            col_nav4.button("↓", disabled=True, use_container_width=True, key=f"{prefix}-next-grupo-dis")

    render_nav("nav-top")

    st.divider()

    col_ref, col_ent = st.columns(2)

    with col_ref:
        st.markdown("#### Referencia")
        if item["tipo"] == "código":
            enun_cell = find_cell(enunciado_nb, ej["enunciado_cell"])
            st.markdown(cell_source(enun_cell))

            st.markdown("**Solución oficial (código):**")
            sol_cell = find_cell(solucion_nb, item["code_cell"])
            st.code(cell_source(sol_cell), language="python")
        else:
            if item.get("pregunta_cell"):
                preg_cell = find_cell(enunciado_nb, item["pregunta_cell"])
                st.markdown(cell_source(preg_cell))
            else:
                # Ejercicios solo-análisis: la pregunta vive en el enunciado.
                enun_cell = find_cell(enunciado_nb, ej["enunciado_cell"])
                st.markdown(cell_source(enun_cell))

            st.markdown("**Solución oficial (análisis):**")
            sol_cell = find_cell(solucion_nb, item["answer_cell"])
            st.markdown(cell_source(sol_cell) or "_(no definida en el notebook de solución)_")

    with col_ent:
        st.markdown(f"#### Entrega — {grupo}")
        _render_entrega_panel(
            wd=wd,
            grupo=grupo,
            item=item,
            item_key=item_key,
            entrega_nb=entrega_nb,
            enunciado_nb=enunciado_nb,
        )

    st.divider()

    # Feedback
    st.markdown("### Feedback")

    fb_path = feedback_path(wd, grupo, item_key)
    current_status, current_content, current_level = read_feedback(fb_path)
    draft_p = draft_path(wd, grupo, item_key)
    ai_ok, draft_content = read_draft(draft_p)

    if current_status == STATUS_PENDIENTE:
        status_text, status_bg = "pendiente", COLOR_PENDIENTE
    elif current_status == STATUS_OK:
        status_text, status_bg = "sin observaciones", COLOR_OK
    elif current_level == LEVEL_BIEN:
        status_text, status_bg = "con observación · bien", COLOR_OK
    elif current_level == LEVEL_MAL:
        status_text, status_bg = "con observación · mal", COLOR_MAL
    elif current_level == LEVEL_REGULAR:
        status_text, status_bg = "con observación · regular", COLOR_REGULAR
    else:
        # Observación legacy sin nivel: cuenta como pendiente hasta que
        # se clasifique. Color gris para que sea consistente con la matriz.
        status_text, status_bg = "con observación · sin clasificar", COLOR_PENDIENTE

    st.markdown(
        f"<div style='margin-bottom:10px'>Estado actual: "
        f"<span style='background:{status_bg};padding:2px 10px;"
        f"border-radius:3px;color:#222;font-weight:500'>{status_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # El textarea es siempre la observación validada (o vacío si pendiente).
    # El borrador IA se muestra como referencia en una sección aparte abajo;
    # nunca pisa lo que el corrector ya guardó.
    widget_key = f"obs-{item_key}-{grupo}"
    pending_key = f"pending-{widget_key}"

    # Streamlit no permite modificar `session_state[widget_key]` después de
    # que el widget se instanció en el run actual. Para empujar texto al
    # textarea desde un handler (p.ej. "Usar este borrador", "Borrar",
    # "Marcar sin observaciones"), el handler escribe en `pending_key` y
    # acá, ANTES de instanciar el widget, lo aplicamos.
    if pending_key in st.session_state:
        st.session_state[widget_key] = st.session_state.pop(pending_key)

    text = st.text_area(
        "Observación al alumno (se incluye tal cual en el `grupo_XX.txt`)",
        value=current_content,
        height=180,
        placeholder=(
            "Solo escribí algo cuando haya un error a señalar. "
            "Si el ejercicio está bien, dejá esto vacío y hacé click en "
            "\"Marcar sin observaciones\"."
        ),
        key=widget_key,
    )

    # Selector de puntaje: independiente del texto. El corrector elige
    # bien/regular/mal acá; el botón "Guardar observación" guarda con
    # ese nivel + el texto del textarea (que puede estar vacío). El
    # textarea controla SOLO si va o no observación al txt final; el
    # puntaje siempre se aplica.
    #
    # Por default no hay nivel pre-seleccionado: el corrector tiene que
    # elegir explícitamente. Excepciones donde sí pre-seleccionamos:
    # - el ítem ya tiene un nivel guardado → mostramos ese para ver/editar.
    # - el ítem está en "sin observaciones" → por definición es bien (1pt),
    #   pre-seleccionamos `bien` para que si el corrector agrega un comentario
    #   no tenga que volver a clickear el puntaje.
    # En el resto (pendiente, obs legacy sin nivel) queda sin selección.
    level_key = f"level-{item_key}-{grupo}"
    pending_level_key = f"pending-{level_key}"
    level_options = [LEVEL_BIEN, LEVEL_REGULAR, LEVEL_MAL]
    level_labels = {
        LEVEL_BIEN:    "bien (1pt)",
        LEVEL_REGULAR: "regular (½pt)",
        LEVEL_MAL:     "mal (0pt)",
    }

    # Aplicar cambio de nivel pendiente antes de instanciar el radio. Lo
    # usan handlers como "Marcar sin observaciones" (setea bien) y
    # "Borrar" (limpia la selección) que corren después del widget en
    # orden de código.
    if pending_level_key in st.session_state:
        pending_val = st.session_state.pop(pending_level_key)
        if pending_val is None:
            st.session_state.pop(level_key, None)
        else:
            st.session_state[level_key] = pending_val

    if current_level in level_options:
        default_idx = level_options.index(current_level)
    elif current_status == STATUS_OK:
        default_idx = level_options.index(LEVEL_BIEN)
    else:
        default_idx = None

    # Auto-guardar el puntaje cuando el corrector lo cambia: el cambio del
    # nivel se persiste de inmediato. Se usa lo que esté en el textarea
    # como cuerpo (puede haber sido tipeado y no guardado todavía); si el
    # textarea está vacío, se cae al cuerpo del archivo en disco.
    #
    # Restricción: regular/mal no pueden quedar sin observación. Si el
    # corrector elige uno de esos dos sin haber escrito ni guardado nada,
    # bloqueamos el cambio y revertimos el radio.
    def _on_level_change() -> None:
        new_level = st.session_state.get(level_key)
        if new_level not in level_options:
            return
        status_now, body_disk, prev_level = read_feedback(fb_path)
        typed = st.session_state.get(widget_key, "")
        if not isinstance(typed, str):
            typed = ""
        body_to_save = typed if typed.strip() else body_disk

        if new_level in (LEVEL_REGULAR, LEVEL_MAL) and not body_to_save.strip():
            st.toast(
                "Para puntaje 'regular' o 'mal' hace falta una observación.",
                icon="⚠️",
            )
            # Revertir el radio al estado previo del archivo.
            if prev_level in level_options:
                st.session_state[level_key] = prev_level
            elif status_now == STATUS_OK:
                st.session_state[level_key] = LEVEL_BIEN
            else:
                st.session_state.pop(level_key, None)
            return

        # Si el archivo está como STATUS_OK y el corrector elige `bien`, no
        # hay nada que cambiar (sin obs ya implica bien = 1pt).
        if status_now == STATUS_OK and new_level == LEVEL_BIEN:
            return
        save_observation(fb_path, body_to_save, new_level)

    level = st.radio(
        "Puntaje",
        options=level_options,
        index=default_idx,
        format_func=lambda v: level_labels[v],
        horizontal=True,
        key=level_key,
        on_change=_on_level_change,
    )

    col_trasladar, col_save, col_ok, col_clear = st.columns([2, 2, 2, 1])

    has_draft = ai_ok or bool(draft_content)
    if col_trasladar.button(
        "Trasladar a observación",
        disabled=(not draft_content),
        use_container_width=True,
        key="btn-usar-borrador",
        help=(
            "No hay borrador IA todavía — generalo abajo." if not draft_content
            else "Copia el texto del borrador al recuadro de observación "
            "de arriba (sustituyendo lo que tenga). El borrador IA queda "
            "igual en su campo — podés volver a trasladarlo o regenerarlo "
            "cuantas veces quieras."
        ),
    ):
        st.session_state[pending_key] = draft_content
        st.rerun()

    if col_ok.button(
        "Marcar sin observaciones",
        use_container_width=True,
        key="btn-sin-obs",
    ):
        # "Sin observaciones" implica 1pt (bien). Si el corrector ya marcó
        # un puntaje distinto, no tiene sentido pasar a sin obs — sería
        # contradictorio. Si no hay puntaje seleccionado, lo ponemos en
        # bien automáticamente.
        current_radio = st.session_state.get(level_key)
        if current_radio in (LEVEL_REGULAR, LEVEL_MAL):
            st.toast(
                "El puntaje actual no es 'bien' — primero cambialo o "
                "borralo si querés marcar sin observaciones.",
                icon="⚠️",
            )
        else:
            save_sin_observaciones(fb_path)
            st.session_state[pending_key] = ""
            st.session_state[pending_level_key] = LEVEL_BIEN
            st.rerun()

    if current_status != STATUS_PENDIENTE:
        if col_clear.button(
            "Borrar",
            help="Volver a pendiente",
            use_container_width=True,
            key="btn-borrar",
        ):
            clear_feedback(fb_path)
            st.session_state[pending_key] = ""
            # `None` como pending → limpia la selección del radio.
            st.session_state[pending_level_key] = None
            st.rerun()

    if col_save.button(
        "Guardar observación",
        use_container_width=True,
        key="btn-guardar",
        disabled=(not text.strip()),
        help=(
            "El recuadro está vacío — escribí algo o usá 'Marcar sin "
            "observaciones' si no hay nada que decir."
            if not text.strip() else
            "Guarda el texto del recuadro como observación. Si todavía "
            "no elegiste puntaje, queda 'sin clasificar' (gris en la "
            "matriz) hasta que lo asignes."
        ),
    ):
        save_observation(fb_path, text, level)
        st.rerun()

    # Borrador IA: campo aparte, debajo de los botones. Se persiste en
    # disco (`.draft.md`) y se muestra como textarea read-only. El botón
    # "Trasladar a observación" (arriba) copia su contenido al recuadro
    # de observación. El botón "Generar/Regenerar borrador IA" queda
    # acá abajo, junto a la sección que produce: así el flujo espacial
    # va "abajo se genera → arriba se traslada".
    st.markdown("---")
    st.markdown("**Borrador IA**")
    if ai_ok:
        st.info(
            "La IA considera que esta entrega cumple la rúbrica. "
            "Si coincidís, marcá **sin observaciones**."
        )
    elif draft_content:
        # Sin `key=`: el widget no guarda estado entre runs, así siempre
        # refleja el `draft_content` actual (incluso después de regenerar).
        st.text_area(
            label="Borrador IA (referencia, no va al txt)",
            value=draft_content,
            height=140,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.caption("_Todavía no hay borrador IA para este ítem._")

    ia_label = "Regenerar borrador IA" if has_draft else "Generar borrador IA"
    if st.button(
        ia_label,
        key="btn-ia",
        help=(
            "Usa Claude Code local (sin API key) para redactar un borrador "
            "de observación en base al enunciado, la solución oficial y la "
            "entrega. El borrador aparece acá como referencia — no pisa "
            "el recuadro de observación."
        ),
    ):
        with st.spinner("Generando borrador con IA…"):
            shared = _shared_ai_context(
                item=item, ej=ej, enunciado_nb=enunciado_nb, solucion_nb=solucion_nb
            )
            payload = _grupo_ai_payload(
                item=item,
                entrega_nb=entrega_nb,
                overrides=read_cell_overrides(wd, grupo),
            )
            rubric = item.get("rubric", {}) or {}
            prompt = build_prompt(
                ej_titulo=ej.get("titulo", ""),
                item_tipo=item["tipo"],
                expected=rubric.get("expected", ""),
                common_errors=rubric.get("common_errors", []) or [],
                enunciado_src=shared["enunciado_src"],
                pregunta_src=shared["pregunta_src"],
                solucion_src=shared["solucion_src"],
                entrega_src=payload["entrega_src"],
                entrega_text_outputs=payload["entrega_text_outputs"],
                codigo_ctx_solucion=shared["codigo_ctx_solucion"],
                codigo_ctx_entrega=payload["codigo_ctx_entrega"],
                codigo_ctx_outputs=payload["codigo_ctx_outputs"],
            )
            try:
                draft = generate_draft(prompt)
            except ClaudeSDKError as e:
                st.error(f"No pude generar el borrador: {e}")
                st.stop()

        save_draft(draft_p, draft)
        st.rerun()

    st.divider()
    render_nav("nav-bottom")

    if st.button(
        "← Volver a la matriz",
        key="back-to-matrix-bottom",
        use_container_width=True,
    ):
        st.query_params.clear()
        st.rerun()


# ─── Dispatch ────────────────────────────────────────────────────────────────

def main() -> None:
    # Auto-seleccionar el workdir más reciente solo en el primer dispatch
    # de la sesión. Después, "Cambiar workdir" debe llevar al landing.
    if "workdir" not in st.session_state and not st.session_state.get("_auto_select_done"):
        rec = recents.list_recents()
        if rec:
            st.session_state["workdir"] = str(rec[0])
    st.session_state["_auto_select_done"] = True

    wd_str = st.session_state.get("workdir")
    if not wd_str:
        landing()
        return

    wd = Path(wd_str)
    if not has_config(wd):
        st.error(f"`{wd}` ya no tiene `.corrector/config.json`.")
        if st.button("Volver al landing"):
            st.session_state.pop("workdir", None)
            st.rerun()
        return

    cfg = load_config(wd)
    errs = validate_config(cfg)
    if errs:
        for e in errs:
            st.error(e)
        if st.button("Cambiar workdir"):
            st.session_state.pop("workdir", None)
            st.rerun()
        return

    try:
        rubrica = load_rubrica(Path(cfg.rubrica))
    except (FileNotFoundError, yaml.YAMLError) as e:
        st.error(f"No pude leer la rúbrica `{cfg.rubrica}`: {e}")
        if st.button("Cambiar workdir"):
            st.session_state.pop("workdir", None)
            st.rerun()
        return

    items = build_items(rubrica)
    grupos = list_grupos(wd)

    sidebar(wd, cfg)

    params = st.query_params
    if params.get("view") == "corr" and "grupo" in params and "item" in params:
        view_correccion(wd, rubrica, cfg, items, grupos, params["item"], params["grupo"])
    else:
        view_matriz(wd, rubrica, cfg, items, grupos)


if __name__ == "__main__":
    main()
