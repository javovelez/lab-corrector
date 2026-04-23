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

import recents
from ai import build_prompt, generate_draft
from export import build_grupo_txt, count_observaciones
from grupos import list_grupos, notebook_path
from intake import intake_zip
from nbparse import cell_outputs, cell_source, find_cell, load_notebook
from rubric import load_rubrica, save_rubrica
from rubric_gen import generate_rubrica
from state import (
    STATUS_OBS,
    STATUS_OK,
    STATUS_PENDIENTE,
    clear_feedback,
    feedback_path,
    read_feedback,
    save_observation,
    save_sin_observaciones,
)
from workdir import (
    WorkdirConfig,
    config_path,
    has_config,
    load_config,
    save_config,
    validate_config,
)


COLOR_PENDIENTE = "#E0E0E0"
COLOR_OK        = "#A5D6A7"
COLOR_OBS       = "#FFB74D"
COLOR_MISSING   = "#EF9A9A"


st.set_page_config(page_title="Corrector de notebooks", layout="wide")


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
    path_str = st.text_input(
        "Path absoluto al workdir",
        key="landing-open-path",
        placeholder="/Users/javiervelez/Downloads/RNP_Lab3_abril",
    )
    if st.button("Abrir", key="landing-open-btn"):
        clean = _clean_path(path_str)
        p = Path(clean).expanduser() if clean else None
        if p is None or not p.is_dir():
            st.error("Esa carpeta no existe.")
        else:
            _enter_workdir(p)

    st.markdown("### Nueva corrección")
    with st.form("new-workdir"):
        wd_str = st.text_input(
            "Carpeta (workdir) — donde tenés o vas a poner el zip de Moodle",
            placeholder="/Users/javiervelez/Downloads/RNP_Lab3_abril",
        )
        title = st.text_input("Título del lab", placeholder="Laboratorio 3 — Transferencia")
        nb_enun = st.text_input("Notebook de enunciado (.ipynb)", placeholder="/ruta/al/Laboratorio_X.ipynb")
        nb_sol  = st.text_input("Notebook de solución (.ipynb)",  placeholder="/ruta/al/Laboratorio_X_Solucion.ipynb")
        zip_path = st.text_input(
            "Zip de Moodle (opcional — podés importarlo después)",
            placeholder="/Users/javiervelez/Downloads/entregas_lab3.zip",
        )
        rubric_mode = st.radio(
            "Rúbrica",
            ["Generar automáticamente desde la solución (usa Claude)", "Ya tengo una (.rubric.yaml)"],
            index=0,
        )
        rubric_existing = st.text_input(
            "Path al .rubric.yaml existente (solo si elegiste la segunda opción)",
            placeholder="/ruta/a/rubrica.yaml",
        )

        submitted = st.form_submit_button("Crear", type="primary", use_container_width=True)
        if submitted:
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
        zip_str = st.text_input(
            "Ruta al zip de Moodle",
            value=st.session_state.get("intake-zip", ""),
            placeholder="/Users/javiervelez/Downloads/entregas_lab3.zip",
            key="intake-zip-input",
        )
        if st.button("Importar", use_container_width=True, key="btn-intake"):
            clean = _clean_path(zip_str)
            zp = Path(clean).expanduser() if clean else None
            if zp is None:
                st.error("Indicá la ruta al zip.")
            else:
                with st.spinner("Importando…"):
                    report = intake_zip(zp, wd)
                st.session_state["intake-last-report"] = {
                    "imported": report.imported,
                    "skipped":  report.skipped,
                    "warnings": report.warnings,
                }
                st.session_state["intake-zip"] = zip_str
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


# ─── Vista matriz ────────────────────────────────────────────────────────────

def view_matriz(wd: Path, rubrica: dict, items: list[dict], grupos: list[str]) -> None:
    st.title(rubrica.get("title") or "Corrector")

    col_meta1, col_meta2 = st.columns([1, 3])
    col_meta1.metric("Grupos", len(grupos))
    col_meta2.metric("Ítems corregibles", len(items))

    st.markdown("### Estado de corrección")
    pill = "padding:3px 10px;border-radius:3px;font-size:13px;color:#222;font-weight:500"
    st.markdown(
        f"<div style='margin-bottom:12px'>"
        f"<span style='background:{COLOR_PENDIENTE};{pill}'>pendiente</span> &nbsp; "
        f"<span style='background:{COLOR_OK};{pill}'>sin observaciones</span> &nbsp; "
        f"<span style='background:{COLOR_OBS};{pill}'>con observación</span> &nbsp; "
        f"<span style='background:{COLOR_MISSING};{pill}'>entrega faltante / rota</span>"
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
            header_cols[i + 1].markdown(
                f"<div style='text-align:center'><b>{label}</b><br>"
                f"<span style='color:#c00;font-size:10px'>sin ipynb</span></div>",
                unsafe_allow_html=True,
            )
        else:
            if header_cols[i + 1].button(
                label,
                key=f"open-nb-{g}",
                use_container_width=True,
                type="tertiary",
                help=f"Abrir {nb.name} con la app por defecto del SO",
            ):
                if not open_in_os(nb):
                    st.toast(f"No pude abrir {nb}", icon="⚠️")
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

    prev_ej = None
    for item in items:
        if prev_ej is not None and item["ej_id"] != prev_ej:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        prev_ej = item["ej_id"]

        row_cols = st.columns([3] + [1] * len(grupos))
        if item["tipo"] == "código":
            label = f"**{item['ej_id']}** · código &nbsp; — &nbsp; {item['titulo']}"
        else:
            label = (
                f"<span style='color:#666'>&nbsp;&nbsp;&nbsp;&nbsp;↳ "
                f"<b>{item['ej_id']}</b> · análisis</span>"
            )
        row_cols[0].markdown(label, unsafe_allow_html=True)

        for i, g in enumerate(grupos):
            nb = notebook_path(wd, g)
            if nb is None:
                content = cell_div(COLOR_MISSING, "—")
            else:
                fb = feedback_path(wd, g, item["key"])
                status, _ = read_feedback(fb)
                if status == STATUS_OK:
                    color, cell_label = COLOR_OK, "ok"
                elif status == STATUS_OBS:
                    color, cell_label = COLOR_OBS, "obs"
                else:
                    color, cell_label = COLOR_PENDIENTE, "abrir"
                href = f"?view=corr&grupo={g}&item={item['key']}"
                content = cell_link(color, href, cell_label)
            row_cols[i + 1].markdown(content, unsafe_allow_html=True)


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
        if item["tipo"] == "código":
            student_cell = find_cell(entrega_nb, item["code_cell"])
            if student_cell is None:
                st.warning(f"El alumno no tiene la celda `{item['code_cell']}` en su entrega.")
            else:
                st.code(cell_source(student_cell), language="python")
                st.markdown("**Outputs:**")
                render_student_outputs(cell_outputs(student_cell))
        else:
            student_cell = find_cell(entrega_nb, item["answer_cell"])
            if student_cell is None:
                st.warning(f"El alumno no tiene la celda `{item['answer_cell']}` en su entrega.")
            else:
                source = cell_source(student_cell).strip()
                if not source or source == "*(Escribí tu respuesta acá)*":
                    st.caption("_El alumno no respondió._")
                else:
                    st.markdown(source)

    st.divider()

    # Feedback
    st.markdown("### Feedback")

    fb_path = feedback_path(wd, grupo, item_key)
    current_status, current_content = read_feedback(fb_path)

    status_label = {
        STATUS_PENDIENTE: ("pendiente", COLOR_PENDIENTE),
        STATUS_OK:        ("sin observaciones", COLOR_OK),
        STATUS_OBS:       ("con observación", COLOR_OBS),
    }[current_status]
    st.markdown(
        f"<div style='margin-bottom:10px'>Estado actual: "
        f"<span style='background:{status_label[1]};padding:2px 10px;"
        f"border-radius:3px;color:#222;font-weight:500'>{status_label[0]}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    widget_key = f"obs-{item_key}-{grupo}"
    draft_key = f"draft-{item_key}-{grupo}"

    if draft_key in st.session_state:
        st.session_state[widget_key] = st.session_state.pop(draft_key)

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

    col_ia, col_save, col_ok, col_clear = st.columns([2, 2, 2, 1])

    if col_ia.button(
        "Generar borrador IA",
        help=(
            "Usa Claude Code local (sin API key) para redactar un borrador "
            "de observación en base al enunciado, la solución oficial y la entrega."
        ),
        use_container_width=True,
        key="btn-ia",
    ):
        with st.spinner("Generando borrador con IA…"):
            target_cell_id = item["code_cell"] if item["tipo"] == "código" else item["answer_cell"]
            entrega_cell_for_ai = find_cell(entrega_nb, target_cell_id)
            entrega_src = cell_source(entrega_cell_for_ai)
            entrega_outs_text = (
                text_outputs_concat(cell_outputs(entrega_cell_for_ai))
                if item["tipo"] == "código" else ""
            )
            solucion_src = cell_source(find_cell(solucion_nb, target_cell_id))
            pregunta_src = (
                cell_source(find_cell(enunciado_nb, item.get("pregunta_cell") or ""))
                if item["tipo"] == "análisis" else ""
            )

            # Para ítems de análisis, incluimos el código del mismo ejercicio
            # (solución, entrega y sus outputs de texto) como contexto: la
            # pregunta suele depender de lo que el alumno hizo o vio. Si el
            # ejercicio tiene varios ítems de código (parte A, parte B, ...),
            # se concatenan todos.
            codigo_ctx_solucion = codigo_ctx_entrega = codigo_ctx_outputs = ""
            companion = item.get("companion_code_cells", []) if item["tipo"] == "análisis" else []
            if companion:
                sol_parts: list[str] = []
                ent_parts: list[str] = []
                out_parts: list[str] = []
                for code_id in companion:
                    label = f"# celda {code_id}"
                    sol_src = cell_source(find_cell(solucion_nb, code_id))
                    if sol_src.strip():
                        sol_parts.append(f"{label}\n{sol_src}")
                    ent_cell = find_cell(entrega_nb, code_id)
                    ent_src = cell_source(ent_cell)
                    if ent_src.strip():
                        ent_parts.append(f"{label}\n{ent_src}")
                    out_text = text_outputs_concat(cell_outputs(ent_cell))
                    if out_text.strip():
                        out_parts.append(f"{label}\n{out_text}")
                codigo_ctx_solucion = "\n\n".join(sol_parts)
                codigo_ctx_entrega = "\n\n".join(ent_parts)
                codigo_ctx_outputs = "\n\n".join(out_parts)

            rubric = item.get("rubric", {}) or {}
            prompt = build_prompt(
                ej_titulo=ej.get("titulo", ""),
                item_tipo=item["tipo"],
                expected=rubric.get("expected", ""),
                common_errors=rubric.get("common_errors", []) or [],
                enunciado_src=cell_source(find_cell(enunciado_nb, ej["enunciado_cell"])),
                pregunta_src=pregunta_src,
                solucion_src=solucion_src,
                entrega_src=entrega_src,
                entrega_text_outputs=entrega_outs_text,
                codigo_ctx_solucion=codigo_ctx_solucion,
                codigo_ctx_entrega=codigo_ctx_entrega,
                codigo_ctx_outputs=codigo_ctx_outputs,
            )
            try:
                draft = generate_draft(prompt)
            except ClaudeSDKError as e:
                st.error(f"No pude generar el borrador: {e}")
                st.stop()

        st.session_state[draft_key] = draft
        st.rerun()

    if col_save.button(
        "Guardar observación",
        type="primary",
        disabled=not text.strip(),
        use_container_width=True,
        key="btn-guardar",
    ):
        save_observation(fb_path, text)
        st.rerun()

    if col_ok.button(
        "Marcar sin observaciones",
        use_container_width=True,
        key="btn-sin-obs",
    ):
        save_sin_observaciones(fb_path)
        st.rerun()

    if current_status != STATUS_PENDIENTE:
        if col_clear.button(
            "Borrar",
            help="Volver a pendiente",
            use_container_width=True,
            key="btn-borrar",
        ):
            clear_feedback(fb_path)
            st.rerun()

    st.divider()
    render_nav("nav-bottom")


# ─── Dispatch ────────────────────────────────────────────────────────────────

def main() -> None:
    # Auto-seleccionar el workdir más reciente si no hay uno en la sesión.
    if "workdir" not in st.session_state:
        rec = recents.list_recents()
        if rec:
            st.session_state["workdir"] = str(rec[0])

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
        view_matriz(wd, rubrica, items, grupos)


if __name__ == "__main__":
    main()
