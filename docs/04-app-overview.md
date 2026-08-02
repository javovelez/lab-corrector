# 04 — Arquitectura de la app

La app de corrección es una aplicación **Streamlit** que vive en
[`app/`](../app/). Está pensada para correr local (no como servicio
web), porque depende de la sesión de Claude Code del usuario y
porque abre archivos en el SO con `open` / `xdg-open` / `os.startfile`.

Esta página describe el modelo de datos central (el *workdir*), el
ciclo de vida de la app y los módulos que la componen. El flujo de
corrección de punta a punta está en
[05-app-flujo-correccion.md](05-app-flujo-correccion.md); el detalle
de la UI en [06-app-ui-detallada.md](06-app-ui-detallada.md).

## Concepto central: el workdir

La app no conoce ni labs ni asignaturas. Opera sobre un **workdir**:
una carpeta del filesystem elegida por el docente donde vive una
tanda de corrección. Todo lo persistente (config, entregas, feedback,
txt finales) vive adentro del workdir. Al reabrir la app con un
workdir conocido, retoma la corrección sin arrancar de cero.

Layout del workdir:

```
<workdir>/
├── .corrector/
│   ├── config.json                  ← paths absolutos al enunciado/solución/rúbrica
│   └── rubrica.yaml                 ← solo si fue auto-generada (si no, vive afuera)
├── entregas_lab3.zip                ← input de Moodle, intacto
├── grupo_01/
│   ├── entrega.ipynb                ← copia normalizada del zip (nombre original descartado)
│   ├── grupo_01.txt                 ← devolución final para Moodle (escrito al pedido)
│   ├── cell_overrides.json          ← mapeos `expected_id → actual_id` (opcional)
│   └── feedback/
│       ├── ej1-code.md              ← una observación por ítem corregible
│       ├── ej1-code.draft.md        ← borrador IA opcional
│       ├── ej1-analisis.md
│       └── ...
├── grupo_02/
│   └── ...
└── grupo_15/
    └── ...
```

Detalle de cada formato en [09-formatos-archivo.md](09-formatos-archivo.md).

### Por qué es un workdir y no una base de datos

- Backup trivial: zippeás la carpeta y tenés todo (incluido el
  feedback).
- Inspección con cualquier editor: cada `.md` de feedback es un
  archivo abrible.
- Multi-cuatrimestre / multi-lab: cada workdir es su propia tanda. Si
  el docente corrige Lab 2 2026 y Lab 3a 2026 en paralelo, son dos
  carpetas separadas.
- Sin estado compartido entre máquinas: el workdir vive en disco
  local (o en OneDrive/Drive si el docente quiere sync). La app no
  asume nada.

### Registry global de workdirs recientes

`~/.lab_corrector/recent.json` mantiene una lista de paths recientes
para que el landing los pueda ofrecer como atajo. Lo gestiona
[`app/recents.py`](../app/recents.py):

- `list_recents()` → devuelve los paths que siguen existiendo en
  disco, ordenados por recencia. Filtra los que se borraron.
- `touch(workdir)` → mueve el path al tope. Limita a `MAX_RECENTS=10`.
- `forget(workdir)` → lo saca de la lista.

La app llama a `touch` cada vez que el docente entra a un workdir,
y `forget` desde el botón "quitar" del landing.

## Ciclo de vida de la app

1. **Streamlit arranca**, ejecuta `app/main.py::main()`.
2. Si no hay `workdir` en `st.session_state`, intenta auto-seleccionar
   el más reciente. Si no hay recientes, muestra el `landing()`.
3. **Landing**: lista de recientes + form "abrir otra carpeta" + form
   "nueva corrección". Detalle en [05](05-app-flujo-correccion.md).
4. Al elegir un workdir, lo carga en `st.session_state["workdir"]` y
   re-ejecuta.
5. **Validación**: chequea que existan `<workdir>/.corrector/config.json`
   y los paths que apunta (notebook enunciado, notebook solución,
   rúbrica). Si algo falla, ofrece volver al landing.
6. **Carga de rúbrica + items**: `load_rubrica` lee el YAML y
   normaliza a v2; `build_items` aplana los ejercicios en una lista
   plana de ítems corregibles.
7. **Sidebar**: título del lab + path del workdir + botón "cambiar
   workdir" + expander para importar más entregas + expander con la
   config.
8. **Dispatch por query params**:
   - `?view=corr&grupo=grupo_03&item=ej2-analisis` → vista corrección
     del ítem para ese grupo.
   - cualquier otra cosa → vista matriz.
9. La interacción del usuario dispara handlers que escriben al
   filesystem y llaman a `st.rerun()` para refrescar el estado.

## Módulos

| Archivo | Responsabilidad |
|---|---|
| [`app/main.py`](../app/main.py) | UI: landing, sidebar, dispatch matriz/corrección, navegador de celdas, panel de IA, niveles de puntaje. Único archivo que toca Streamlit. |
| [`app/workdir.py`](../app/workdir.py) | `WorkdirConfig` dataclass + I/O de `.corrector/config.json`. `validate_config` chequea que los paths existan. |
| [`app/recents.py`](../app/recents.py) | Registry global `~/.lab_corrector/recent.json`: `list_recents`, `touch`, `forget`. |
| [`app/intake.py`](../app/intake.py) | `intake_zip(zip_path, workdir)`: descomprime el zip de Moodle a temp, normaliza nombres `Grupo N_<id>_assignsubmission_file/` → `grupo_NN/entrega.ipynb`. Preserva `feedback/` al re-importar. |
| [`app/rubric.py`](../app/rubric.py) | `load_rubrica` (lee YAML y normaliza v1 → v2 en memoria) y `save_rubrica` (siempre escribe v2). |
| [`app/rubric_gen.py`](../app/rubric_gen.py) | Generación de rúbrica. `scan_ejercicios` (sin IA) + `generate_rubrica` (con IA, una llamada por ítem). |
| [`app/state.py`](../app/state.py) | Persistencia del feedback. Estados (`STATUS_PENDIENTE` / `STATUS_OK` / `STATUS_OBS`), niveles (`bien` / `regular` / `mal`), markers (`SIN_OBS_MARKER`, `AI_OK_MARKER`, `LEVEL_MARKERS`). Drafts IA (`.draft.md`) y cell overrides (`cell_overrides.json`). |
| [`app/export.py`](../app/export.py) | `build_grupo_txt` (ensambla el txt final a partir de los `feedback/*.md`), `count_observaciones`, `compute_grupo_score` (porcentaje del grupo basado en niveles). |
| [`app/ai.py`](../app/ai.py) | `build_prompt` + `generate_draft` (una entrega) y `build_batch_prompt` + `generate_batch_drafts` (N entregas en una sola llamada). Define `GroupEntrega` (TypedDict). |
| [`app/grupos.py`](../app/grupos.py) | `list_grupos(workdir)` (subdirs `grupo_*`) y `notebook_path(workdir, grupo)` (único `.ipynb` adentro). |
| [`app/nbparse.py`](../app/nbparse.py) | Parseo de notebooks Jupyter sin dependencia de `nbformat`. `find_cell`, `cell_source`, `cell_outputs` (normaliza streams, execute_results, errors a `{kind, text/data_b64}`). Cachea con `lru_cache` por path. |
| [`app/requirements.txt`](../app/requirements.txt) | `streamlit>=1.32`, `pyyaml>=6.0`, `claude-agent-sdk>=0.1.64`. |
| [`app/.venv/`](../app/.venv) | Virtualenv local (no checked in). Ver [11-instalacion.md](11-instalacion.md). |

## Dependencias internas entre módulos

```
                        main.py
                ╱  ╱  ╱   │   ╲   ╲   ╲
               ╱  ╱  ╱    │    ╲   ╲   ╲
        recents intake  workdir  ai  rubric  state  grupos  nbparse  export
                            │      │     │             │      │
                            │      ▼     ▼             │      │
                            │   rubric_gen ─►  nbparse │      ▼
                            │                          │   nbparse
                            ▼                          ▼
                          (rubric_gen también lee notebooks vía nbparse)
```

`main.py` es el orquestador y el único módulo que conoce Streamlit.
Todos los demás son librerías puras Python sin estado global más allá
del filesystem. Esto facilita testear cada uno por separado y reusar
`rubric_gen` desde `app/rubric_build.py`.

## State management en Streamlit

Streamlit re-ejecuta el script entero ante cualquier interacción del
usuario. La app usa `st.session_state` para tres cosas:

- `st.session_state["workdir"]` — path absoluto del workdir activo.
- Pending edits en widgets (`pending-{key}`) — para empujar texto al
  textarea de observación desde un handler. Streamlit no permite
  modificar un `session_state[widget_key]` después de instanciar el
  widget; el patrón usado es: el handler escribe en `pending-<key>`,
  y al inicio del run siguiente se transfiere a `<key>` antes de
  instanciar el widget.
- Navegación de celdas (`nav-cell-{grupo}-{item_key}`) — qué celda
  mostrar en el panel "entrega" cuando el alumno tiene la respuesta
  en una celda con id distinto al esperado.

El feedback nunca vive en `session_state` — siempre se persiste a
disco. Eso garantiza que un crash de Streamlit (o cerrar el browser)
no pierde datos.

## Caching

`load_notebook(path)` en `nbparse.py` está envuelto con `lru_cache(maxsize=32)`.
Eso significa que el mismo notebook se lee de disco una sola vez por
sesión. Limitación: si el alumno **modifica** el `.ipynb` mientras la
app está corriendo, hay que recargar manualmente (en la práctica
nunca pasa: las entregas son inmutables una vez subidas).

Si esto se vuelve un problema, hay que invalidar el cache en
`intake_zip` después de re-importar.

## Estilo visual

- Tema configurado en [`.streamlit/config.toml`](../.streamlit/config.toml):
  fondo crema cálido (`#F8F5EE`), sidebar más oscura (`#EFEAE0`),
  acento sage (`#7B8B6F`).
- Paleta de la matriz (semáforo): verde `#A5D6A7`, amarillo `#FFE082`,
  rojo `#EF9A9A`, gris `#E0E0E0`.
- CSS injectado en `main.py::st.markdown` para tipear los botones
  secondary/tertiary (Streamlit 1.56 los pinta blancos por defecto).

Detalle visual en [11-instalacion.md](11-instalacion.md).
