# 11 — Instalación

Cómo levantar el framework en una máquina nueva. Cubre el entorno de
ejecución de la app (Streamlit) y los CLIs (`lab_build.py`,
`rubric_build.py`).

## Requisitos del sistema

- **Python 3.10 o superior**. Las anotaciones de tipo (`int | None`,
  `dict[str, str]`, `list[dict]`) requieren ≥ 3.10.
- **Claude Code instalado y con sesión activa** (claude.ai). Necesario
  para `claude-agent-sdk` (rúbrica autogen y borradores IA).
- **macOS, Linux o Windows**. La app fue probada en macOS principalmente;
  el código tiene branches para los tres SO en `open_in_os`.
- (Solo para Windows) PowerShell o cmd con permisos de ejecución de
  scripts si vas a usar el venv directo.

## Paso 1 — Clonar / abrir el repo

```bash
cd /ruta/al/repo/lab-corrector
```

(O cualquier ruta donde tengas el proyecto.)

## Paso 2 — Crear el venv para la app

```bash
python3 -m venv app/.venv
```

Esto crea `app/.venv/` con un Python aislado del sistema. Los CLIs
(`app/rubric_build.py`) también usan este venv.

## Paso 3 — Instalar dependencias

```bash
app/.venv/bin/pip install -U pip
app/.venv/bin/pip install -r app/requirements.txt
```

Contenido de [`app/requirements.txt`](../app/requirements.txt):

```
streamlit>=1.32
pyyaml>=6.0
claude-agent-sdk>=0.1.64
```

Versiones probadas:

- `streamlit==1.56` (la que está instalada en el venv del docente).
- `claude-agent-sdk==0.1.64+`.
- `pyyaml==6.x`.

### Verificar que todo cargó bien

```bash
app/.venv/bin/python -c "import streamlit, yaml, claude_agent_sdk; \
    print('streamlit', streamlit.__version__); \
    print('yaml', yaml.__version__); \
    print('claude-agent-sdk OK')"
```

Salida esperada (versiones pueden variar):

```
streamlit 1.56.0
yaml 6.0.1
claude-agent-sdk OK
```

## Paso 4 — Levantar la app

```bash
app/.venv/bin/streamlit run app/main.py
```

Streamlit escucha en `http://localhost:8501` y abre el browser. La
primera vez muestra el landing (no hay workdirs recientes). Detalle
del flujo en [05-app-flujo-correccion.md](05-app-flujo-correccion.md).

Para frenar la app: `Ctrl+C` en la terminal.

### Levantar en otro puerto

```bash
app/.venv/bin/streamlit run app/main.py --server.port 8502
```

### Levantar sin abrir el browser

```bash
app/.venv/bin/streamlit run app/main.py --server.headless true
```

Útil si querés acceder desde otra ventana del browser que ya tenés
abierta.

## Paso 5 — (Opcional) Configurar el tema

El tema vive en [`.streamlit/config.toml`](../.streamlit/config.toml):

```toml
[theme]
base = "light"
backgroundColor = "#F8F5EE"           # crema cálido — fondo principal
secondaryBackgroundColor = "#EFEAE0"  # sidebar, inputs
primaryColor = "#7B8B6F"              # acento sage
textColor = "#2B2B2B"
font = "sans serif"
```

La paleta combina con los semáforos de la matriz (verde `#A5D6A7`,
amarillo `#FFE082`, rojo `#EF9A9A`, gris `#E0E0E0`). Si lo cambiás,
chequeá que los colores de los badges sigan siendo legibles.

## CLIs disponibles

### `lab_build.py` (plugin)

Compila un `.lab.md` a dos `.ipynb` (enunciado + solución). No usa
el venv (es solo Python estándar):

```bash
python plugins/lab-notebook/scripts/lab_build.py _TPS/sources/Laboratorio_3a.lab.md
```

Detalle en [02-autoria-lab-md.md](../plugins/lab-notebook/skills/lab-notebook/reference/lab-md.md).

### `app/rubric_build.py`

Genera la rúbrica YAML de un lab vía Claude. Usa el venv (necesita
`claude-agent-sdk` y `pyyaml`):

```bash
app/.venv/bin/python app/rubric_build.py 3b
```

Asume layout `_TPS/`: lee
`_TPS/Laboratorios/Laboratorio_<id>.ipynb` y
`_TPS/Soluciones/Laboratorio_<id>_Solucion.ipynb`, escribe a
`_TPS/rubricas/Laboratorio_<id>.rubric.yaml`. La solución tiene que
tener outputs guardados.

Detalle en [03-rubricas.md](03-rubricas.md).

### `plugins/lab-notebook/scripts/lab2_split_pregunta.py`

Parche one-off para Lab 2 (notebooks pre-framework). No usa el venv
(es Python estándar):

```bash
python plugins/lab-notebook/scripts/lab2_split_pregunta.py <notebook.ipynb>
```

Detalle en [10-troubleshooting.md#lab-2-celdas-pegadas](10-troubleshooting.md#lab-2--celdas-pegadas).

## Estructura de archivos relevante para el deploy

```
material clases/
├── app/
│   ├── .venv/                ← virtualenv (no checked-in)
│   ├── *.py                  ← código de la app
│   └── requirements.txt
├── plugins/lab-notebook/scripts/
│   ├── lab_build.py
│   ├── rubric_build.py
│   └── lab2_split_pregunta.py
├── _TPS/
│   ├── sources/              ← .lab.md
│   ├── Laboratorios/         ← .ipynb (enunciados)
│   ├── Soluciones/           ← .ipynb (soluciones)
│   ├── rubricas/             ← .rubric.yaml
│   └── metadata/             ← _Solucion.md, _eliminados.md, prompt.md
├── .streamlit/
│   └── config.toml           ← tema visual de la app
├── .claude/
│   ├── settings.json         ← permisos del agente Claude Code (no afecta la app)
│   └── settings.local.json
└── docs/                     ← este wiki
```

Ningún workdir vive adentro del repo. Los workdirs son carpetas que
el docente elige fuera del repo (típicamente en `OneDrive/UTN/<año>/<lab>/`).

## Variables de entorno

Ninguna. La app no usa env vars. Todo se configura por filesystem
(workdir + `~/.lab_corrector/recent.json`).

## ¿Acceso a Claude Code desde el SDK falla?

Síntomas posibles:

- `ClaudeSDKError: Failed to spawn ...`
- Timeouts al generar borrador.
- "claude-agent-sdk OK" al instalarse pero falla al ejecutarse.

Cosas para chequear:

1. **Claude Code instalado**: en macOS, `which claude` debe devolver
   un path. Si no, instalar desde claude.ai.
2. **Sesión activa**: abrir Claude Code una vez (terminal o VS Code)
   para autenticar la sesión.
3. **Versión del SDK**: `app/.venv/bin/pip show claude-agent-sdk` —
   si es < 0.1.64, actualizar:
   ```
   app/.venv/bin/pip install -U claude-agent-sdk
   ```

## Permisos del agente Claude Code

`.claude/settings.json` tiene una lista `permissions.allow` con los
comandos pre-aprobados que el agente puede ejecutar sin pedir
permiso interactivo. Es útil cuando se trabaja con el agente en VS
Code o terminal y no se quiere autorizar cada `Bash` individualmente.

**No afecta a la app de corrección** — la app pasa
`setting_sources=[]` al SDK explícitamente, así que los permisos del
agente no inciden en `generate_draft` ni `generate_rubrica`.

`settings.local.json` (gitignore) sirve para overrides personales
del docente que no se commitean.

## Actualización de dependencias

Si `streamlit` o `claude-agent-sdk` lanzan una versión nueva:

```bash
app/.venv/bin/pip install -U streamlit claude-agent-sdk pyyaml
```

Probar la app antes de seguir trabajando — un major bump de
Streamlit puede romper algún hack CSS de la matriz (usa selectores
internos como `.st-key-*` que pueden cambiar entre versiones).

## Backup y migración

El framework no tiene base de datos. Para backupear:

- **Workdirs**: zippear cada carpeta entera.
- **Repo**: `git push` (o copiar el directorio).
- **Registry de recientes** (`~/.lab_corrector/recent.json`): solo si
  querés preservar los atajos del landing en otra máquina. No
  contiene datos críticos.

Para mover a otra máquina:

1. Clonar el repo / copiar la carpeta.
2. Crear el venv en la nueva máquina (`python3 -m venv app/.venv`
   + `pip install -r app/requirements.txt`).
3. Copiar los workdirs (si están adentro de OneDrive/Drive, ya se
   sincronizan solos).
4. Instalar Claude Code en la máquina nueva y autenticar.
5. Levantar la app.

## Modo desarrollo

Para tocar el código de la app y ver cambios sin reiniciar:

Streamlit detecta cambios en archivos `.py` automáticamente y ofrece
"Rerun" en el browser. Para que el cambio de un módulo importado se
refleje, conviene "Clear cache" desde el menú de Streamlit (las tres
líneas arriba a la derecha).
