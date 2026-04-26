# 12 — Decisiones y roadmap

Este documento captura las decisiones explícitas del framework
(qué se construyó así por elección, qué se dejó afuera y por qué) y
los pendientes a futuro. Acá no hay especulación: si algo no está,
es porque deliberadamente se dejó para otro momento.

## Decisiones explícitas (qué NO se construyó)

### Sin multi-tenancy ni multi-user

**Qué significa:** la app no distingue usuarios. Cada workdir asume
un único docente operando.

**Por qué:** la materia tiene un docente (Javier Vélez). Agregar
roles, permisos o sync entre máquinas duplicaría la complejidad sin
beneficio claro. Si hay más docentes en el futuro, cada uno usa su
workdir y los txt finales se mergean a mano.

### Sin notas numéricas ni promedio en el `grupo_NN.txt`

**Qué significa:** el txt final solo lleva texto cualitativo (las
observaciones que el docente escribió). No hay nota numérica, ni
suma de puntos, ni promedio.

**Por qué:** el docente prefiere que la nota la asigne él en Moodle
después de leer las observaciones. Aparecer una nota ya calculada
en el txt distorsionaría la lectura del alumno y forzaría una
escala que no necesariamente coincide con la del curso.

El score (porcentaje del grupo según niveles bien/regular/mal) sí se
muestra en la matriz como ayuda visual, pero no se exporta.

### Sin edición directa de los `.ipynb` de los alumnos desde la app

**Qué significa:** la app nunca modifica `entrega.ipynb`. Si una
celda tiene mal el `cell_id`, el override vive en `cell_overrides.json`
adentro del workdir, no en el notebook.

**Por qué:**
- **Trazabilidad**: el docente puede comparar la entrega original
  con lo que la app procesó.
- **Idempotencia**: re-importar el zip sobrescribe `entrega.ipynb`
  sin tocar los overrides.
- **Backup**: el zip de Moodle queda como ground truth.

### Sin Vision API (Claude no ve imágenes)

**Qué significa:** los borradores IA solo usan texto del código y
texto de los outputs. Las imágenes (gráficos, imágenes estilizadas,
plots de loss) las mira el docente.

**Por qué:**
- **Costo**: las imágenes consumen muchos tokens. Para un batch de
  15 grupos × 12 ejercicios × varias imágenes por ejercicio, el
  costo se dispara.
- **Alcance**: la mayoría de los errores que la rúbrica espera son
  algorítmicos (formas de tensores mal calculadas, kernels mal
  definidos, división por la constante equivocada). Esos se ven en
  el código y en los stdout. Las preguntas que dependen de
  interpretar una imagen son una fracción menor.
- **Simplicidad**: agregar Vision implica integrar otra ruta del SDK
  y cambiar el manejo de tokens.

**Trade-off conocido:** preguntas tipo "interpretá la curva de loss"
o "describí qué cambió en la imagen estilizada" generan borradores
genéricos que el docente edita a mano.

### Sin botón "corregir todo con IA" automático

**Qué significa:** no hay un botón que corra el batch sobre todos los
ítems × todos los grupos sin intervención.

**Por qué:** requiere agregar un cuarto estado de feedback
(`ia-draft` no revisado por el docente) para distinguir el borrador
no validado del feedback final. Si no, el txt final estaría
contaminado con texto que el docente nunca leyó.

Discutido y descartado por ahora. Si se retoma, implica:

- Cambios en `state.py` (nuevo status).
- Cambios en `export.py` (filtrar drafts no revisados).
- Cambios en la matriz (nuevo color o badge para "borrador
  pendiente de revisión").

Hoy el batch IA por fila + revisión humana es el flujo aceptado.

### Sin la app abriéndose a un servidor remoto

**Qué significa:** la app está pensada para correr local. Botones
como "click en `G01` abre el `.ipynb` con la app del SO" usan
`subprocess.run(["open", ...])`, que se ejecuta donde corre el
backend de Streamlit.

**Por qué:**
- La app depende de la sesión local de Claude Code (sin API key) —
  un servidor remoto necesitaría su propia sesión, lo cual rompe el
  modelo "el docente paga su plan, la materia no paga API".
- Los workdirs viven en filesystem local del docente.

Si en el futuro alguien quiere desplegarla a un servidor: hay que
deshabilitar `open_in_os`, agregar autenticación, mover los workdirs
a un storage compartido, y resolver el tema de tokens de Claude.

### Sin migración automática de rúbricas v1 a v2 en disco

**Qué significa:** las rúbricas en `_TPS/rubricas/` están en v1 y
se quedan así. La app las normaliza a v2 en memoria al cargar.

**Por qué:** funcionan. No hay razón para tocarlas. Si en el futuro
una rúbrica necesita rubrics distintas para code y analysis (que es
cuando v2 deja de ser equivalente a v1), se migra a mano ese ítem.

## Pendientes (en orden de prioridad)

### 1. Testeo intensivo de los borradores IA con la mitigación A

**Estado:** la app está completa, falta uso real para evaluar la
calidad de los borradores con el contexto de código activo. Si
salen pobres, iterar `SYSTEM_PROMPT` en `app/ai.py`.

**Cómo evaluar:**
- Corregir Lab 2 2026 entero usando la app.
- Anotar (en una hoja aparte) los borradores que el docente tuvo que
  reescribir desde cero vs. los que solo editó vs. los que aceptó tal
  cual.
- Si la tasa "tuvo que reescribir desde cero" supera ~30%, ajustar el
  system prompt o el formato del prompt.

### 2. Lab 3a — corrección cuando lleguen las entregas

**Estado:** los notebooks y la rúbrica están listos. La fecha de
entrega no está confirmada al 2026-04-26.

**Acción:** cuando llegue el zip, crear un workdir nuevo y empezar la
corrección. No hace falta parche (el lab ya cumple la convención).

### 3. Skill `lab-new`

**Qué es:** una skill o script CLI que arranca un `.lab.md` desde
cero (con frontmatter, header, reglas, imports, checklist, footer
ya pre-rellenados) o desde un notebook existente (parseando el
`.ipynb` y emitiendo un `.lab.md` equivalente).

**Por qué:** hoy el flujo "empezar un lab nuevo" es copiar
`_ejemplo_formato.lab.md` y reescribirlo. Una skill que tome
prompts de alto nivel (`lab-new "Laboratorio 4 sobre RNNs"`) o
parse un notebook existente acortaría el ciclo.

**Estado:** planificado, no implementado.

### 4. Skill `lab-extract-metadata`

**Qué es:** un script CLI que aplique el prompt de
[`_TPS/metadata/prompt.md`](../_TPS/metadata/prompt.md) automáticamente
a un lab, generando `_Solucion.md` y `_eliminados.md` en
`_TPS/metadata/lab<id>/`.

**Por qué:** hoy se hace a mano (Lab 1a tiene este par; los demás
no). Si se quiere usar para analytics o búsqueda semántica
cross-lab, hace falta sistematizar.

**Estado:** planificado, no implementado. La lógica de extracción
está en el prompt; falta el wrapper Python que lo ejecute con
Claude SDK.

### 5. Generación de scaffolding de rúbrica desde el `.lab.md`

**Qué es:** extender `tools/lab_build.py` para que al compilar un
`.lab.md` también emita el scaffolding de la `.rubric.yaml`
correspondiente, con los `cell_id` ya pre-rellenados (sacándolos del
`.lab.md` directamente) y dejando `expected`/`common_errors` para
autogen o a mano.

**Por qué:** hoy la rúbrica autogen escanea los notebooks ya
compilados. Si el `.lab.md` ya conoce los `cell_id`, podríamos
ahorrarnos el escaneo.

Hay un esbozo en `_ejemplo_formato.lab.md` con bloques fenced
`yaml rubric` adentro de cada celda — esos bloques **hoy se
ignoran** por `lab_build.py`.

**Estado:** planificado, baja prioridad. La rúbrica autogen funciona
y este sería solo un atajo.

### 6. Pre-generación de borradores IA sin bloqueante

**Qué es:** un modo "background" donde, después de importar un zip,
la app dispare batches IA por fila en background y guarde drafts
para todos los ítems pendientes.

**Por qué:** hoy el docente tiene que clickear "IA" fila por fila,
esperando cada batch. Para 12 ejercicios y 15 grupos son 12 clicks.

**Por qué descartado por ahora:** ver "Sin botón corregir todo con
IA" arriba — la simplificación es no permitir borradores no
validados en el flujo de exportación.

**Camino posible si se retoma:** agregar un cuarto status `ia-draft`,
mostrar las celdas con drafts no revisados con un color distintivo
(p. ej. azul claro), y filtrar `build_grupo_txt` para que solo
exporte feedback validado.

### 7. UI más rápida para "celdas que no matchean"

**Qué es:** cuando un alumno borra varias celdas con `cell_id`
estable, hoy el docente entra a cada item de a uno y usa el navegador
`↑/↓`. Para 5 celdas mal en un grupo son 5 clicks.

**Camino posible:** un panel de "diagnóstico de grupo" que liste los
`cell_id` faltantes y ofrezca un wizard para mapearlos todos en un
flujo continuo.

**Estado:** no priorizado. La cantidad de grupos con este problema
es chica.

### 8. Tests automatizados

**Qué es:** suite de tests unitarios para los módulos de la app
(parsing, rúbrica v1/v2, intake, export) y al menos un test de smoke
para la UI.

**Estado:** no escritos. La app se valida con uso real.

**Camino posible:** `pytest` con fixtures de notebooks pequeños,
test rúbricas de ejemplo. Los módulos puros (`rubric.py`,
`export.py`, `state.py`, `nbparse.py`, `intake.py`) son fáciles de
testear sin Streamlit.

## Cosas que se discutieron y se descartaron explícitamente

- **Editar notebooks de alumnos desde la app** — descartado por
  trazabilidad e idempotencia.
- **Sync de workdirs entre máquinas via Git** — descartado, los
  workdirs no son código y los `.ipynb` no andan bien en git.
  Quien quiera sync, lo hace via OneDrive/Drive.
- **Vista de "diff" entre la entrega y la solución oficial** —
  discutido pero no construido. La vista corrección actual ya
  presenta ambas cosas lado a lado, lo cual fue suficiente.
- **Categorización automática de errores** (clustering de borradores
  IA para detectar errores frecuentes en una tanda) — interesante
  pero out of scope hoy.

## Histórico de decisiones que se cambiaron

### Schema de rúbrica: v1 → v2 con compat

**Decisión inicial (diciembre 2025):** schema v1 con `code_cell` /
`pregunta_cell` / `answer_cell` a nivel de ejercicio.

**Decisión refactorizada (abril 2026):** schema v2 con `items`
flexibles. La razón fue agregar soporte para ejercicios solo-análisis
(Lab 3b ej8) y para ejercicios con N codes o N analysis (que v1 no
permitía).

**Migración:** `load_rubrica` normaliza v1 → v2 en memoria. El disco
queda en v1 — si la app intenta sobrescribir, escribe v2. Hoy
ninguna ruta de la app llama a `save_rubrica` sobre rúbricas
existentes, así que el disco está estable.

### Estado del feedback: tres → cinco

**Decisión inicial:** tres estados (`pendiente` / `sin_observaciones`
/ `con_observacion`).

**Decisión actual:** sigue habiendo tres estados, pero `con_observacion`
tiene tres niveles (`bien` / `regular` / `mal`) que pintan colores
distintos en la matriz y dan puntaje distinto. La razón fue poder
mostrar un score por grupo en el header de la columna sin que el
docente tenga que mirar las observaciones una por una.

**Compat:** observaciones legacy sin marker se cuentan como pendientes
para el score (no se asume nada).

### Borrador IA: en-textarea → campo aparte

**Decisión inicial:** el borrador IA cargaba el textarea de
observación directamente.

**Problema:** si el docente había empezado a escribir su propia
observación y después generaba un borrador, el textarea se sobrescribía.

**Decisión actual:** el borrador vive en una sección aparte (textarea
readonly) y un botón "Trasladar a observación" lo copia al campo
principal cuando el docente decide. El borrador no se pierde si se
traslada — sigue ahí para regenerar o trasladar de nuevo.

### Open en SO: links → buttons

**Decisión inicial:** el header del grupo era un link `<a target="_blank">`
con `href="file://"`.

**Problema:** los browsers modernos bloquean `file://` desde páginas
web (incluso desde localhost) por seguridad.

**Decisión actual:** los headers son botones que llaman a
`subprocess.run(["open", ...])` — funciona porque Streamlit corre
local. La app no sería usable en un servidor remoto sin un cambio.
