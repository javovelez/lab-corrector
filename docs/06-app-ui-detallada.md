# 06 — UI detallada de la app

Esta página entra al detalle de los componentes de la UI: matriz,
navegador de celdas, niveles de puntaje, panel IA y comportamientos
sutiles que no son obvios mirando la pantalla.

## Landing

Cuando `st.session_state["workdir"]` está vacío.

```
┌────────────────────────────────────────────┐
│ Corrector de notebooks                     │
│                                            │
│ ### Recientes                              │
│ [▢ Lab 2 2026   `/Users/.../2026/lab2`] ✕  │
│ [▢ Lab 3a 2025  `/Users/.../...`     ] ✕  │
│                                            │
│ ### Abrir otra carpeta                     │
│ [path absoluto..............] [Abrir]      │
│                                            │
│ ### Nueva corrección                       │
│ ┌──────────────────────────────────────┐   │
│ │ Carpeta (workdir): [          ]      │   │
│ │ Título del lab:    [          ]      │   │
│ │ Notebook enunciado:[          ]      │   │
│ │ Notebook solución: [          ]      │   │
│ │ Zip de Moodle:     [          ]      │   │
│ │ ⊙ Generar rúbrica con Claude         │   │
│ │ ⊚ Ya tengo una (.rubric.yaml)        │   │
│ │ Path rúbrica: [           ]          │   │
│ │                              [Crear] │   │
│ └──────────────────────────────────────┘   │
└────────────────────────────────────────────┘
```

Notas:

- **Auto-selección del más reciente**: si hay recientes y la sesión
  no tiene workdir, la app entra directo al primero. Para forzar el
  landing, "Cambiar workdir" desde el sidebar.
- **Paths se limpian** con `_clean_path`: quita comillas externas
  (útil para pegar paths con espacios desde el Finder de macOS, que
  los wrappea con comillas) y espacios.
- **Validación de paths absolutos**: workdir y notebooks tienen que
  empezar con `/`. La app no hace `expanduser` heroico — pegás el
  path completo.

## Sidebar

Cuando hay workdir activo:

```
┌──────────────────────────┐
│ Lab 2 2026               │
│ /Users/.../2026/lab2     │
│ [Cambiar workdir       ] │
│                          │
│ ▸ Importar más entregas  │
│ ▸ Config                 │
└──────────────────────────┘
```

- **Cambiar workdir**: limpia `session_state["workdir"]` y los
  query params, vuelve al landing.
- **Importar más entregas (Moodle)**: input + botón. Después del
  intake, muestra `imported`/`warnings`/`skipped` del reporte. Re-
  importar el mismo grupo sobreescribe el `entrega.ipynb` pero
  preserva el `feedback/`. Detalle en [05](05-app-flujo-correccion.md).
- **Config**: muestra los paths actuales del `.corrector/config.json`
  (read-only). Para editarlos hay que abrir el JSON a mano.

## Vista matriz

Layout en `view_matriz()` (`main.py`):

- Título del lab.
- Métricas: cantidad de grupos / cantidad de ítems corregibles.
- Leyenda con los colores: pendiente · sin obs/bien · regular · mal.
- Tabla:

```
                  ┌──────────────────────┐
                  │  G01    G02    G03   │  ← clickeable, abre .ipynb
                  │  85%    abrir  67%   │  ← badge por grupo
                  │ [txt 4] [txt 0] [txt]│  ← escribe grupo_NN.txt
┌─────────────────┼──────────────────────┤
│ ej1 · código  …  │ ok    abrir  bien  │
│   ↳ ej1 · análisis│ ok   reg     mal   │
│ ej2 · código  …  │ bien  pend   ok    │
│ ...              │                     │
└─────────────────┴──────────────────────┘
                ↑
            botón "IA" por fila (genera borradores en batch)
```

Detalles importantes:

- **Click en el header del grupo (`G01`)**: abre el `.ipynb` con la
  app por defecto del SO usando `open_in_os(path)`. En macOS llama a
  `open`, en Linux a `xdg-open`, en Windows a `os.startfile`. Si
  falla, toast de error pero la app sigue.
- **Badge del header**: muestra el porcentaje del grupo cuando todos
  los ítems están corregidos. Si quedan pendientes, muestra
  "N pend" en gris.
- **Score por color**: el badge se pinta verde si pct ≥75%, amarillo
  si ≥50%, rojo en otro caso.
- **Entrega faltante**: si un grupo no tiene `.ipynb` (porque el
  alumno no entregó), la columna entera se pinta roja y todos los
  ítems cuentan como 0pt. Score 0%.
- **Botón `txt (N) ✓`**: el badge ✓ aparece si el archivo
  `grupo_NN.txt` ya existe en disco. Re-clickear sobrescribe.
- **Botón "IA" por fila**: ejecuta el batch IA contra los grupos
  pendientes de esa fila. Saltea los que ya tienen feedback validado.
  Cuando ya hay al menos un borrador en la fila, el botón pasa de
  "IA" a "IA ✓" y se pinta verde. Detalle en
  [07-ia.md](07-ia.md#batch-por-fila).

### Cómo se computa el color de cada celda

Para cada (ítem, grupo):

1. Si el grupo no tiene `.ipynb` → rojo, label "—" (no se puede entrar).
2. Lee `feedback/<item_key>.md`:
   - **No existe** → gris, label "abrir".
   - `<!-- sin-observaciones -->` → verde, label "ok".
   - Con marker `<!-- nivel: bien -->` → verde, label "bien".
   - Con marker `<!-- nivel: regular -->` → amarillo, label "reg".
   - Con marker `<!-- nivel: mal -->` → rojo, label "mal".
   - Sin marker (legacy/sin clasificar) → gris, label "abrir" (queda
     pendiente de clasificar para el score).

El semáforo del score y el del color de celda están sincronizados:
ambos derivan de `read_feedback`.

## Vista corrección

Layout en `view_correccion()`:

```
Lab 2 2026 · ej3 · código          [grupo_05] [← Volver]

[← prev item] [next item →] [↑ prev grupo] [↓ next grupo]

────────────────────────────────────────────────────────
│ Referencia                  │ Entrega — grupo_05      │
│                             │ Mostrando: ej3-code     │
│ ### Ejercicio 3 — Sobel     │ (3/12) [↑] [↓] [Usar]   │
│ ...                         │                         │
│                             │ ```python               │
│ Solución oficial (código):  │ def corr2d(X, K): ...   │
│ ```python                   │ ```                     │
│ def corr2d(X, K): ...       │ Outputs: ...            │
│ ```                         │                         │
────────────────────────────────────────────────────────

### Feedback
Estado actual: [con observación · regular]

[textarea de observación............................]
                                                    │
○ bien (1pt)  ⊙ regular (½pt)  ○ mal (0pt)         │
                                                    │
[Trasladar borrador] [Guardar] [Marcar sin obs] [✕] │

────────────────────────────────────────────────────
**Borrador IA**
[textarea readonly con el draft IA]
[Regenerar borrador IA]
────────────────────────────────────────────────────

[← prev item] [next item →] [↑ prev grupo] [↓ next grupo]
[← Volver a la matriz]
```

### Navegador de celdas (mini-controles `↑ ↓`)

Para qué sirve: a veces el alumno borra la celda con el id estable
`ej3-code` y reescribe la respuesta en otra celda con id random
generado por Jupyter. La app no la encuentra y la columna izquierda
queda vacía. El navegador permite buscarla.

Componentes:

- Label `Mostrando: <id_actual>` con el id de la celda mostrada.
- Contador `(N/M)` con la posición entre las celdas del mismo tipo
  (code o markdown) en el notebook del alumno.
- Botones `↑` (anterior) y `↓` (siguiente).
- Botón `Usar esta para <expected_id>` que aparece cuando lo que se
  está mostrando no es la celda esperada.

Lógica del default:

1. Si hay un override persistido en `cell_overrides.json` para este
   `expected_id`, mostrar la celda apuntada.
2. Si no, mostrar la celda con `expected_id` literal.
3. Si tampoco existe, dejar el panel en estado "celda esperada
   faltante" y deshabilitar el código rendering hasta que el docente
   navegue.

Lógica de `↑/↓`:

- Si la celda actual existe en el notebook, navegar adyacente del
  mismo tipo.
- Si la celda esperada **no existe** en el notebook, anclar la
  navegación en la posición canónica del `expected_id` en el
  notebook de **enunciado** (no en el del alumno). Eso significa que
  `↓` aterriza en el primer candidato cuya posición canónica sea
  ≥ la del expected, y `↑` en el último anterior. En la práctica:
  el docente busca cerca de donde la celda *debería* estar, no en
  los extremos del notebook del alumno.

Click en "Usar esta para `<expected_id>`":

- Persiste el mapeo en `<workdir>/grupo_NN/cell_overrides.json`.
- A partir de ahí toda la app (incluyendo el prompt IA y el feedback
  ya guardado) **resuelve** `expected_id` a la celda actual.
- Si después se quiere quitar, banner azul ofrece "Quitar mapeo".

**Importante**: el notebook del alumno NUNCA se modifica. El
override es un mapeo lógico que vive en el workdir.

### Selector de puntaje (radio)

Tres opciones: `bien (1pt)` / `regular (½pt)` / `mal (0pt)`. Notas:

- **Sin pre-selección por default**: el corrector tiene que elegir.
  Excepciones donde se pre-selecciona:
  - El ítem ya tiene un nivel guardado en disco → se pre-selecciona
    para ver/editar.
  - El ítem está en "sin observaciones" → se pre-selecciona `bien`
    (ya que sin obs implica bien por definición).
- **Auto-save al cambiar**: el cambio del radio guarda el feedback
  inmediatamente.
- **Restricción regular/mal**: estos niveles requieren texto en la
  observación. Si el corrector elige uno con el textarea vacío, toast
  "para regular o mal hace falta una observación" y revierte el
  radio al estado previo.
- **bien sin observación**: permitido. Implica 1pt y no agrega texto
  al `grupo_NN.txt`. Es lo mismo que "marcar sin observaciones" pero
  vía el radio.

### Botones de la sección feedback

| Botón | Cuándo está activo | Qué hace |
|---|---|---|
| **Trasladar a observación** | Cuando hay borrador IA | Copia el texto del borrador al textarea (sustituye lo que tenga). El borrador no se borra. |
| **Guardar observación** | Cuando el textarea tiene texto | Persiste el texto + el nivel actual. Si no se eligió nivel, queda "sin clasificar" (gris en la matriz). |
| **Marcar sin observaciones** | Siempre, salvo si el radio está en regular/mal (toast bloqueante) | Persiste como `<!-- sin-observaciones -->` y pre-selecciona bien. |
| **Borrar** | Cuando el ítem no está pendiente | Elimina el `.md` validado (pero no el draft). Vuelve a pendiente y limpia el radio. |

### Borrador IA (sección al final)

Vive separado de la observación validada. Layout:

- Si el `feedback/<item_key>.draft.md` no existe → caption "Todavía
  no hay borrador IA para este ítem".
- Si existe con marker `<!-- ai-ok -->` → info "La IA considera que
  esta entrega cumple la rúbrica". El docente decide si marca "sin
  observaciones".
- Si tiene texto → textarea readonly (font negro y negrita para que
  se lea bien aunque esté disabled).
- Botón "Generar borrador IA" o "Regenerar borrador IA" según haya o
  no draft. Detalle del prompt en [07-ia.md](07-ia.md).

## Comportamientos sutiles

### Pending writes a session_state

Streamlit no permite escribir `session_state[widget_key]` después de
que el widget se instanció en el run actual. La app necesita empujar
texto al textarea desde handlers que corren en orden de código (después
del widget): "Trasladar a observación", "Marcar sin observaciones",
"Borrar". El patrón usado:

```python
# Handler escribe en pending-<key>:
st.session_state[pending_key] = "nuevo texto"
st.rerun()

# En el run siguiente, ANTES de instanciar el widget:
if pending_key in st.session_state:
    st.session_state[widget_key] = st.session_state.pop(pending_key)
```

El mismo patrón se usa para el radio del puntaje (`pending_level_key`).
`None` como pending limpia la selección.

### Cache de notebooks

`load_notebook` cachea por path. Si el alumno modifica el notebook
mientras la app está abierta (raro), hay que reiniciar Streamlit. En
la práctica las entregas de Moodle son inmutables.

### Estilo de los botones

Streamlit 1.56 pinta los botones secondary/tertiary blancos por
defecto, lo que rompe la paleta crema. La app inyecta CSS al inicio
para tonarlos a `#EFEAE0` (secondary) y `#F2EFE7` (tertiary). Las
filas con botón "IA ✓" usan `.st-key-batch-ia-<row>` para pintarlo
verde — este selector apunta a la clase que Streamlit ≥1.36 agrega
al contenedor del widget según su `key=`.

### Open en SO

`open_in_os(path)` corre `subprocess.run(["open", str(path)])` en
macOS, `xdg-open` en Linux, `os.startfile` en Windows. Si la
plataforma no es ninguna, devuelve `False` y la app muestra toast.

Esto **no funciona** si la app se mueve a un servidor remoto: el
subprocess se ejecuta en la máquina del servidor, no en la del
docente. Es una decisión consciente: la app está pensada para correr
local. Si en el futuro alguien la pone detrás de un proxy, hay que
deshabilitar estos botones (o reemplazarlos por links a archivos
servidos por Streamlit).
