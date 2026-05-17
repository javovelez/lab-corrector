# 05 — Flujo de corrección end-to-end

Este documento recorre paso por paso una tanda de corrección, desde que
bajás el zip de Moodle hasta que subís los txt finales. La UI detallada
(matriz, navegador de celdas, niveles, etc.) está en
[06-app-ui-detallada.md](06-app-ui-detallada.md).

## Pre-requisitos

- App instalada (venv en `app/.venv/`, dependencias OK — ver
  [11-instalacion.md](11-instalacion.md)).
- Notebook de enunciado y de solución del lab (en `_TPS/Laboratorios/`
  y `_TPS/Soluciones/`).
- Rúbrica del lab (en `_TPS/rubricas/Laboratorio_X.rubric.yaml`) o,
  alternativamente, dejar que la app la auto-genere durante "nueva
  corrección".
- Zip de Moodle con las entregas. Estructura habitual:
  ```
  entregas_lab3.zip
  └── Laboratorio 3/
      ├── Grupo 01_123456_assignsubmission_file/
      │   └── grupo01_lab3.ipynb
      ├── Grupo 02_123457_assignsubmission_file/
      │   └── entrega_grupo2.ipynb
      └── ...
  ```

## Paso 1 — Levantar la app

```bash
app/.venv/bin/streamlit run app/main.py
```

Streamlit abre `http://localhost:8501` y lanza el browser. La primera
vez muestra el **landing**. Si ya hay workdirs recientes en
`~/.lab_corrector/recent.json`, la app **auto-selecciona el más
reciente** y entra directo a la corrección — para ir al landing,
clickear "Cambiar workdir" en el sidebar.

## Paso 2 — Crear o abrir un workdir

### Si es una tanda nueva (form "Nueva corrección")

Campos del form:

- **Carpeta (workdir)**: path absoluto donde va a vivir todo (zip,
  entregas, feedback, txt). Si no existe, la app la crea.
- **Título del lab**: lo que aparece en la barra de título de la
  matriz. Defaultea al nombre de la carpeta si lo dejás vacío.
- **Notebook de enunciado**: path absoluto al `.ipynb` del enunciado
  oficial.
- **Notebook de solución**: path absoluto al `.ipynb` de solución.
- **Zip de Moodle (opcional)**: si lo das, la app importa las
  entregas en el mismo paso. Si no, podés importarlo después desde el
  expander "Importar más entregas" del sidebar.
- **Rúbrica**: dos opciones:
  - "Generar automáticamente desde la solución (usa Claude)" → llama
    a `generate_rubrica` con barra de progreso. Tarda ~1-2 min para
    un lab de 10-15 ítems. La rúbrica queda en
    `<workdir>/.corrector/rubrica.yaml`.
  - "Ya tengo una (.rubric.yaml)" → input para pegar el path absoluto.

Submit → la app:
1. Crea la carpeta si no existe.
2. Genera o copia la rúbrica al destino.
3. Escribe `<workdir>/.corrector/config.json` con los paths absolutos.
4. (Si pasaste zip) corre `intake_zip`.
5. Inscribe el workdir en `recent.json`.
6. Carga la sesión y refresca a la matriz.

### Si es una tanda existente

Tres opciones en el landing:

- **Recientes**: botón por workdir (label = nombre de la carpeta +
  path). Click → carga.
- **Abrir otra carpeta**: input + botón "Abrir". Si la carpeta no
  tiene `.corrector/config.json`, el landing avisa que use "Nueva
  corrección".
- **Quitar de recientes**: botón al lado de cada reciente.

## Paso 3 — Importar el zip (si no se hizo en el paso 2)

Sidebar → expander "Importar más entregas (Moodle)" → input "Ruta al
zip" → botón "Importar". El intake:

1. Descomprime el zip a un temp del SO.
2. Lista subdirectorios. Si los nombres top-level matchean
   `^Grupo \d+_\d+_assignsubmission_file$`, los toma directo. Si hay
   un solo directorio en la raíz, baja un nivel y reintenta (Moodle a
   veces empaqueta todo dentro de una carpeta con el nombre de la
   entrega).
3. Para cada carpeta de grupo:
   - Extrae `N` del prefijo `Grupo N_...`.
   - Renombra a `grupo_NN` (zero-padded a 2 dígitos).
   - Copia el primer `.ipynb` adentro de la carpeta a
     `<workdir>/grupo_NN/entrega.ipynb`. **El nombre original del
     archivo se descarta.**
   - Si hay más de un `.ipynb`, advierte y toma el primero.
   - Si no hay ninguno, agrega un skip al reporte.
4. **Preserva** lo que ya hubiera en `<workdir>/grupo_NN/feedback/`.
   El re-import solo sobrescribe el `entrega.ipynb`.
5. Devuelve un `IntakeReport` con tres listas:
   - `imported`: nombres de grupos importados (`grupo_03`, etc.).
   - `skipped`: tuplas `(nombre_carpeta, motivo)`.
   - `warnings`: tuplas `(grupo, mensaje)` — p. ej. múltiples ipynb.

La UI muestra los tres listados después del intake.

### Cuándo re-importar

Si un grupo entrega tarde o entrega de nuevo, bajás un zip nuevo y lo
importás. La app sobrescribe `entrega.ipynb` pero **no toca el
feedback ya guardado**, así que no perdés trabajo.

## Paso 4 — Corregir desde la matriz

La matriz es la pantalla central (`view_matriz` en `main.py`):

- **Filas**: los ítems corregibles del lab, en el orden que define la
  rúbrica. Cada ejercicio aporta 1 o más filas (1 por code, 1 por
  analysis). Las filas de analysis llevan un sangrado visual debajo
  del code para señalar la jerarquía.
- **Columnas**: los grupos importados, ordenados alfabéticamente.
- **Cada celda**: un cuadradito coloreado según el estado del
  feedback de ese (ítem × grupo):
  - **gris** = pendiente (sin corregir todavía).
  - **verde** = "sin observaciones" o "con observación · bien" (1pt).
  - **amarillo** = "con observación · regular" (½pt).
  - **rojo** = "con observación · mal" (0pt) o entrega faltante.
- **Header de cada columna** (`G01`, `G02`, ...): clickeable, abre el
  `.ipynb` del grupo con la app por defecto del SO (Jupyter Lab, VS
  Code, etc.). Debajo va un badge con el porcentaje del grupo (color
  según rango: ≥75% verde, ≥50% amarillo, resto rojo) y un botón
  `txt (N)` para escribir el `grupo_NN.txt` con N = cantidad de
  observaciones reales.
- **Botón "IA" por fila**: genera borradores en batch para todos los
  grupos pendientes de ese ítem, en una sola llamada a Claude. No
  pisa feedback ya validado. El botón se pinta verde si la fila ya
  tiene al menos un borrador. Detalle en [07-ia.md](07-ia.md).

Click en una celda gris/verde/amarilla/roja → vista corrección de ese
ítem para ese grupo.

## Paso 5 — Vista corrección

Layout (`view_correccion`):

- **Breadcrumb arriba**: título del lab · ejercicio · tipo (código o
  análisis) · botón con el nombre del grupo (clickeable, abre el
  notebook) · botón "← Volver a la matriz".
- **Navegación prev/next**: 4 botones (item anterior, item siguiente,
  grupo anterior, grupo siguiente). Duplicados arriba y abajo de la
  vista para no tener que scrollear.
- **Columna izquierda** = referencia:
  - Para ítems code: el enunciado del ejercicio (renderizado como
    markdown) + el código de la solución oficial (en bloque
    `python`).
  - Para ítems analysis: la pregunta de análisis del enunciado (o el
    enunciado del ejercicio si el item no tiene `pregunta_cell`) + la
    respuesta oficial (markdown).
- **Columna derecha** = entrega del grupo:
  - Mini-navegador `↑/↓` con un label tipo "Mostrando: `ej3-code`
    (3/12)" — más detalle en [06-app-ui-detallada.md](06-app-ui-detallada.md).
  - El código (con outputs renderizados: text en `st.code`, imágenes
    con `st.image` y fullscreen built-in, errores con `st.error`) o
    la respuesta del alumno (markdown).
- **Sección de feedback abajo**:
  - Indicador del estado actual ("pendiente" / "sin observaciones" /
    "con observación · bien" / etc.) con su color.
  - Textarea "Observación al alumno (se incluye tal cual en el txt)".
  - Selector de puntaje (radio): `bien (1pt)` / `regular (½pt)` /
    `mal (0pt)`.
  - Botones: **Trasladar a observación** (copia el borrador IA al
    textarea), **Guardar observación**, **Marcar sin observaciones**,
    **Borrar** (vuelve al estado pendiente).
  - Auto-save del puntaje al cambiar el radio: bien sin texto se
    permite (es 1pt sin observación al alumno); regular y mal con
    textarea vacío disparan un toast bloqueante "para regular o mal
    hace falta una observación".
- **Sección "Borrador IA"**:
  - Si la IA marcó el ítem como correcto (devolvió `OK`), muestra un
    hint sugiriendo "marcá sin observaciones".
  - Si hay borrador real, lo muestra como textarea readonly con el
    texto en negrita.
  - Botón "Generar borrador IA" o "Regenerar borrador IA". El
    borrador queda en `feedback/<item_key>.draft.md` (independiente
    del feedback validado).

## Paso 6 — Generar el txt del grupo

La app **regenera el `grupo_NN.txt` automáticamente** cuando todos
los ítems del grupo están corregidos. No hace falta tocar nada: en
cada render de la matriz, si el grupo está completo y el contenido
en disco difiere del que produciría el estado actual de `feedback/`,
el archivo se reescribe. La lógica de construcción:

1. Recorre los ítems en orden.
2. Para cada uno, lee el `feedback/<item_key>.md`.
3. Si el archivo es `<!-- sin-observaciones -->`, lo salta (el alumno
   no necesita feedback en ese ítem).
4. Si es pendiente (no existe), también lo salta.
5. Si tiene texto real, lo concatena con un header
   `Ej <N> (<código|análisis>):`.
6. Junta todos los bloques con doble salto de línea.
7. Escribe el resultado a `<workdir>/grupo_NN/grupo_NN.txt`.

El botón `txt (N)` de la matriz pasa a tener un solo rol: **copiar
el contenido del txt al portapapeles** del sistema (vía `pbcopy` en
macOS, `xclip` en Linux, `clip` en Windows). Cuando se hace click,
toast confirma "grupo_NN.txt copiado al portapapeles". El badge ✓
aparece cuando el archivo existe en disco.

**Si quedan ítems pendientes**, el botón se deshabilita y no se
genera ni se actualiza el txt — así no se filtra al portapapeles
una devolución incompleta. El tooltip indica cuántos ítems faltan.
**Si el grupo está completo pero todo es "sin observaciones"**, el
txt queda vacío y el botón también se deshabilita (no hay nada que
copiar). El score sigue computándose para el header de la columna.

## Paso 7 — Subir los txt a Moodle

Cada `<workdir>/grupo_NN/grupo_NN.txt` tiene la devolución para ese
grupo, listo para pegar en Moodle. En la práctica, el flujo es:

1. Click en `txt (N) ✓` en la matriz → contenido copiado al
   portapapeles.
2. Cambiar al tab del browser con Moodle, pegar (Cmd+V) en la caja
   de feedback del grupo, guardar.
3. Volver a la app, siguiente grupo.

La app no se ocupa de Moodle (no tiene API ni credenciales) — el
upload lo hace el docente a mano.

## Re-correcciones

Si después de un primer pase el docente quiere ajustar:

- **Cambiar la observación**: entrar a la vista corrección del ítem
  → editar el textarea → "Guardar observación". Sobrescribe el `.md`.
- **Cambiar el puntaje sin tocar el texto**: cambiar el radio → la
  app auto-guarda.
- **Volver a pendiente**: botón "Borrar" → elimina el `.md` (el
  draft `.md` no se borra; queda como referencia para regenerar).
  Al volver a pendiente, el `grupo_NN.txt` queda con el contenido
  viejo (la auto-regeneración solo corre cuando el grupo está
  completo) — al re-completar el grupo, el txt se actualiza solo
  en el siguiente render.
- **Re-copiar al portapapeles**: clickear el botón `txt (N) ✓` de
  nuevo después de los ajustes.

## Errores comunes durante la corrección

Cubiertos en [10-troubleshooting.md](10-troubleshooting.md):

- Una entrega no tiene la celda con el id que la rúbrica espera —
  usar el navegador `↑/↓` para encontrarla y "Usar esta para `<id>`".
- Una entrega es directamente inválida (no es ipynb, está corrupta) —
  la app muestra error en la columna y el ítem cuenta como 0pt.
- La IA devuelve un texto raro — toast con error, se puede
  regenerar.
