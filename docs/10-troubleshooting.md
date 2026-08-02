# 10 — Troubleshooting

Catálogo de los problemas más comunes que aparecen al usar el
framework, con la causa, el síntoma y la solución concreta. Si lo que
te pasa no está acá, mirá la página del componente afectado
(autoría → [02](../plugins/lab-notebook/skills/lab-notebook/reference/lab-md.md), rúbrica → [03](03-rubricas.md),
app → [05](05-app-flujo-correccion.md) – [07](07-ia.md)).

## Lab 2 — celdas pegadas

**Síntoma:** la app abre el Lab 2 y muestra las preguntas de análisis
en lugar del lugar para responder, o cualquier item analysis aparece
con `answer_cell` apuntando a una celda que tiene mezclado el texto
de la pregunta con el placeholder `*(Escribí tu respuesta acá)*`.

**Causa:** Lab 2 se escribió antes de que el framework separara la
pregunta de la respuesta del alumno. En las versiones originales,
cada `ejN-pregunta` contenía:

```markdown
**Pregunta de análisis:**
¿...?

*(Escribí tu respuesta acá)*
```

en una sola celda, con id `ejN-pregunta`. La app espera dos celdas:
`ejN-pregunta` con el texto de la pregunta y `ejN-respuesta` con la
respuesta del alumno.

**Solución:** correr el parche
[`lab2_split_pregunta.py`](../plugins/lab-notebook/scripts/lab2_split_pregunta.py).
El script:

1. Toma un notebook in place.
2. Recorre las celdas markdown con id `ejN-pregunta`.
3. Si ya existe `ejN-respuesta` para ese N → saltea.
4. Si hay placeholder en la celda → divide por placeholder: lo de
   antes queda como `pregunta`, lo del placeholder en adelante queda
   como `respuesta`.
5. Si hay un notebook de **referencia** (con la pregunta canónica) y
   la celda arranca con esa pregunta → divide por la pregunta exacta.
6. Si nada de eso aplica → asume que el alumno borró la pregunta y
   escribió solo la respuesta. Usa la pregunta canónica del
   reference como `pregunta` y lo que estaba en la celda como
   `respuesta`.
7. Crea `.bak` antes de sobrescribir. **Idempotente.**

Uso:

```bash
# Para los notebooks de Lab 2 oficiales (one-time):
python plugins/lab-notebook/scripts/lab2_split_pregunta.py _TPS/Laboratorios/Laboratorio_2.ipynb
python plugins/lab-notebook/scripts/lab2_split_pregunta.py _TPS/Soluciones/Laboratorio_2_Solucion.ipynb

# Para una entrega de un alumno (con reference para split exacto):
python plugins/lab-notebook/scripts/lab2_split_pregunta.py \
    <workdir>/grupo_05/entrega.ipynb \
    --reference _TPS/Laboratorios/Laboratorio_2.ipynb
```

**Estado actual:** el parche ya se aplicó a los dos notebooks
oficiales y a las 13 entregas del Lab 2 2026 (`<workdir>/grupo_NN/entrega.ipynb`
con `.bak` al lado). Para Lab 3a, 3b y futuros (que se autoraron con
`lab_build.py` desde el principio), este parche **no hace falta**.

## Una entrega no tiene la celda con el `cell_id` esperado

**Síntoma:** al entrar a la vista corrección, la columna derecha
("Entrega — grupo_NN") muestra un warning "El grupo no tiene la
celda `ejN-code` en su entrega."

**Causa:** el alumno borró la celda con id estable (intencionalmente
o por accidente — Jupyter a veces genera ids hexadecimales nuevos al
copy/paste) y reescribió la respuesta en otra.

**Solución 1 (recomendada): usar el navegador `↑/↓`**

1. Click en `↑` o `↓` para recorrer las celdas del mismo tipo en el
   notebook del alumno. La app ancla la navegación en la posición
   canónica que la celda tiene en el enunciado, así aterrizás cerca
   de donde la celda *debería* estar.
2. Cuando encuentres la respuesta del alumno, click en
   "Usar esta para `<expected_id>`".
3. La app guarda un override en `<workdir>/grupo_NN/cell_overrides.json`
   y a partir de ahí resuelve `expected_id` a la celda actual. El
   notebook del alumno **no se modifica**. Tampoco se modifica el
   prompt de la IA — el override se aplica en `_grupo_ai_payload`.

**Solución 2: editar el override a mano**

```bash
echo '{"ej3-code": "8a3f2e1b-..."}' > <workdir>/grupo_03/cell_overrides.json
```

Refrescar la app.

**Solución 3: reparar el notebook del alumno** (no recomendada)

Editar el `.ipynb` a mano para renombrar el id de la celda. La app
nunca toca los notebooks de los alumnos como decisión de diseño,
pero el docente sí puede hacerlo si quiere — los `.bak` quedan en el
zip original por si hay que revertir.

## Un grupo no tiene `.ipynb` válido

**Síntoma:** la columna del grupo aparece roja con label "sin ipynb"
y todos los ítems en 0pt sin posibilidad de entrar.

**Causa:** una de:

- El grupo no entregó (no hay carpeta o está vacía).
- El grupo entregó algo que no es un `.ipynb` (zip, pdf, txt). El
  intake los descartó como "skipped".
- El grupo tiene **dos o más** `.ipynb` y la app no sabe cuál tomar.

**Solución:**

- Si no entregaron: dejar como está. La app trata "entrega faltante"
  como 0pt en todos los ítems, lo que es la regla razonable. Score
  del grupo va a ser 0%.
- Si entregaron pero el archivo no es ipynb: revisar el zip original.
  Si hay un `.ipynb` adentro, copiarlo a mano:

  ```bash
  cp /ruta/al/archivo_correcto.ipynb <workdir>/grupo_NN/entrega.ipynb
  ```

  Refrescar la app.
- Si hay múltiples ipynb: el intake ya emitió un warning. Borrar los
  sobrantes a mano:

  ```bash
  ls <workdir>/grupo_NN/*.ipynb        # ver cuáles hay
  rm <workdir>/grupo_NN/<sobrante>.ipynb
  mv <workdir>/grupo_NN/<correcto>.ipynb <workdir>/grupo_NN/entrega.ipynb
  ```

## La rúbrica autogen sale mala

**Síntoma:** después de "Generar automáticamente desde la solución",
los `expected` son genéricos o los `common_errors` son tips en lugar
de errores reales.

**Causa probable:** la solución oficial no estaba ejecutada (sin
outputs guardados) o el enunciado del ejercicio es muy corto y no le
dio a Claude suficiente contexto.

**Solución:**

1. Abrí `_TPS/Soluciones/Laboratorio_X_Solucion.ipynb` en Jupyter,
   "Restart Kernel and Run All", guardá.
2. Re-corré:
   ```
   app/.venv/bin/python app/rubric_build.py X
   ```
3. Si sigue saliendo flojo: editá la rúbrica a mano. Es un YAML
   editable. Lo importante es:
   - `expected` corto pero específico (nombrá funciones reales).
   - `common_errors` que sean errores que cambien el resultado, no
     tips.
   - El crítico (que rompe el ítem) al inicio con `"CRÍTICO: ..."`.

## La IA devuelve algo raro

### Error del SDK

**Síntoma:** toast rojo "No pude generar el borrador: <error>".

**Causa:** la sesión local de Claude Code no respondió.

**Solución:**

- Probar de nuevo (regenerar). A veces es transitorio.
- Si es persistente: chequear que `claude-agent-sdk` esté instalado:
  ```
  app/.venv/bin/python -c "import claude_agent_sdk; print(claude_agent_sdk.__version__)"
  ```
- Verificar que tenés sesión activa de Claude Code en la máquina.

### El batch devuelve JSON inválido

**Síntoma:** toast rojo "La IA devolvió una respuesta no parseable".

**Causa:** Claude se desvió del formato JSON exigido. Pasa más con
prompts largos.

**Solución:**

- Reintentar. El batch es estatelesss — el segundo intento parte de
  cero.
- Si pasa repetidamente, generar borradores uno por uno con el botón
  "IA" individual de cada celda en lugar del batch por fila.

### Faltan grupos en la respuesta del batch

**Síntoma:** toast amarillo "La IA no devolvió borrador para:
grupo_07, grupo_11".

**Causa:** Claude se salteó esos grupos en el JSON (lo más común con
batches de 15+ grupos).

**Solución:**

- Reintentar el batch. Como salta los grupos ya validados, va a
  procesar solo los faltantes — quedan menos en la pasada y es menos
  probable que se saltee.
- Alternativamente, entrar a esos grupos uno por uno y darle "Generar
  borrador IA" individual.

## El score del grupo no se actualiza

**Síntoma:** después de guardar varias observaciones, el badge de la
columna sigue mostrando "N pend" en gris.

**Causa:** alguna observación quedó **sin clasificar** — texto sin
marker de nivel. La app la cuenta como pendiente para el score.

**Solución:** entrar al ítem afectado, mirar el indicador "Estado
actual": si dice "con observación · sin clasificar", elegir un
nivel en el radio (bien/regular/mal). El auto-save ya lo persiste.

Para identificar cuáles están sin clasificar más rápido: en la
matriz, las celdas grises con label "abrir" pero sin estar en
pendiente puro (ya las visitaste) son las que faltan clasificar.

## Streamlit no levanta

**Síntoma:** `app/.venv/bin/streamlit run app/main.py` falla con
`ImportError` o `ModuleNotFoundError`.

**Causa:** venv no creado o dependencias desactualizadas.

**Solución:**

```bash
python3 -m venv app/.venv
app/.venv/bin/pip install -U pip
app/.venv/bin/pip install -r app/requirements.txt
```

Si claude-agent-sdk no se instala (necesita Node.js para el
backend de Claude Code), verificar que tenés Claude Code instalado
en la máquina.

## La app abre pero no carga ningún workdir

**Síntoma:** llega al landing pero no aparece el form, o el form
no responde.

**Causa probable:** corruption en `~/.lab_corrector/recent.json`.

**Solución:**

```bash
rm ~/.lab_corrector/recent.json
```

La app crea uno nuevo en el próximo `touch`. No se pierde nada
(los workdirs siguen existiendo en disco; solo se limpia el atajo
"recientes").

## Click en `G01` no abre el notebook

**Síntoma:** click en el header del grupo y no pasa nada (o aparece
un toast "no pude abrir").

**Causa:** `open` / `xdg-open` / `os.startfile` no encontró una
aplicación asociada al `.ipynb`.

**Solución:**

- macOS: instalar VS Code o JupyterLab y configurar como app por
  defecto para `.ipynb`.
- Linux: `xdg-mime default` para asociar `.ipynb` con tu editor.
- Como atajo: la ruta del notebook está en
  `<workdir>/grupo_NN/entrega.ipynb`; abrilo a mano.

## La rúbrica se queda en v1 después de "guardar"

**Síntoma:** la app "guardó" la rúbrica y al inspeccionarla en disco
sigue en formato v1 (sin `items`).

**Causa real:** la app **nunca** guarda la rúbrica desde la UI. El
único `save_rubrica` se ejecuta al crear un workdir nuevo con auto-
generación. Las rúbricas de Lab 1a/1b/1c/2/3a/3b están en v1 en
disco y se quedan así, normalizándose a v2 en memoria al cargarse.

**Solución:** si querés migrar a v2 en disco, abrir el YAML, mover los
campos `code_cell`/`pregunta_cell`/`answer_cell` a una lista `items`
con `kind: code` / `kind: analysis`, y guardar. La app va a leer
ambas formas indefinidamente.

## Cambios al `.lab.md` no se reflejan en el `.ipynb`

**Síntoma:** editás el `.lab.md` pero el `.ipynb` no cambia.

**Causa:** olvidaste recompilar.

**Solución:**

```
python plugins/lab-notebook/scripts/lab_build.py _TPS/sources/Laboratorio_X.lab.md
```

Si modificaste el id de una celda y la app de corrección estaba
abierta sobre un workdir que la usa, va a aparecer "celda esperada
faltante" — la rúbrica vieja apunta al id viejo. Regenerar la
rúbrica:

```
app/.venv/bin/python app/rubric_build.py X
```

## La app dice "rúbrica no existe" pero el archivo está ahí

**Síntoma:** error rojo "no pude leer la rúbrica: ..." y al chequear
en disco el archivo está.

**Causa probable:** el path en `.corrector/config.json` apunta a un
lugar viejo (la rúbrica se movió o se renombró).

**Solución:** editar `<workdir>/.corrector/config.json` con el path
actualizado. Refrescar la app.

## Cómo "resetear" un grupo (volver a corregirlo desde cero)

```bash
rm -rf <workdir>/grupo_NN/feedback/
rm -f <workdir>/grupo_NN/cell_overrides.json
rm -f <workdir>/grupo_NN/grupo_NN.txt
```

Refrescar la app. La columna vuelve a aparecer toda en gris/pendiente.
El `entrega.ipynb` se preserva.

Si querés borrar todo el grupo (raro):

```bash
rm -rf <workdir>/grupo_NN/
```

## Cómo "resetear" un workdir entero

Borrá la carpeta. El registry global se va a auto-limpiar
(`list_recents` filtra paths inexistentes). Si te molesta verlo en
"Recientes" hasta el próximo refresh, click en "quitar" al lado del
item.
