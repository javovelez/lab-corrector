# Guía de estilo — notebooks de laboratorio

Cubre el aspecto **visual y textual** de los notebooks. Es transversal a
todas las materias. Lo específico de una materia (nombre de asignatura,
bloques temáticos, librerías, inventario de labs) va en el `CLAUDE.md`
del repo de esa materia, no acá.

Para el contrato técnico ver [cell-ids.md](cell-ids.md) y
[lab-md.md](lab-md.md).

---

## Estructura del notebook

### 1. Encabezado visual

Primera celda, markdown, solo la imagen institucional de la materia, sin
texto adicional. Cell id: `header`.

### 2. Título y metadatos

Segunda celda, markdown. Cell id: `titulo`.

```markdown
# Laboratorio n° X. Parte Y: Título del laboratorio

**Asignatura:** Nombre de la materia
**Bloque:** N — Nombre del bloque

---

## Introducción

[Párrafo introductorio que contextualiza el tema.]

[Lista de objetivos del trabajo:]

- Objetivo 1
- Objetivo 2

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para resolver cada ejercicio, consultá el material teórico de la Clase N.
```

Observaciones:

- El título usa `n°` (ordinal masculino), no "nro." ni "Nro.".
- Los metadatos van como `**Asignatura:**` y `**Bloque:**`, dos puntos
  adentro de la negrita, después un espacio y el valor.
- El separador entre número de bloque y nombre es raya larga (`—`), no
  guion (`-`) ni dos guiones (`--`).
- Restricciones particulares del lab (por ejemplo "no está permitido usar
  bucles `for` o `while` salvo que el enunciado lo indique") se agregan
  como ítem en negrita al final de las instrucciones, y solo cuando
  aplican.

### 3. Reglas de entrega

Celda markdown fija, mismo texto en todos los laboratorios. Cell id:
`reglas`.

```markdown
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de
actividad, que son las que aparecen con el comentario `# Tu código aquí` o
el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados,
explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la
corrección se hace celda por celda de manera automática y modificar lo que no
corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia
aparte y revertí los cambios antes de entregar.
```

Esta celda no es cosmética: es la que hace viable la corrección por
`cell_id`. Va en todos los labs.

### 4. Imports

Celda de código con los imports necesarios, ya ejecutable. Cell id:
`imports`. En labs extensos, el setup completo va en una celda `setup-*`
que el alumno recibe lista para correr.

### 5. Secciones temáticas

Cada sección abre con una celda markdown propia. Cell ids `secA`, `secB`,
`secC`, ...

```markdown
---
## Sección X: Nombre de la sección
```

Si la sección necesita una introducción conceptual, va en una celda
markdown separada inmediatamente después del encabezado.

### 6. Bloque de ejercicio

El patrón usual son cuatro celdas consecutivas.

**Enunciado** (markdown, `ejN-enunciado`):

```markdown
### Ejercicio N — Título descriptivo del ejercicio

**Objetivo:** Una o dos oraciones que describen qué habilidad se practica.

**Enunciado:**

1. Primer paso.
2. Segundo paso.

> **Pista:** Texto de ayuda.
```

- Título con raya larga (`—`), no guion.
- `**Objetivo:**` es obligatorio y va en línea propia.
- `**Enunciado:**` contiene los pasos numerados.
- Las pistas van en blockquote (`>`). Si son varias o extensas, lista
  adentro del blockquote.
- La pista nunca revela la solución: orienta hacia el concepto o la
  función.

**Código** (code, `ejN-code`):

```python
# Tu código aquí
```

El placeholder es exactamente `# Tu código aquí`. Si la celda trae
andamiaje preescrito, el placeholder aparece en el lugar exacto donde el
alumno tiene que escribir.

**Pregunta de análisis** (markdown, `ejN-pregunta`):

```markdown
**Pregunta de análisis:**

¿La pregunta conceptual relacionada con el ejercicio?
```

**Respuesta** (markdown, `ejN-respuesta`):

```markdown
*(Escribí tu respuesta acá)*
```

El placeholder es exactamente `*(Escribí tu respuesta acá)*`, siempre en
cursiva. Pregunta y respuesta van en **celdas separadas** — si van
pegadas en una sola celda, la app no puede aislar lo que escribió el
alumno.

**Celda de test** (code, opcional): verifica que la arquitectura o una
parte de ella esté bien implementada. Solo cuando aporta.

### 7. Checklist de entrega

Antes del cierre, celda markdown. Cell id: `checklist`.

```markdown
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] Los valores numéricos que imprimo son razonables (no hay infinitos, ni `NaN`, ni errores de unidades).
- [ ] Todos los gráficos tienen título, etiquetas en los ejes y grilla.
- [ ] No modifiqué ninguna celda fuera de las de actividad.
```

Los ítems se ajustan por laboratorio (por ejemplo agregar "Los tests
pasan sin errores" si el lab tiene celdas de test).

### 8. Cierre

Celda markdown final. Cell id: `footer`.

```markdown
---
## ¡Listo!

[Mensaje de cierre. Menciona qué se practicó y anticipa el próximo laboratorio.]
```

---

## Convenciones de escritura

### Idioma y registro

- **Español rioplatense con voseo**: "creá", "usá", "imprimí",
  "completá", "respondé", "observá".
- Los términos técnicos establecidos en inglés se mantienen en inglés y
  van en cursiva la primera vez que aparecen en una sección:
  *broadcasting*, *forward pass*, *overfitting*.
- Nombres de funciones, métodos, atributos y parámetros siempre en código
  inline: `.reshape()`, `requires_grad=True`, `.backward()`.
- **No se usan emoticones** en ningún contexto.
- Evitar el spanglish conjugado ("reshapeá", "batcheado", "printeá"): o
  el término técnico en inglés en cursiva, o castellano técnico estándar.
- El tono es técnico pero didáctico: explica el concepto y el porqué, no
  solo el procedimiento.

### Énfasis y formato inline

| Elemento | Formato |
|---|---|
| Términos técnicos clave (primera mención o énfasis) | `**negrita**` |
| Nombres de funciones, métodos, atributos | `` `código inline` `` |
| Términos en inglés de uso técnico | `*cursiva*` |
| Fórmulas matemáticas en línea | `$fórmula$` |
| Fórmulas matemáticas en bloque | `$$fórmula$$` |

### Separadores

- Las secciones principales abren con `---` en celda markdown propia.
- Adentro de una celda, `---` separa conceptualmente bloques de
  contenido.

### Tablas

Se usan para información tabular del dataset, reglas o clasificaciones, y
notas de corrección (solo en soluciones). Encabezados breves, sin punto
final.

---

## Convenciones de código en celdas de solución

### Encabezados de bloque internos

Los bloques lógicos adentro de una celda de código se separan con
comentarios de línea ancha:

```python
# ─── Descripción del bloque ───────────────────────────────────────────────────
```

El carácter es `─` (U+2500, BOX DRAWINGS LIGHT HORIZONTAL), no un guion
común. La línea llega hasta aproximadamente la columna 80.

### Comentarios

- Explican el **por qué**, no solo el qué.
- Cada decisión de diseño no evidente lleva un comentario que la
  justifica.
- Van en español.

```python
# torch.rand_like() es más seguro que torch.rand(2, 3, 4) a mano: si cambiamos
# la forma del tensor base, este se actualiza automáticamente.
aleatorio = torch.rand_like(ceros)
```

### Reproducibilidad

Cuando una celda genera valores aleatorios que se mencionan en los
comentarios o en el enunciado, se fija la semilla.

### Docstrings

Las funciones definidas en el notebook —sobre todo las de setup que el
alumno recibe preescritas— llevan docstring en español con parámetros y
retorno.

### Verificaciones

Las soluciones incluyen `print()` explícitos que confirman el resultado
esperado. Estos outputs son los que después alimentan `graded_outputs` en
la rúbrica, así que conviene que sean informativos.

### Errores esperados

Cuando el ejercicio pide provocar un error a propósito para observarlo,
se captura con `try/except` y se imprime el mensaje.

---

## Diferencias del archivo de solución

1. El título agrega `-- SOLUCION` al final. Lo hace el compilador
   automáticamente sobre la celda `header`.
2. Las celdas `# Tu código aquí` se reemplazan por el código completo y
   comentado.
3. Las celdas de pregunta mantienen el texto del enunciado; la celda de
   respuesta lleva la respuesta oficial, encabezada por
   `**Respuesta a la pregunta de análisis:**`.
4. Opcionalmente, al final, una sección `## Notas de corrección` con una
   tabla de conceptos clave y errores frecuentes por ejercicio. Cell id:
   `notas-correccion`. Es material de apoyo para el docente; la rúbrica
   YAML la genera la app aparte.

---

## Checklist antes de publicar

- [ ] La primera celda es solo la imagen de encabezado.
- [ ] La celda de reglas de entrega está presente.
- [ ] El título sigue el patrón `# Laboratorio n° X. Parte Y: Título`.
- [ ] Los metadatos de asignatura y bloque están presentes.
- [ ] Todas las secciones abren con `---` en celda propia.
- [ ] Cada ejercicio tiene sus celdas de enunciado, código y análisis.
- [ ] Los placeholders son exactamente `# Tu código aquí` y
      `*(Escribí tu respuesta acá)*`.
- [ ] Pregunta y respuesta están en celdas separadas.
- [ ] Las pistas van en blockquote con `**Pista:**` en negrita.
- [ ] No hay emoticones en ninguna celda.
- [ ] El lenguaje usa voseo de forma consistente.
- [ ] `lab_validate.py` da cero errores sobre el enunciado.
- [ ] La solución está ejecutada y guardada con sus outputs.
- [ ] La celda de checklist de entrega está antes del cierre.
- [ ] La celda de cierre anticipa el siguiente laboratorio, si
      corresponde.
