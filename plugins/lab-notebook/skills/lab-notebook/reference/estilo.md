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
actividad, que son las que aparecen con el comentario `# Tu código aquí` o el
texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas —enunciados,
explicaciones, ejemplos provistos y encabezado— **no se tocan**.

La corrección se hace celda por celda: cada respuesta se busca en la celda
donde el enunciado la pide. Si escribís en otro lado, o si movés, renombrás o
borrás celdas del enunciado, esa parte de tu entrega queda sin poder
corregirse.

Si querés probar algo suelto, hacelo en la misma celda de actividad o en una
celda nueva que agregues, y borrala antes de entregar.
```

Esta celda no es cosmética: es la que hace viable la corrección por
`cell_id`. Va en todos los labs.

Dos cosas que el texto evita a propósito:

- **No dice que la corrección sea "automática".** Corrige el docente, con la
  app como herramienta. Lo que hay que transmitirle al alumno es la
  consecuencia práctica —si no usa las celdas previstas, su respuesta no se
  puede corregir—, no cómo funciona la app por dentro.
- **No le pide trabajar sobre una copia del notebook.** Es mucho pedir y
  nadie lo hace. Probar en la misma celda, o en una nueva que después se
  borra, alcanza.

### 4. Imports y celdas de setup

Celda de código con los imports necesarios, ya ejecutable. Cell id:
`imports`. En labs extensos, el setup completo va en una o varias celdas
`setup-*` que el alumno recibe listas para correr.

**Toda celda de setup se presenta en una celda markdown propia, antes de la
celda de código.** Cell id: el del setup con el sufijo `-intro`
(`setup-datos` → `setup-datos-intro`). La presentación explica qué hace la
celda, qué nombres deja definidos y qué decisiones ya tomadas conviene que el
alumno entienda antes de seguir. Cuando hay varias celdas de setup seguidas,
la primera presentación abre con un encabezado `## Preparación` y anticipa
qué hace cada una.

Los comentarios adentro de la celda de código **no son el lugar para esa
presentación**. Ahí van, como mucho, explicaciones cortas de qué hace una
línea o de la forma de un tensor. Un bloque de diez líneas de comentario al
tope de una celda es texto de enunciado escrito en el lugar equivocado:
mucha gente no lee los comentarios, y en un notebook la prosa se lee mejor
renderizada que adentro de un `#`.

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

**Ejercicio con partes A y B:** cada parte se presenta por separado, con su
enunciado inmediatamente antes de su celda de código:

```
ej2-enunciado      (objetivo del ejercicio + Parte A)
ej2-code-a
ej2-enunciado-b    (Parte B)
ej2-code-b
ej2-pregunta
ej2-respuesta
```

No juntar las dos partes en un solo enunciado y después poner las dos celdas
de código seguidas: obliga al alumno a subir y bajar, y hace que la Parte B
se lea con la cabeza puesta en la A. El enunciado sin sufijo es el que la app
usa como enunciado del ejercicio; los que llevan sufijo se ignoran, así que
lo que tiene que quedar sí o sí en el primero es el `### Ejercicio N —` y el
`**Objetivo:**`. Las pistas van con la parte a la que corresponden.

### 7. Cómo se redacta una consigna

Es lo que más cuesta y lo que más rinde. Tres reglas:

**1. Si el ejercicio practica algo que está en la teoría, la consigna se
redacta; no se muestra el código.** El trabajo que se le pide al alumno es
volver al notebook de teoría, entender qué hace ese código y trasladarlo a
una situación nueva. Si la consigna trae la línea escrita, ese trabajo
desaparece y el ejercicio se vuelve copiar y pegar.

```markdown
mal   Fijá la semilla con `torch.manual_seed(0)` y creá
      `vectores = torch.rand(4, 6, 8)`.

bien  Fijá la semilla del generador de PyTorch en 0 y creá, en una variable
      llamada `vectores`, un tensor de forma `(4, 6, 8)` con números al azar
      entre 0 y 1.
```

Esto exige consignas bien escritas: la ambigüedad que antes tapaba el código
ahora hay que resolverla con vocabulario preciso.

**2. Si el ejercicio va más allá de la teoría, se puede mostrar código —pero
hay que explicar qué se está agregando y por qué.** Cuando el lab introduce
una función, un método o una técnica que la teoría no trata, o la trata con
menos profundidad, corresponde presentarla: qué hace, para qué la usamos acá,
y que no estaba en la clase. Lo que no va nunca es una consigna que dé por
sabido algo que el alumno no vio.

**3. Vocabulario preciso, sin ambigüedad.** La consigna se lee una vez y
tiene que quedar clara.

- Al pedir que se cree algo, decir **qué se crea y cómo se llama**: "creá un
  tensor y guardalo en una variable llamada `ids`", no "construí ids".
- Nombrar cada cosa por lo que es: *tensor*, *variable*, *función*, *método*,
  *capa*. No "`ids` y `vectores` tienen tipos distintos" sino "los tensores
  `ids` y `vectores`".
- Nada de referencias sueltas a "los datos que ya tenés" o "pasale esto": si
  hay un dato de entrada, va escrito; si hay una operación, se dice cuál.
- Los datos de entrada (una lista de casos de prueba, un corpus juguete, una
  lista de palabras de sondeo) **sí** van como bloque de código en la
  consigna. Eso es dato, no solución.

### 8. Checklist de entrega

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

### 9. Cierre

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
- **Evitar la metáfora contable** ("esto se paga después", "el precio es",
  "sale más barato"). Suena a manual de divulgación y además esconde la
  información: decí *qué* se pierde y *en qué unidad*. "Se paga en tiempo de
  cómputo", "usa el doble de memoria", "recorta el 30% de las órdenes".
- Ningún párrafo debería obligar a releerlo para entender de qué habla. Si
  una oración arranca con una construcción abstracta ("entre X y la primera
  capa hay una cadena de decisiones"), primero se dice concretamente de qué
  se trata y después se la nombra.

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
- [ ] Cada celda de setup tiene su celda markdown de presentación antes.
- [ ] En los ejercicios de varias partes, cada parte tiene su enunciado
      inmediatamente antes de su celda de código.
- [ ] Ninguna consigna muestra el código que el alumno tiene que escribir,
      salvo que sea contenido que la teoría no cubre — y en ese caso está
      explicado.
- [ ] Las consignas dicen cómo se llama cada variable que hay que crear.
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
