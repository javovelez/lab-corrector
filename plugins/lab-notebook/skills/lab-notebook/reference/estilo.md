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

Es lo que más cuesta y lo que más rinde. Seis reglas:

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
- Vale en particular al **pedir que se escriba algo**: un identificador con
  paréntesis no dice si lo que hay que escribir es una función suelta, un
  método de una clase o la clase entera, y el alumno lo tiene que deducir de
  la firma. Se dice.

  ```markdown
  mal   Escribí `codificar_lote(self, textos, largo)`, que devuelve un
        tensor `(B, largo)`.

  bien  Escribí el método `codificar_lote(self, textos, largo)` de la clase
        `Vocabulario`, que devuelve un tensor `(B, largo)`.
  ```
- Nada de referencias sueltas a "los datos que ya tenés" o "pasale esto": si
  hay un dato de entrada, va escrito; si hay una operación, se dice cuál.
- Los datos de entrada (una lista de casos de prueba, un corpus juguete, una
  lista de palabras de sondeo) **sí** van como bloque de código en la
  consigna. Eso es dato, no solución.

**4. La consigna tiene que construir todo lo que un ejercicio posterior usa.**
Es el defecto más caro de un lab y el más difícil de ver leyendo ejercicio por
ejercicio, porque cada uno por separado se entiende bien. Aparece de cuatro
formas:

- Una **función se pide con una firma** y la solución la define con un parámetro
  de más, que recién hace falta dos ejercicios después.
- Una **variable que un enunciado posterior nombra** como si existiera y que
  ninguna consigna pidió crear ("Tenés `E_dist`...").
- Una **estructura intermedia que solo aparece en una pista**. Una pista orienta;
  no pide. Lo que sobrevive al ejercicio va en un punto numerado.
- Una **celda de setup que deja definidos nombres que su presentación no lista**.

El alumno que hace exactamente lo pedido queda trabado, o produce números que no
coinciden con la solución oficial, y en la corrección eso se lee como error suyo.
La solución oficial suele disimularlo con valores por defecto o parámetros que no
están dichos en ninguna parte, y entonces los números salen bien por accidente de
la firma.

La forma de revisarlo es **seguir la cadena de nombres**, no el orden de los
ejercicios: para cada identificador que un enunciado menciona, buscar la consigna
numerada que lo crea; y comparar cada firma de la consigna contra la de la
solución, parámetro por parámetro. Cuando un parámetro existe solo para un
ejercicio posterior, se lo pide igual y se dice para qué.

**5. La metáfora que no describe el mecanismo se reemplaza.** Es la hermana de la
regla de la metáfora contable, más abajo. Una imagen vale cuando el lector se
lleva el mecanismo; cuando solo se lleva la imagen, ocupa el lugar de la
explicación y encima suele ser inexacta.

```markdown
mal   Muestreo negativo: de 15.548 clases a una moneda cargada
bien  Muestreo negativo: de 15.548 clases a una decisión binaria

mal   mostrar dónde se le ve la costura
bien  mostrar dónde falla

mal   un esquema de pesos disfrazado de sorteo
bien  un esquema de pesos implementado como sorteo
```

**6. La respuesta oficial contesta lo que se preguntó, y nada más.** La
rúbrica se genera *desde* la respuesta oficial: todo lo que ella diga se
convierte en algo que se le exige al alumno. Un párrafo de más —el dato
lindo, el adelanto de la unidad que viene, la evidencia extra que a nadie se
le pidió— no es generosidad, es un ítem más en la rúbrica y una injusticia
con el que contestó exactamente lo que se le preguntó.

La forma de revisarlo es leer la pregunta y la respuesta juntas y marcar, para
cada párrafo de la respuesta, cuál de las consignas contesta. El párrafo que
no contesta ninguna sobra, y ahí hay dos salidas:

- **Podar la respuesta.** Es lo habitual, sobre todo si lo que sobra es un
  cierre que mira hacia adelante. Ese material no se pierde: va al enunciado,
  al cierre del lab o a una explicación en prosa que no se corrige.
- **Ampliar la pregunta.** Vale cuando lo que sobra es tan bueno que uno
  quiere que el alumno lo piense. Entonces se pide de manera explícita, y se
  paga el costo de un ítem más — respetando el tope de dos incisos.

El caso inverso también se corrige tocando la pregunta: si la respuesta se va
de largo porque la consigna encadenó cuatro pedidos en un inciso, el problema
es la consigna. Y si un inciso repite algo que ya se preguntó en otro lab o en
otro ejercicio del mismo, conviene bajarlo a prosa del enunciado —queda dicho,
sin ocupar un ítem corregible.

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
- **El voseo no es una licencia para el registro coloquial.** La forma verbal
  es rioplatense; el vocabulario es técnico. Un apunte de cátedra no dice
  "arrancá de un tensor de ceros", "mirá esta tabla" ni "sus vecinos son
  basura". Los reemplazos habituales:

  | coloquial | técnico |
  |---|---|
  | arrancá / arrancar de | partí / partir de, empezar en |
  | mirá, fijate | observá, revisá, notá |
  | ojo con | atención a, cuidado con |
  | acordate de | recordá |
  | andá a ver | revisá, consultá |
  | a ojo | por intuición, sin medir |
  | tirar (datos, un error) | descartar; lanzar (una excepción) |
  | el truco, la magia | el mecanismo, la técnica |
  | de un plumazo | de una sola vez |
  | tal cual | sin cambios, sin modificarlo |
  | basura, poquísimo, fortísimo | sin sentido, muy poco, muy fuerte |
  | entrar (un dato al modelo) | ser la entrada de, ingresar a |
  | meter (un dato, un índice) | pasar, representar |
  | sacar del medio | descartar, excluir |
  | bajar (un archivo) | descargar |
  | anda / anda mejor | funciona / funciona mejor |
  | de verdad (entrenarlo, cuánto sirve) | sobre el corpus real; cuánto aporta |
  | la excusa, el pretexto | el medio, no el fin |
  | dado vuelta | invertido |
  | pegado a cero, lejísimos | muy cerca de cero, muy lejos |
  | se lo lleva; le gana / le pierde | lo ocupa; supera / queda por debajo |
  | se cae (la clase, la celda) | se interrumpe, falla |
  | rankear | ubicar, ordenar |
  | amontonado, desparramado, aplastado | agrupado, disperso, comprimido |
  | chiquito, rarísimo, lejísimo | mínimo, muy raro, muy lejos |
  | no tiene nada que ver con | no guarda relación con |
  | la letra chica; cuánto hay que creerle | qué no dice; cuánta información conserva |

  Los diminutivos y los superlativos en *-ísimo* son el caso más fácil de
  detectar y el más frecuente: no hay ninguno que no tenga una forma técnica.

  La prueba es leerlo en voz alta como si lo dijera la cátedra en el pizarrón:
  lo que suene a conversación de pasillo se reemplaza.
- **Es un texto académico de ingeniería: la imprecisión es un error, no un
  matiz de estilo.** Bajar el registro coloquial no alcanza si la oración
  resultante sigue diciendo algo aproximado. Cinco reglas:

  1. **Cada cifra se atribuye a lo que la produjo**: qué modelo, sobre qué
     partición, con cuántas clases. "Llega al 80% de *accuracy*" es una
     afirmación sin sujeto, y es la vía por la que el número de un lab termina
     citado en otro. Verificar contra la corrida, siempre.
  2. **Las formas de los tensores se nombran por lo que son.** Una reseña es
     una secuencia de $L$ índices; `(B, L)` es la forma del *lote*. Escribir
     "cada reseña se convirtió en un tensor `(B, L)`" es directamente falso.
  3. **El verbo nombra el mecanismo.** `nn.Embedding` *indexa* una tabla, no
     "cambia" el índice ni "lo mete" en el modelo; el promedio *excluye las
     posiciones de relleno*, no "ignora el relleno"; el entrenamiento *ajusta*
     las filas, no las "acomoda".
  4. **El cuantificador vago se reemplaza por la medida, cuando existe.** No
     "muy por encima de las demás" sino "0,850 contra 0,59 del segundo
     vecino"; no "casi todos" sino "siete de los ocho".
  5. **Un término, un sentido, en todo el material.** El caso que ya costó una
     ronda: en esta materia **"forma" es el *shape* de un tensor** —así la usa
     el notebook de teoría, decenas de veces— así que no se la puede usar
     además para "entrada del vocabulario". Una entrada del vocabulario es un
     **token** (es lo que la teoría dice, y es lo correcto: `<pad>`, `<unk>`,
     `42` y `xl` no son palabras); una ocurrencia en el texto es una
     **aparición**; y "palabra" se reserva para cuando el objeto se trata como
     palabra —sus vecinos, su polaridad, una fila por palabra—. Antes de
     introducir un término, revisar con `grep` si ya está tomado en el
     notebook de teoría.
  6. **La voz es consistente**: impersonal en el material teórico ("se
     normaliza cada fila"), segunda persona en las consignas ("normalizá cada
     fila"). Mezclar "nos quedamos con" y "se conserva" en el mismo párrafo es
     un defecto de redacción.
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
- **No unir con «, y» dos oraciones con sujetos distintos.** Es la
  construcción que más se repite en la redacción por defecto: un hecho, una
  coma, una «y», y una segunda oración con otro sujeto que suele ser la causa,
  el propósito o la consecuencia de la primera, cerrada muchas veces con un
  pronombre como «cada una». La «y» no dice qué relación hay entre las dos
  ideas. El pronombre del final puede referirse a más de un nombre. El
  arreglo es partir la oración en dos y nombrar la relación entre las ideas
  con el conector que corresponde («para», «porque», «por eso») o con el
  orden.

  | dice | pasó a decir |
  |---|---|
  | Una entidad puede ocupar varias palabras, y las etiquetas siguen el esquema BIO para marcar dónde empieza cada una. | Una entidad puede ocupar varias palabras. Para marcar en qué palabra empieza cada entidad, las etiquetas siguen el esquema BIO. |

  No es un error la enumeración («A, B y C») ni la «y» que une dos
  predicados del mismo sujeto («la celda descarga el corpus y lo divide»).
- **La primera oración de un párrafo dice un hecho que se entiende sin leer
  las siguientes.** La redacción por defecto abre el párrafo con una oración
  que intenta encuadrarlo. Ese encuadre falla de dos maneras, y las dos se
  repiten:

  1. **La oración síntesis.** La oración resume el párrafo entero en una
     fórmula que solo se entiende después de leerlo. «Un etiquetador de
     entidades se mide contando entidades» no dice qué se cuenta, contra qué
     se compara ni qué es un acierto. El lector la retiene sin poder
     interpretarla.
  2. **El contraste con lo que el lector ya sabe.** La oración junta dos
     ideas con un «pero» para oponer la nueva a una conocida. «El etiquetador
     asigna una etiqueta a cada palabra, pero su desempeño se mide por entidad
     y no por palabra» mezcla qué hace el modelo con cómo se lo evalúa. La
     primera mitad no aporta nada. La oposición queda afirmada sin explicarse.
     Si el contraste importa, el párrafo lo muestra más adelante con un caso
     concreto.

  El arreglo es abrir con el primer hecho del razonamiento, en una oración
  con una sola idea. Ese hecho dice qué se hace y con qué objetos.

  El resto del párrafo presenta **el concepto antes que la herramienta que
  lo implementa**. El orden es el hecho, la definición de cada término nuevo,
  el caso que muestra la consecuencia y, al final, la función o la biblioteca
  que lo calcula. En el caso que originó la regla, la función `medir`
  aparecía en la segunda oración, antes de que el lector supiera qué tenía
  que medir.

  | dice | pasó a decir |
  |---|---|
  | Un etiquetador de entidades se mide contando entidades. La función `medir` extrae las entidades de las etiquetas de referencia y las de las etiquetas predichas. Cada entidad queda definida por su tipo, su primera palabra y su última palabra… | El desempeño del etiquetador se mide comparando las entidades predichas con las entidades de referencia. Una entidad queda definida por tres datos, que son su tipo, su primera palabra y su última palabra. Una entidad predicha es un acierto cuando una entidad de referencia tiene los mismos tres datos. Por eso una entidad reconocida a medias, por ejemplo con una palabra de menos, no es un acierto… La función `medir` (al final, en el párrafo de la implementación) |

  La prueba es leer la primera oración de cada párrafo sola, tapando el
  resto. Si no se puede decir qué afirma con precisión, o si afirma dos
  cosas, se reescribe.

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
- [ ] Cada nombre que un enunciado menciona como existente tiene una consigna
      numerada que lo crea, y ninguna estructura que sobreviva al ejercicio se
      introduce solo en una pista.
- [ ] Las firmas de las consignas coinciden, parámetro por parámetro, con las de
      la solución.
- [ ] Cada celda de setup lista, en su presentación, los nombres que deja
      definidos.
- [ ] No hay metáforas que reemplacen la explicación del mecanismo.
- [ ] Los placeholders son exactamente `# Tu código aquí` y
      `*(Escribí tu respuesta acá)*`.
- [ ] Pregunta y respuesta están en celdas separadas.
- [ ] Cada párrafo de la respuesta oficial contesta alguna de las consignas
      de su pregunta, y ninguna consigna quedó sin contestar.
- [ ] Las pistas van en blockquote con `**Pista:**` en negrita.
- [ ] No hay emoticones en ninguna celda.
- [ ] El lenguaje usa voseo de forma consistente, y sin coloquialismos.
- [ ] Cada consigna que pide escribir algo dice si es una función, un método
      o una clase.
- [ ] `lab_validate.py` da cero errores sobre el enunciado.
- [ ] La solución está ejecutada y guardada con sus outputs.
- [ ] La celda de checklist de entrega está antes del cierre.
- [ ] La celda de cierre anticipa el siguiente laboratorio, si
      corresponde.
