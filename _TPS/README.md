# Guía de estilo y formato — Trabajos Prácticos de Redes Neuronales Profundas

Este documento describe con exactitud el formato, la estructura y las convenciones de escritura utilizadas en todos los trabajos prácticos de la materia. Debe consultarse antes de redactar cualquier laboratorio nuevo para garantizar coherencia en la serie.

> **Nota:** esta guía cubre el aspecto **visual/textual** de los notebooks. Para el formato de fuente única `.lab.md`, la rúbrica YAML, la convención de `cell_id` y la app de corrección, ver el wiki técnico en [../docs/](../docs/README.md).

---

## Estructura de directorios

```
_TPS/
├── README.md
├── Laboratorios/
│   ├── README.md          ← instrucciones de entrega para el alumno
│   ├── links.md           ← links de apertura en Colab (uno por laboratorio)
│   ├── Laboratorio_1a.ipynb
│   ├── Laboratorio_1b.ipynb
│   └── Laboratorio_1c.ipynb
└── Soluciones/
    ├── Laboratorio_1a_Solucion.ipynb
    ├── Laboratorio_1b_Solucion.ipynb
    └── Laboratorio_1c_Solucion.ipynb
```

Los enunciados van en `Laboratorios/` y las soluciones en `Soluciones/`. La solución de cada laboratorio tiene el mismo nombre que el enunciado con el sufijo `_Solucion` antes de la extensión. El par enunciado/solución debe mantenerse en sincronía.

El archivo `Laboratorios/links.md` contiene los links de apertura directa en Google Colab para cada laboratorio. **Cada vez que se crea un nuevo laboratorio, debe agregarse una fila a ese archivo** siguiendo el patrón:

```
https://colab.research.google.com/github/javovelez/tps_RNP/blob/main/Laboratorio_Xn.ipynb
```

## Archivos existentes

| Archivo | Carpeta | Descripción |
|---|---|---|
| `Laboratorio_1a.ipynb` | `Laboratorios/` | Fundamentos de PyTorch — Tensores |
| `Laboratorio_1b.ipynb` | `Laboratorios/` | Entrenamiento de redes neuronales |
| `Laboratorio_1c.ipynb` | `Laboratorios/` | Denoising Autoencoder |
| `Laboratorio_1a_Solucion.ipynb` | `Soluciones/` | Solución completa del 1a |
| `Laboratorio_1b_Solucion.ipynb` | `Soluciones/` | Solución completa del 1b |
| `Laboratorio_1c_Solucion.ipynb` | `Soluciones/` | Solución completa del 1c |
| `Laboratorio_2.ipynb` | `Laboratorios/` | Redes Neuronales Convolucionales |
| `Laboratorio_2_Solucion.ipynb` | `Soluciones/` | Solución completa del 2 |
| `Laboratorio_3.ipynb` | `Laboratorios/` | Transferencia de conocimiento |
| `Laboratorio_3_Solucion.ipynb` | `Soluciones/` | Solución completa del 3 |

---

## Estructura del notebook

### 1. Celda de encabezado visual

La primera celda de todo notebook es una celda Markdown con la imagen institucional, sin texto adicional:

```markdown
![Imgur](https://i.imgur.com/acSOZRh.png)
```

### 2. Celda de título y metadatos

La segunda celda es Markdown con el título del laboratorio y los metadatos de la materia:

```markdown
# Laboratorio n° X. Parte Y: Título del laboratorio

**Asignatura:** Redes Neuronales Profundas
**Bloque:** N — Nombre del bloque

---

## Introducción

[Párrafo introductorio que contextualiza el tema.]

[Lista de objetivos del trabajo:]

- Objetivo 1
- Objetivo 2
- Objetivo 3

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para resolver cada ejercicio, consultá el material teórico de la Clase N.
- **No está permitido usar bucles `for` o `while` salvo que el enunciado lo indique explícitamente.**
```

Observaciones:
- El título usa `n°` (con el símbolo de ordinal masculino, no "nro." ni "Nro.").
- Los dos puntos en los metadatos van después de `**Asignatura:**` y `**Bloque:**` seguidos de un espacio y el valor.
- El separador entre bloque y nombre es una raya larga (`—`), no un guion (`-`) ni dos guiones (`--`).
- La instrucción sobre bucles se incluye siempre en laboratorios de tensores/operaciones. En laboratorios de implementación de redes se omite si no aplica.

### 3. Celda de reglas de entrega

Inmediatamente después de los metadatos va una celda Markdown con las reglas de entrega:

```markdown
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
```

Esta celda es fija (mismo texto en todos los laboratorios). El cell ID es `reglas`.

### 4. Celda de imports

Inmediatamente después de las reglas hay una celda de código con los imports necesarios:

```python
import torch
print(f"Versión de PyTorch: {torch.__version__}")
```

En laboratorios más extensos (1b, 1c) los imports completos van en una celda de setup que el enunciado ya provee lista para ejecutar.

### 5. Secciones temáticas

Los ejercicios se agrupan en secciones. Cada sección se abre con una celda Markdown:

```markdown
---
## Sección X: Nombre de la sección
```

Cuando la sección tiene una introducción conceptual (por ejemplo, la introducción al broadcasting), esta va en una celda Markdown separada inmediatamente después del encabezado de sección.

### 6. Bloque de ejercicio

Cada ejercicio ocupa tres celdas consecutivas: enunciado, código y pregunta.

#### 5a. Celda de enunciado (Markdown)

```markdown
### Ejercicio N — Título descriptivo del ejercicio

**Objetivo:** Una o dos oraciones que describen qué habilidad se practica.

**Enunciado:**

1. Primer paso.
2. Segundo paso.
3. Tercer paso.

> **Pista:** Texto de ayuda.
```

- El título del ejercicio usa raya larga (`—`), no guion ni dos guiones.
- El campo `**Objetivo:**` es obligatorio y siempre va en una línea propia.
- El campo `**Enunciado:**` contiene los pasos numerados.
- Las pistas van en blockquote (`>`). Si hay varias pistas o son extensas, se usan listas dentro del blockquote.
- La pista nunca revela la solución directamente; orienta hacia el concepto o la función a usar.

#### 5b. Celda de código (Code)

Para el enunciado (versión del alumno):

```python
# Tu código aquí
```

Si la celda tiene código de andamiaje preescrito (variables, imports, estructura parcial), el placeholder `# Tu código aquí` aparece en el lugar exacto donde el alumno debe escribir.

#### 5c. Celda de pregunta de análisis (Markdown)

```markdown
**Pregunta de análisis:**

¿La pregunta conceptual relacionada con el ejercicio?

- La pregunta de análisis siempre va en la celda siguiente a la celda de código.
- En algunos ejercicios con dos partes (A y B) hay dos celdas de código y una sola celda de pregunta al final.

```
#### 5d. Celda de respuesta a pregunta de análisis (Markdown)
```markdown
*(Escribí tu respuesta acá)*
```

- El texto `*(Escribí tu respuesta acá)*` es el placeholder estándar, siempre en cursiva.

#### 5e. Celda de test

Es una celda de código para verificar si la arquitectura o un pedazo de arquitectura de una red neuronal está bien implementada.

Solo se coloca cuando es necesaria.



### 7. Celda de checklist de entrega

Antes del cierre, una celda Markdown con la checklist de entrega:

```markdown
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] Los valores numéricos que imprimo son razonables (no hay infinitos, ni `NaN`, ni errores de unidades).
- [ ] Todos los gráficos tienen título, etiquetas en los ejes y grilla.
- [ ] No modifiqué ninguna celda fuera de las de actividad.
```

El cell ID es `checklist`. Los ítems pueden ajustarse por laboratorio (por ejemplo, agregar "Los tests pasan sin errores" si el lab tiene celdas de test).

### 8. Celda de cierre

El notebook termina con una celda Markdown de cierre:

```markdown
---
## ¡Listo!

[Mensaje de cierre. Menciona qué se practicó y anticipa el próximo laboratorio.]
```

---

## Convenciones de escritura

### Idioma y registro

- Todo el texto está en **español rioplatense** con uso del **voseo**: "creá", "usá", "imprimí", "completá", "respondé", "observá".
- Los términos técnicos establecidos en inglés se mantienen en inglés y se escriben en cursiva la primera vez que aparecen en una sección: *broadcasting*, *forward pass*, *overfitting*.
- Los nombres de funciones, métodos, atributos y parámetros de PyTorch siempre van en código inline: `.reshape()`, `requires_grad=True`, `.backward()`.
- **No se usan emoticones** en ningún contexto.
- El tono es técnico pero didáctico: explica el concepto, no solo el procedimiento.

### Énfasis y formato inline

| Elemento | Formato |
|---|---|
| Términos técnicos clave (primera mención o énfasis) | `**negrita**` |
| Nombres de funciones, métodos, atributos | `` `código inline` `` |
| Términos en inglés de uso técnico | `*cursiva*` |
| Fórmulas matemáticas en línea | `$fórmula$` |
| Fórmulas matemáticas en bloque | `$$fórmula$$` |

### Separadores

- Las secciones principales siempre se abren con `---` en una celda Markdown independiente.
- Dentro de una celda, `---` se usa para separar conceptualmente bloques de contenido.

### Tablas

Las tablas se usan para:
- Información tabular del dataset (etiquetas y clases).
- Reglas de broadcasting.
- Notas de corrección (solo en archivos de solución).

Formato de encabezados: texto breve, sin puntos al final.

---

## Convenciones de código (celdas de solución)

### Encabezados de bloque internos

Los bloques lógicos dentro de una celda de código se separan con comentarios de línea ancha:

```python
# ─── Descripción del bloque ───────────────────────────────────────────────────
```

El carácter usado es `─` (U+2500, BOX DRAWINGS LIGHT HORIZONTAL), no un guion común. La línea se extiende hasta aproximadamente la columna 80.

### Comentarios

- Los comentarios explican el **por qué**, no solo el qué.
- Cada decisión de diseño no evidente lleva un comentario que la justifica.
- El estilo es: una línea de código, opcionalmente un comentario en la misma línea o en la línea inmediatamente anterior.
- Los comentarios están en español.

Ejemplo de buen comentario:
```python
# torch.rand_like() es más seguro que torch.rand(2, 3, 4) a mano: si cambiamos
# la forma del tensor base, este se actualiza automáticamente.
aleatorio = torch.rand_like(ceros)
```

### Reproducibilidad

Cuando una celda genera valores aleatorios que se mencionan en los comentarios o en el enunciado, se fija la semilla:

```python
torch.manual_seed(0)
```

### Docstrings de funciones

Las funciones definidas en el notebook (especialmente las de setup que el alumno recibe preescritas) tienen docstring en español:

```python
def nombre_funcion(param1, param2):
    """
    Descripción breve.

    Parámetros:
    param1 (tipo): descripción
    param2 (tipo): descripción

    Retorna:
    nombre: descripción
    """
```

### Verificaciones en celdas de solución

Las soluciones incluyen verificaciones explícitas con `print()` que confirman el resultado esperado:

```python
print(f"El número 8 está en: fila={1}, columna={3} → mat_3x4[1, 3] = {mat_3x4[1, 3].item()}")
```

### Manejo de errores esperados

Cuando el ejercicio pide provocar un error a propósito para observarlo, se captura con `try/except`:

```python
try:
    X - medias
except RuntimeError as e:
    print("Error al restar directamente:")
    print(e)
```

---

## IDs de celdas (cell id)

Los IDs de celdas son descriptivos y siguen estas convenciones:

| Patrón | Uso |
|---|---|
| `header` | Celda de logo institucional |
| `reglas` | Celda de reglas de entrega (fija) |
| `imports` | Celda de imports iniciales |
| `secA`, `secB`, `secC`, `secD` | Encabezados de sección |
| `ej1-enunciado`, `ej2-enunciado` | Celda de enunciado del ejercicio N |
| `ej1-code`, `ej2-code` | Celda de código del ejercicio N |
| `ej1-code-b` | Segunda celda de código del ejercicio N (cuando hay dos partes) |
| `ej1-pregunta`, `ej2-pregunta` | Celda de pregunta del ejercicio N |
| `ej1-respuesta` | Celda de respuesta (solo en archivo de solución) |
| `setup-*` | Celdas de setup preescritas (no modificar) |
| `checklist` | Celda de checklist de entrega |
| `footer` | Celda de cierre |
| `notas-correccion` | Tabla de notas (solo en archivo de solución) |

Para ejercicios sin número de sección predefinido, se usa un ID hexadecimal de 8 caracteres (generado por Jupyter automáticamente).

---

## Archivos de solución

Los archivos `_Solucion.ipynb` tienen la misma estructura que los enunciados con estas diferencias:

1. El título agrega `-- SOLUCION` al final: `# Laboratorio n° 1. Parte A: ... -- SOLUCION`.
2. Las celdas de código del alumno (`# Tu código aquí`) se reemplazan por celdas con el código completo y comentado.
3. Las celdas de pregunta de análisis mantienen el texto del enunciado; se agrega una celda Markdown nueva inmediatamente después con la respuesta:

```markdown
**Respuesta a la pregunta de análisis:**

[Texto de la respuesta.]
```

4. Al final del notebook se agrega una sección de corrección:

```markdown
---
## Notas de corrección

| Ejercicio | Conceptos clave a evaluar | Errores frecuentes |
|---|---|---|
| 1 | ... | ... |
```

---

## Estructura de ejercicio — resumen visual

```
[Celda Markdown] ### Ejercicio N — Título
                 **Objetivo:** ...
                 **Enunciado:**
                 1. ...
                 > **Pista:** ...

[Celda Code]     # Tu código aquí

[Celda Markdown] **Pregunta de análisis:**
                 ¿...?
                 *(Escribí tu respuesta acá)*
```

En el archivo de solución:

```
[Celda Markdown] ### Ejercicio N — Título   (sin cambios)
                 **Objetivo:** ...
                 **Enunciado:**

[Celda Code]     # ─── Bloque ────
                 codigo_completo()

[Celda Markdown] **Pregunta de análisis:**  (sin cambios)
                 ¿...?

[Celda Markdown] ** Respuesta a la pregunta de análisis:**
                 Texto de la respuesta.
```

---

## Checklist antes de publicar un nuevo laboratorio

- [ ] La primera celda es solo la imagen de encabezado.
- [ ] La celda de reglas de entrega ("IMPORTANTE: qué celdas podés modificar") está presente.
- [ ] El título sigue el patrón `# Laboratorio n° X. Parte Y: Título`.
- [ ] Los metadatos de asignatura y bloque están presentes.
- [ ] Todas las secciones abren con `---` en celda propia.
- [ ] Cada ejercicio tiene: celda de enunciado, celda de código, celda de pregunta.
- [ ] Los placeholders de código son exactamente `# Tu código aquí`.
- [ ] Los placeholders de respuesta son exactamente `*(Escribí tu respuesta acá)*`.
- [ ] Las pistas van en blockquote con `**Pista:**` o `**Pistas:**` en negrita.
- [ ] No hay emoticones en ninguna celda.
- [ ] El lenguaje usa voseo argentino de forma consistente.
- [ ] El notebook de solución existe y está sincronizado.
- [ ] La sección `## Notas de corrección` está presente en el archivo de solución.
- [ ] La celda de checklist de entrega ("Antes de entregar") está presente antes del cierre.
- [ ] La celda de cierre anticipa el siguiente laboratorio (si corresponde).
- [ ] Se agregó una fila al archivo `Laboratorios/links.md` con el link de Colab del nuevo laboratorio.
