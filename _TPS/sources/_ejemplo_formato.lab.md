---
lab: "1a"
title: "Laboratorio n° 1. Parte A: Fundamentos de PyTorch — Tensores"
subject: "Redes Neuronales Profundas"
block: "1 — Fundamentos de Deep Learning"
intro: |
  Este laboratorio te introduce a las operaciones fundamentales con tensores en PyTorch.
  Al completarlo vas a poder:
  - Crear tensores de distintas formas y tipos.
  - Manipular dimensiones con `reshape` y `view`.
  - Aplicar *broadcasting* en operaciones entre tensores de formas distintas.
instructions:
  - "Completá el código en las celdas marcadas con `# Tu código aquí`."
  - "Respondé las preguntas de análisis en las celdas de texto (tipo Markdown)."
  - "Para resolver cada ejercicio, consultá el material teórico de la Clase 1."
  - "**No está permitido usar bucles `for` o `while` salvo que el enunciado lo indique explícitamente.**"
---

<!-- ════════════════════════════════════════════════════════════════════════
     NOTA: Este archivo es un EJEMPLO del formato .lab.md usando un solo
     ejercicio del Lab 1a. No es la fuente completa del laboratorio.
     ════════════════════════════════════════════════════════════════════════ -->


<!-- ──────────────────────────────────────────────────────────────────────
     CELDAS FIJAS (header, imports, sección) — se incluyen para completitud
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::

::::cell{#titulo type=markdown role=title}
<!-- El build genera esta celda automáticamente a partir del frontmatter
     del documento. Si se prefiere, se puede escribir a mano aquí y el
     build la respeta tal cual. -->
::::

::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::

::::cell{#imports type=code role=setup}
```python
import torch
print(f"Versión de PyTorch: {torch.__version__}")
```
::::

::::cell{#secA type=markdown role=section}
---
## Sección A: Creación de tensores y atributos
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 1 — ejemplo completo con las 4 celdas
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Creación y exploración de un tensor tridimensional

**Objetivo:** Entender cómo se estructura la información en tensores de varias dimensiones y qué atributos los describen.

**Enunciado:**

1. Inicializá un tensor tridimensional lleno de **ceros** con forma `(2, 3, 4)` — es decir, 2 bloques, 3 filas y 4 columnas. Utilizar la función de PyTorch diseñada para esto.
2. A partir de ese tensor, creá otro tensor con **valores aleatorios** (entre 0 y 1) que tenga exactamente la misma forma, usando la función `*_like` correspondiente.
3. Imprimí por pantalla los siguientes atributos del tensor aleatorio:
- Su forma (`.shape`)
- Su tipo de dato (`.dtype`)
- Su número de dimensiones (`.ndim`)
- La cantidad total de elementos (`.numel()`)

> **Pista:** Buscá en la documentación de PyTorch funciones que terminen en `_like` — aceptan un tensor como argumento y devuelven otro con la misma forma y tipo.
::::

::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Paso 1: tensor de ceros con forma (2, 3, 4) ────────────────────────────
# torch.zeros() crea un tensor con todos sus elementos inicializados a 0.0.
# Es el constructor más común cuando queremos reservar espacio con un valor seguro.
ceros = torch.zeros(2, 3, 4)
print("Tensor de ceros:")
print(ceros)

# ─── Paso 2: tensor aleatorio con la misma forma ─────────────────────────────
# torch.rand_like() genera valores uniformes en [0,1) con EXACTAMENTE
# la misma forma y tipo de dato que el tensor que le pasamos.
# Es más seguro que escribir torch.rand(2, 3, 4) a mano: si cambiamos la
# forma del tensor base, este se actualiza automáticamente.
aleatorio = torch.rand_like(ceros)
print("\nTensor aleatorio:")
print(aleatorio)

# ─── Paso 3: atributos del tensor ────────────────────────────────────────────
print("\n--- Atributos ---")
print(f"Forma (.shape): {aleatorio.shape}")
print(f"Tipo de dato (.dtype): {aleatorio.dtype}")
print(f"Número de dimensiones: {aleatorio.ndim}")
print(f"Total de elementos: {aleatorio.numel()}")
# 2 * 3 * 4 = 24 elementos en total
```

```yaml rubric
expected: |
  Tensor de ceros creado con torch.zeros(2,3,4).
  Tensor aleatorio con torch.rand_like(ceros).
  Los 4 atributos impresos correctamente: shape=(2,3,4), dtype=float32, ndim=3, numel()=24.
common_errors:
  - "Usar torch.rand(2,3,4) en vez de torch.rand_like(ceros) — funciona pero no demuestra el concepto."
  - "Olvidar .numel() o confundirlo con len()."
  - "Escribir torch.Zeros (mayúscula) — Python es case-sensitive."
```
::::

::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Qué diferencia hay entre `torch.zeros(2, 3, 4)` y `torch.empty(2, 3, 4)`? ¿En qué situación usarías cada uno?
::::

::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- `torch.zeros(2, 3, 4)`: crea un tensor **inicializado** con ceros. Sabemos exactamente qué valor tiene cada elemento. Es el constructor a usar cuando el valor inicial importa.
- `torch.empty(2, 3, 4)`: **reserva memoria** para el tensor pero no inicializa los valores. Lo que aparece son los bytes que ya estaban en esa zona de memoria (basura). Es marginalmente más rápido que `zeros`, pero solo se usa cuando *inmediatamente después* vamos a asignar todos los valores (por ejemplo, en implementaciones de capas a mano), ya que trabajar con valores no inicializados puede generar resultados imprevisibles.
```

```yaml rubric
expected: |
  Mencionar que zeros inicializa a 0 y empty no inicializa (basura de memoria).
  Dar un caso de uso válido para empty (asignación inmediata posterior, performance).
common_errors:
  - "Decir que empty crea un tensor 'vacío' o 'sin elementos' — confundir vacío con no inicializado."
  - "No mencionar el riesgo de usar valores no inicializados."
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     CHECKLIST + CIERRE
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#checklist type=markdown role=checklist}
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] Los valores numéricos que imprimo son razonables (no hay infinitos, ni `NaN`, ni errores de unidades).
- [ ] Todos los gráficos tienen título, etiquetas en los ejes y grilla.
- [ ] No modifiqué ninguna celda fuera de las de actividad.
::::

::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Completaste la primera parte del laboratorio de fundamentos. Practicaste la creación de tensores, sus atributos y las funciones `*_like`. En la siguiente sección vas a trabajar con `reshape` y `view`.
::::
