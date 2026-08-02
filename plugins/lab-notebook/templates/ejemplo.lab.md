---
lab: "1"
title: "Laboratorio n° 1: Título del laboratorio"
subject: "Nombre de la materia"
block: "1 — Nombre del bloque temático"
---

<!-- ════════════════════════════════════════════════════════════════════════
     EJEMPLO DEL FORMATO .lab.md

     Este archivo es una referencia ejecutable: compila con lab_build.py y
     pasa lab_validate.py sin errores. Copialo, borrá lo que no uses y
     escribí encima.

     Muestra los tres tipos de ejercicio que soporta el contrato:
       ej1  — el patrón usual: enunciado, código, pregunta, respuesta
       ej2  — varias partes de código (sufijos -a / -b)
       ej3  — solo análisis, sin celda de código
     ════════════════════════════════════════════════════════════════════════ -->


<!-- ──────────────────────────────────────────────────────────────────────
     CELDAS FIJAS — sus ids no matchean el regex, así que la app las ignora
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#header type=markdown role=header}
![Encabezado](URL_DE_LA_IMAGEN_INSTITUCIONAL)
::::

::::cell{#titulo type=markdown role=title}
# Laboratorio n° 1: Título del laboratorio

**Asignatura:** Nombre de la materia
**Bloque:** 1 — Nombre del bloque temático

---

## Introducción

Párrafo que contextualiza el tema y explica para qué sirve.

Al completar este laboratorio vas a poder:

- Objetivo 1.
- Objetivo 2.

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para resolver cada ejercicio, consultá el material teórico de la Clase 1.
::::

::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::

::::cell{#imports type=code role=setup}
```python
import numpy as np

print(f"Versión de NumPy: {np.__version__}")
```
::::

::::cell{#secA type=markdown role=section}
---
## Sección A: Nombre de la sección
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 1 — el patrón usual: 4 celdas

     La celda de código lleva DOS bloques fenced: el primero (```python) va
     al enunciado, el segundo (```python solution) va a la solución.
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Título descriptivo del ejercicio

**Objetivo:** Una o dos oraciones que describen qué habilidad se practica.

**Enunciado:**

1. Primer paso, redactado con voseo.
2. Segundo paso.
3. Imprimí el resultado.

> **Pista:** Orientá hacia el concepto o la función, sin revelar la solución.
::::

::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Paso 1: descripción del bloque ─────────────────────────────────────────
# Los comentarios explican el POR QUÉ de la decisión, no solo el qué.
datos = np.arange(12).reshape(3, 4)

# ─── Paso 2: verificación explícita ─────────────────────────────────────────
# Los print() de la solución son los que alimentan `graded_outputs` en la
# rúbrica, así que conviene que sean informativos.
print(f"Forma: {datos.shape}")
print(f"Suma total: {datos.sum()}")
```
::::

::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿La pregunta conceptual relacionada con el ejercicio?
::::

::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La respuesta oficial, redactada como se la explicarías a un alumno: primero
el concepto, después el procedimiento.
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 2 — varias partes de código

     Los sufijos -a y -b generan DOS ítems corregibles por separado en la
     app. La pregunta de análisis es una sola, del ejercicio entero.
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Ejercicio con dos partes

**Objetivo:** Mostrar cómo se parte un ejercicio en varias celdas de código.

**Enunciado:**

**Parte A.** Lo que se pide en la primera celda.

**Parte B.** Lo que se pide en la segunda celda.
::::

::::cell{#ej2-code-a type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# Solución de la parte A.
vector = np.linspace(0, 1, 5)
print(vector)
```
::::

::::cell{#ej2-code-b type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# Solución de la parte B.
print(f"Media: {np.linspace(0, 1, 5).mean():.3f}")
```
::::

::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Pregunta que cubre las dos partes?
::::

::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Respuesta oficial.
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 3 — solo análisis, sin código

     Típico del cierre reflexivo. No lleva `ej3-pregunta`: la app usa el
     enunciado como pregunta.
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — Reflexión de cierre

**Objetivo:** Integrar lo practicado en una conclusión conceptual.

**Enunciado:**

Compará los dos enfoques que usaste en los ejercicios anteriores. ¿Cuál
conviene en qué situación, y por qué?
::::

::::cell{#ej3-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta:**

Respuesta oficial de la reflexión de cierre.
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

Mensaje de cierre: qué se practicó y qué viene en el próximo laboratorio.
::::
