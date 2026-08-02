# 08 — Convenciones de `cell_id`

Los `cell_id` son el **pegamento** de los tres pilares del framework.
La app de corrección los usa para encontrar la celda del alumno que
corresponde a un ítem; `rubric_gen` los escanea para listar los
ítems corregibles; `lab_build` los emite literales desde el `.lab.md`.

Cualquier notebook que respete las convenciones de esta página
funciona con la app — incluso si nunca pasó por `lab_build.py`. El
contrato es ligero a propósito.

## Regex canónico

```python
EJ_ID_RE = re.compile(
    r"^ej(\d+)-(enunciado|code|pregunta|respuesta)(?:-(\w+))?$"
)
```

Captura tres grupos:

1. **Número del ejercicio** — `\d+`. Sin zero-padding (`ej1`, `ej12`,
   `ej100`). Dos dígitos también valen pero no se usa la convención.
2. **Rol** — exactamente uno de:
   - `enunciado` (markdown): texto del ejercicio.
   - `code` (code): celda donde el alumno escribe código.
   - `pregunta` (markdown): pregunta de análisis.
   - `respuesta` (markdown): celda donde el alumno responde.
3. **Sufijo opcional** — `\w+`. Para distinguir múltiples ítems del
   mismo rol dentro de un ejercicio.

Ejemplos válidos:

```
ej1-enunciado
ej1-code
ej1-pregunta
ej1-respuesta

ej5-code-a       ej5-code-b      ej5-code-c        # parte A/B/C
ej8-pregunta-2   ej8-respuesta-2                   # múltiples preguntas
```

Ejemplos inválidos (no matchean):

```
EJ1-CODE              ← case-sensitive
ej-code               ← falta número
ej1_code              ← guion bajo en vez de guion
ej1-CODE              ← rol en mayúsculas
ej1-codigo            ← rol en español
ejercicio1-code       ← prefijo distinto
ej1-code-              ← sufijo vacío
```

## Pareo de pregunta + respuesta por sufijo

Adentro de un ejercicio, un item `analysis` está formado por una
pregunta + una respuesta. El pareo se hace por **sufijo**:

| pregunta | respuesta | item analysis |
|---|---|---|
| `ej1-pregunta` | `ej1-respuesta` | `ej1-analisis` (sin sufijo) |
| `ej8-pregunta-2` | `ej8-respuesta-2` | `ej8-analisis-2` |
| (no hay) | `ej3-respuesta` | `ej3-analisis` (cae al enunciado como pregunta) |
| `ej4-pregunta` | (no hay) | item degenerado, se descarta |

Casos posibles según `rubric_gen.scan_ejercicios`:

- **Pareo simétrico**: ambos cell_ids existen → item con `pregunta_cell`
  + `answer_cell`.
- **Solo respuesta**: `answer_cell` sin `pregunta_cell` → la app cae
  al enunciado del ejercicio como pregunta. Es el caso de los
  ejercicios solo-análisis (Lab 3b ej8).
- **Solo pregunta**: `pregunta_cell` sin `answer_cell` → no hay nada
  para corregir, se descarta.

## Casos especiales

### Ejercicio solo-análisis (sin code)

Es el ejercicio que cierra con una reflexión conceptual y no requiere
código del alumno. Caso: Lab 3b ej8 (comparación Gatys vs Johnson).

Layout esperado:

```
ej8-enunciado    (markdown — puede contener la pregunta dentro)
ej8-respuesta    (markdown — donde el alumno responde)
```

Sin `ej8-code` y sin `ej8-pregunta`. La rúbrica queda con un solo
item `kind: analysis` que tiene `enunciado_cell` y `answer_cell` pero
**sin** `pregunta_cell`. La app, al renderizar la vista corrección,
detecta el caso y muestra el enunciado entero como referencia (en
lugar de pregunta + solución oficial separadas).

### Ejercicio con N partes de código

Caso: ejercicio de varias subpartes (parte A, parte B, parte C).
Cada parte va en su propia celda de código, todas dentro del mismo
ejercicio.

Layout:

```
ej5-enunciado
ej5-code-a       (parte A)
ej5-code-b       (parte B)
ej5-pregunta     (pregunta global del ejercicio)
ej5-respuesta
```

Genera 3 items: dos `code` (sufijos `a` y `b`) y un `analysis`. Los
dos codes se corrigen por separado en la app.

### Ejercicio con múltiples preguntas

Caso: el docente pidió varias preguntas de análisis en el mismo
ejercicio.

Layout:

```
ej7-enunciado
ej7-code
ej7-pregunta       (pregunta 1)
ej7-respuesta      (respuesta 1)
ej7-pregunta-2     (pregunta 2)
ej7-respuesta-2    (respuesta 2)
```

Genera 3 items: 1 code + 2 analysis. Cada analysis se corrige por
separado.

### Ejercicio con código pero sin pregunta de análisis

Layout:

```
ej2-enunciado
ej2-code
```

Genera 1 item `code`. La app no muestra la fila de análisis.

## Celdas no-evaluables (que igual están en el notebook)

Cualquier `cell_id` que **no matchea** el regex se ignora silenciosamente
por `rubric_gen` y por la app. Esto incluye:

- `header` — logo institucional.
- `titulo` — título del lab.
- `reglas` — celda fija de reglas de entrega.
- `imports`, `setup-*` — celdas de configuración.
- `secA`, `secB`, ... — encabezados de sección.
- `checklist` — checklist de entrega.
- `footer` — celda de cierre.
- `notas-correccion` — tabla de notas (solo en la solución).
- IDs hexadecimales random generados por Jupyter cuando se inserta
  una celda extra.

Convención adicional desde [`estilo.md`](estilo.md): los
ids "fijos" del framework usan nombres descriptivos en kebab-case
sin guion bajo. No empiecen con `ej` para evitar el regex.

## Convenciones de orden en el notebook

Aunque la app no las exige, ayuda a la legibilidad seguir este orden
adentro de cada ejercicio:

```
ejN-enunciado
ejN-code   (o ejN-code-a, ejN-code-b en orden alfabético)
ejN-pregunta
ejN-respuesta
```

Para ejercicios con sufijos:

```
ejN-enunciado
ejN-pregunta            (sin sufijo)
ejN-respuesta
ejN-pregunta-2          (con sufijo)
ejN-respuesta-2
```

`rubric_gen.scan_ejercicios` no depende del orden — recorre el
notebook entero acumulando por ejercicio, después emite items
ordenando por sufijo. Pero un orden consistente facilita revisar los
notebooks a mano.

## Por qué este regex y no otro

- **`ej` en español**: la materia es en español rioplatense. Mantener
  los ids en español hace los notebooks legibles para los alumnos.
- **Roles en una palabra**: `enunciado`, `code`, `pregunta`,
  `respuesta`. `code` queda en inglés porque es la convención
  universal de Jupyter para tipo de celda; los demás van en español.
- **Sufijo después del rol** (no antes): así `ej5-code-a` y
  `ej5-code-b` quedan agrupadas alfabéticamente y el rol se ve a
  primer golpe de vista.
- **Sin separadores raros (`_`, `.`, `:`)**: los `cell_id` van a la
  URL de la app (`?item=ej5-code-a&grupo=grupo_03`). Quedarse con
  `[a-z0-9-]` evita escapar.

## Validación

El framework no valida los `cell_id` por adelantado. Los problemas
aparecen al cargar la rúbrica o al entrar a la vista corrección:

- **`cell_id` que matchea pero apunta a una celda del tipo equivocado**
  (p. ej. `ej3-code` en una celda markdown): `rubric_gen` igual lo
  toma. La app puede crashear al intentar renderizarlo. Solución:
  arreglar el id a mano.
- **`cell_id` duplicado en el notebook**: `find_cell` devuelve el
  primero. La app corrige el primero, ignora silenciosamente el
  segundo. Solución: dejar uno solo.
- **`cell_id` esperado por la rúbrica que no existe en el notebook**
  del alumno: la app muestra un panel "celda esperada faltante" y
  ofrece el navegador `↑/↓`. Detalle en
  la doc de la app (`docs/10-troubleshooting.md` en el repo `lab-corrector`).

## Migración de notebooks pre-framework (Lab 2)

Lab 2 se escribió antes del framework: cada `ejN-pregunta` contenía
la pregunta + el placeholder `*(Escribí tu respuesta acá)*` en la
**misma celda**. La convención exige celdas separadas. Hay un parche
en [`scripts/lab2_split_pregunta.py`](../../../scripts/lab2_split_pregunta.py)
que divide cada `ejN-pregunta` en pregunta + respuesta — detalle en
la doc de la app (`docs/10-troubleshooting.md` en el repo `lab-corrector`).
