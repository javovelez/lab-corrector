---
name: lab-notebook
description: Escribir, compilar y validar notebooks de laboratorio (.lab.md → .ipynb) compatibles con la app de corrección. Usar cuando se cree o modifique un laboratorio, trabajo práctico o TP entregable de una materia; cuando haya que asignar o revisar cell_id de un notebook de cátedra; cuando se pida compilar un .lab.md; o cuando haya que verificar que un notebook se pueda corregir con la app. Aplica a cualquier materia (Redes Neuronales Profundas, Análisis de Señales y Sistemas, etc.).
---

# Autoría de notebooks de laboratorio

Un laboratorio se escribe **una sola vez** en un archivo `.lab.md` y se
compila a dos notebooks: el enunciado (con celdas a completar) y la
solución (con el código y las respuestas del docente). La app de
corrección después lee esos dos notebooks más las entregas de los
alumnos, y encuentra cada ítem corregible por su `cell_id`.

Todo el contrato se reduce a esto: **los `cell_id` tienen que seguir la
convención**. Lo demás es estilo.

Versión del contrato: **1.0.0** (ver `reference/CHANGELOG.md` en el repo
`lab-corrector`).

## Antes de escribir

Averiguá el layout de la materia en la que estás. Buscá un
`.labconfig.yaml` en la raíz del repo — define dónde van las fuentes,
los enunciados y las soluciones, y cómo se nombran los archivos. Si no
existe, asumí el layout por defecto:

```
<raiz>/sources/         ← los .lab.md (fuente de verdad)
<raiz>/Laboratorios/    ← .ipynb de enunciado (generado)
<raiz>/Soluciones/      ← .ipynb de solución (generado)
```

**Nunca edites un `.ipynb` a mano si existe su `.lab.md`.** Se
regeneran en cada compilación y perdés el cambio.

## El contrato de `cell_id`

Regex canónico. Una celda es corregible si —y solo si— su id matchea:

```
^ej(\d+)-(enunciado|code|pregunta|respuesta)(?:-(\w+))?$
```

| id | tipo de celda | qué es |
|---|---|---|
| `ej3-enunciado` | markdown | el texto del ejercicio |
| `ej3-code` | code | donde el alumno escribe código |
| `ej3-pregunta` | markdown | la pregunta de análisis |
| `ej3-respuesta` | markdown | donde el alumno responde |

El sufijo opcional distingue varios ítems del mismo rol en un mismo
ejercicio: `ej5-code-a` / `ej5-code-b`, `ej7-pregunta-2` /
`ej7-respuesta-2`. **Pregunta y respuesta se parean por sufijo** — si
no coinciden, el ítem no se corrige.

Cualquier id que no matchee se ignora sin error: `header`, `titulo`,
`reglas`, `imports`, `setup-*`, `secA`, `checklist`, `footer`. Usalos
para todo lo que no sea evaluable, y **que no empiecen con `ej`**.

Las cinco reglas que rompen la app, por orden de frecuencia:

1. Rol mal escrito — `ej1-codigo`, `ej1-CODE`, `ej1_code`. Los roles son
   esos cuatro literales, en minúscula, separados por guion medio.
2. `cell_id` duplicado en el notebook. La app corrige el primero e
   ignora el resto en silencio.
3. Tipo de celda equivocado — un `ejN-code` sobre una celda markdown.
4. `pregunta` sin su `respuesta`. La pregunta sola no genera ítem.
5. Placeholder inexacto. Tienen que ser literalmente `# Tu código aquí`
   y `*(Escribí tu respuesta acá)*`.

Detalle completo, casos especiales (ejercicio solo-análisis, N partes de
código, N preguntas) y qué hace la app en cada caso borde:
`reference/cell-ids.md`.

## Escribir el `.lab.md`

Frontmatter YAML, después celdas delimitadas al estilo Pandoc:

```
::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
ceros = torch.zeros(2, 3, 4)
print(ceros.shape)
```
::::
```

Un solo bloque fenced → la celda va idéntica a los dos notebooks (setup,
imports, andamiaje). Dos bloques (`python` + `python solution`) → el
primero al enunciado, el segundo a la solución. En celdas markdown la
lógica análoga usa `markdown solution`.

Formato completo, frontmatter y limitaciones conocidas del compilador:
`reference/lab-md.md`.

## Compilar y validar

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/lab_build.py <archivo.lab.md>
python ${CLAUDE_PLUGIN_ROOT}/scripts/lab_validate.py <enunciado.ipynb> [solucion.ipynb]
```

`lab_build.py` es stdlib pura, no necesita entorno virtual.

**Validá siempre después de compilar.** El compilador no verifica los
`cell_id` — emite lo que le escribiste. El validador es el que aplica el
mismo regex y el mismo pareo que usa la app, así que si pasa el
validador, la app lo va a poder corregir. Corrigelo hasta que dé cero
errores antes de dar el lab por terminado.

El validador también acepta un `.lab.md` directamente, útil mientras
escribís:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/lab_validate.py <archivo.lab.md>
```

## Estilo de redacción

Español rioplatense con voseo ("creá", "usá", "respondé"). Sin
emoticones. Términos técnicos del inglés en cursiva la primera vez;
nombres de funciones y parámetros en código inline. El tono explica el
porqué, no solo el cómo.

Patrón usual por ejercicio: enunciado, código, pregunta de análisis,
respuesta. También son válidos el ejercicio solo-análisis (sin celda de
código) y el ejercicio de solo código (sin pregunta).

Guía completa —estructura del notebook, encabezados, checklist de
entrega, convenciones de las celdas de solución, checklist antes de
publicar: `reference/estilo.md`.

## Rúbrica

La rúbrica YAML **no se escribe a mano**: la genera la app de corrección
desde el par enunciado/solución, una llamada a Claude por ítem. Para eso
la solución tiene que estar **ejecutada y guardada con sus outputs**
(los prints y gráficos alimentan el campo `graded_outputs`).

Si necesitás leer o retocar una rúbrica existente, el schema está en
`reference/rubrica-schema.md`.

## Ciclo completo

1. Escribir o editar `sources/Laboratorio_X.lab.md`.
2. `lab_build.py` sobre ese archivo — sobreescribe los dos `.ipynb`.
3. `lab_validate.py` sobre el enunciado generado — hasta cero errores.
4. Abrir la solución en Jupyter/Colab, "Reiniciar y ejecutar todo",
   guardar con outputs.
5. Probar el enunciado como si fueras el alumno (ejecutar todo).
6. Generar la rúbrica desde la app cuando lleguen las entregas.
