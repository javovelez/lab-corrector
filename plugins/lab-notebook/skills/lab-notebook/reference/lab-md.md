# 02 — Autoría con `.lab.md`

## Por qué fuente única

Un laboratorio vive en dos notebooks: el enunciado (con celdas a
completar por el alumno) y la solución (con código y respuestas
escritas por el docente). Mantenerlos sincronizados a mano es la
fuente número uno de bugs en publicaciones anteriores: una corrección
en el enunciado se olvida en la solución, o al revés.

La fuente única `.lab.md` resuelve esto editando un solo archivo
markdown que se compila a los dos notebooks. El compilador es
[`scripts/lab_build.py`](../../../scripts/lab_build.py) y vive en el repo.

## Layout de archivos

```
_TPS/
├── sources/                          ← fuente de verdad
│   ├── Laboratorio_3a.lab.md
│   ├── Laboratorio_3b.lab.md
│   └── _ejemplo_formato.lab.md       ← ejemplo legible, no se compila
├── Laboratorios/                     ← .ipynb generado (enunciados)
│   └── Laboratorio_X.ipynb
└── Soluciones/                       ← .ipynb generado (soluciones)
    └── Laboratorio_X_Solucion.ipynb
```

Los `.ipynb` se regeneran cada vez que se compila — **nunca se editan
a mano**. Si necesitás cambiar algo, editás el `.lab.md` y volvés a
compilar.

## Formato del archivo

Un `.lab.md` tiene tres secciones: **frontmatter**, **celdas**,
**comentarios libres**.

### Frontmatter

YAML simple delimitado por `---`. Lo lee `lab_build.py` para
determinar el número de lab (que define el path de salida) y los
metadatos del título. Otros campos (`subject`, `block`, `intro`,
`instructions`) están reservados para una versión futura que genere
la celda de título y la de instrucciones automáticamente — al día
de hoy esas celdas se escriben a mano dentro del `.lab.md`.

```yaml
---
lab: "3a"
title: "Laboratorio n° 3. Parte A: Transferencia de estilo (Gatys)"
subject: "Redes Neuronales Profundas"
block: "3 — Transferencia de conocimiento"
---
```

### Celdas

Cada celda se delimita con un bloque Pandoc-style:

```
::::cell{#cell_id type=markdown|code role=<role>}
contenido de la celda
::::
```

- `#cell_id` es el `cell_id` literal que va al `.ipynb`. Tiene que
  cumplir la convención del framework — ver
  [cell-ids.md](cell-ids.md).
- `type` es `markdown` o `code`.
- `role` es informativo (`enunciado`, `student-code`, `pregunta`,
  `student-answer`, `setup`, `header`, `rules`, `section`,
  `checklist`, `footer`, etc.). El compilador no usa el role hoy, pero
  ayuda a leer el archivo.

### Bloques fenced para enunciado vs. solución

Dentro de una celda `code` se puede escribir uno o dos bloques de
código entre triple-backtick:

- **Un solo bloque `python`** → la misma celda aparece igual en los
  dos notebooks. Útil para celdas de setup, imports, clases auxiliares
  que el alumno recibe pre-escritas.

  ````
  ```python
  import torch
  print(f"Versión de PyTorch: {torch.__version__}")
  ```
  ````

- **Dos bloques: `python` (enunciado) + `python solution` (solución)**
  → el primero va al notebook de enunciado, el segundo al de solución.
  El típico caso "tu código aquí":

  ````
  ```python
  # Tu código aquí
  ```

  ```python solution
  ceros = torch.zeros(2, 3, 4)
  aleatorio = torch.rand_like(ceros)
  print(aleatorio.shape, aleatorio.dtype)
  ```
  ````

Dentro de una celda `markdown` aplica una lógica análoga con bloques
`markdown solution`:

- **Sin bloque `markdown solution`** → la celda es la misma en los dos
  notebooks. Es lo que usás para enunciados, encabezados de sección,
  preguntas de análisis (la pregunta es idéntica en enunciado y
  solución), etc.

- **Con bloque `markdown solution`** → el contenido afuera del bloque
  es el enunciado del alumno (típicamente el placeholder
  `*(Escribí tu respuesta acá)*`); el contenido del bloque reemplaza
  esa celda en la solución (es la respuesta oficial).

  ````
  ::::cell{#ej1-respuesta type=markdown role=student-answer}
  *(Escribí tu respuesta acá)*

  ```markdown solution
  **Respuesta a la pregunta de análisis:**

  `torch.zeros` inicializa todos los elementos a cero, mientras que
  `torch.empty` solo reserva memoria sin tocar los valores...
  ```
  ::::
  ````

### Bloques `yaml rubric` (reservados)

El ejemplo `_ejemplo_formato.lab.md` muestra bloques fenced
`yaml rubric` adentro de las celdas. Hoy `lab_build.py` los **ignora**
(no los procesa). Están pensados para una versión futura del
compilador que genere el scaffolding de la `.rubric.yaml` directamente
desde la fuente — ver pendiente #4 en
el roadmap del repo `lab-corrector`.

## Cómo se compila

```
python <plugin>/scripts/lab_build.py sources/Laboratorio_3a.lab.md
```

El script:

1. **Parsea el frontmatter** y se queda con `lab` (número de lab).
2. **Divide el cuerpo en celdas** matcheando `::::cell{...}` y `::::`.
3. **Para cada celda code**, extrae los bloques fenced `python` y
   `python solution`. Si hay uno solo, usa el mismo en ambos
   notebooks; si hay dos, parea por orden (primero enunciado, después
   solución).
4. **Para cada celda markdown**, busca un bloque `markdown solution`.
   Si existe, su contenido reemplaza la celda en la solución; si no,
   se replica.
5. **Renderiza dos `.ipynb`**:
   - `_TPS/Laboratorios/Laboratorio_<lab>.ipynb`
   - `_TPS/Soluciones/Laboratorio_<lab>_Solucion.ipynb`
   Con `cell_id` estables (los del `.lab.md`), kernel `python3` y sin
   outputs.
6. **Tweak para la solución**: si la celda con id `header` contiene
   un título que matchea `# Laboratorio ...`, le agrega `-- SOLUCION`
   al final.

Los outputs no se generan; hay que abrir el `_Solucion.ipynb` en
Colab/Jupyter y ejecutarlo a mano para que tenga prints e imágenes
guardados (eso lo necesita la rúbrica para inferir `graded_outputs`).

## Ciclo de edición habitual

1. Editar `_TPS/sources/Laboratorio_X.lab.md`.
2. `python <plugin>/scripts/lab_build.py sources/Laboratorio_X.lab.md` — sobreescribe los dos `.ipynb`.
3. Abrir `_TPS/Soluciones/Laboratorio_X_Solucion.ipynb` en Jupyter,
   "Reiniciar y ejecutar todo", y guardar (esto persiste los outputs).
4. (Si es un lab nuevo) regenerar la rúbrica:
   generarla desde la app de corrección.
5. Probar el enunciado en Colab haciendo "ejecutar todo" como si
   fueras el alumno.

## Limitaciones conocidas

- **No hay validación de cell_ids duplicados.** Si por error usás dos
  veces el mismo `#id` adentro del `.lab.md`, el `.ipynb` queda con
  cells de id repetido y todo lo de abajo (rúbrica, app) se confunde.
  Convención: nombrar los ids siguiendo el regex del framework, que
  ya implica unicidad.
- **No hay validación de balance de bloques fenced.** Una celda code
  con tres bloques `python` rompe `lab_build.py` con un `ValueError`.
- **El frontmatter solo se usa para `lab`.** Los demás campos quedan
  documentados pero no se inyectan al notebook automáticamente. Las
  celdas de título, reglas, imports y checklist se escriben adentro
  del `.lab.md` como cualquier otra celda.

## Skill `lab-new` (planificada)

`CLAUDE.md` lista una skill `lab-new` planificada que arrancaría un
`.lab.md` desde un notebook de referencia o desde cero. Hoy no está
implementada — el flujo manual es copiar el ejemplo y reescribirlo.
