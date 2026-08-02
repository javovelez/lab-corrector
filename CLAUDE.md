# CLAUDE.md — lab-corrector

App de corrección de laboratorios + contrato de autoría de notebooks
compatibles con ella. Materias que lo consumen: ver
[`materias/registry.yaml`](materias/registry.yaml).

Panorama general en [README.md](README.md). Doc de la app en
[docs/](docs/README.md).

## La regla que importa

`plugins/lab-notebook/scripts/lab_contract.py` es la **fuente única** del
contrato: el regex de `cell_id`, los roles, los placeholders y la
agrupación en ítems corregibles.

- No dupliques el regex en ningún otro archivo. Si lo necesitás,
  importalo de ahí. [`app/rubric_gen.py`](app/rubric_gen.py) lo hace así
  a propósito.
- La documentación en prosa (`reference/cell-ids.md`) explica y
  justifica el contrato; no lo define. Si cambia el código, actualizá la
  prosa, no al revés.

## Al cambiar el contrato

1. Editar `lab_contract.py`.
2. Subir `CONTRACT_VERSION` si cambió el regex, los roles, los
   placeholders o la agrupación. Cambios de redacción no la mueven.
3. Agregar una entrada al
   [CHANGELOG del contrato](plugins/lab-notebook/skills/lab-notebook/reference/CHANGELOG.md)
   con una línea `**Impacto:**` — `materias/check.py` la parsea para
   armar el reporte, así que es obligatoria.
4. `python materias/check.py --validar` y reportar qué materias quedaron
   desalineadas y qué hay que tocar en cada una.

No actualices la `contract_version` de una materia en el registro hasta
que esa materia esté efectivamente migrada.

## Dónde vive la rúbrica

En los dos lados a la vez, y **no son la misma rúbrica**:

- La del repo de la materia (p. ej. `_TPS/rubricas/`) es la de **testeo**:
  la que se usa para probar el lab mientras se desarrolla.
- La del workdir (`<workdir>/.corrector/rubrica.yaml`) es la de
  **corrección**: la que se aplica a las entregas reales de esa cursada.

Divergen a propósito. No hay sincronización automática ni la debe haber:
si hace falta llevar una de un lado al otro, la copia el docente. No
agregues código que las unifique, ni avisos de que están desincronizadas.

## Al cambiar la app

Si el cambio afecta lo que la app espera de un notebook (nuevos campos
de rúbrica, otro criterio de pareo, otra forma de resolver celdas
faltantes), es un cambio de contrato: seguí el flujo de arriba. Si es
solo UI, flujo de corrección o export, no toca el contrato.

Antes de tocar el código de la app, leer
[docs/04-app-overview.md](docs/04-app-overview.md) y
[docs/09-formatos-archivo.md](docs/09-formatos-archivo.md) para entender
el modelo de datos.

## Convenciones

- Español rioplatense, voseo. Sin emoticones.
- La app es **agnóstica de la materia**: paths absolutos en
  `.corrector/config.json`, nada cableado a un layout de repo. Mantenerlo
  así.
- `lab_build.py` y `lab_validate.py` son stdlib pura, sin dependencias.
  Se corren desde repos de materia que no tienen venv. No les agregues
  imports de terceros.

## Qué NO vive acá

El contenido de las materias — notebooks de clase, enunciados,
soluciones, rúbricas concretas. Eso vive en el repo de cada materia. Este
repo define el formato; no guarda instancias.
