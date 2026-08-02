# Changelog del contrato

Versiones de `CONTRACT_VERSION` en
[`scripts/lab_contract.py`](../../../scripts/lab_contract.py).

Se sube la versión cuando cambia el regex de `cell_id`, los roles, los
placeholders del alumno o el algoritmo de agrupación en ítems. Cambios de
redacción en la guía de estilo no mueven la versión.

Cada entrada declara su **impacto**: qué hay que hacer en los repos de
materia. `materias/check.py` lo lee para armar el reporte, así que la
línea que empieza con `**Impacto:**` tiene que estar en cada versión.

---

## 1.0.0 — 2026-08-02

Primera versión formalizada del contrato. Extrae a `lab_contract.py` lo
que hasta ahora vivía duplicado entre `app/rubric_gen.py` y la
documentación en prosa: regex de `cell_id`, roles válidos, tipo de celda
esperado por rol, placeholders y agrupación en ítems.

Cambios respecto de lo que ya estaba en producción: ninguno en el
comportamiento. El regex, los roles y la agrupación son idénticos a los
que venía aplicando `rubric_gen.scan_ejercicios`.

Nuevo: `lab_validate.py`, que aplica el contrato en tiempo de autoría.
Antes los incumplimientos aparecían recién al abrir la app.

Nuevo: `lab_build.py` acepta `.labconfig.yaml` para redefinir el layout
de salida por materia, en vez de asumir `Laboratorios/` y `Soluciones/`.

**Impacto:** ninguno sobre los notebooks existentes. Las materias que ya
compilaban con el layout por defecto siguen compilando igual sin
`.labconfig.yaml`. Correr `lab_validate.py` sobre los labs publicados
para ver qué avisos aparecen — los avisos no rompen nada, pero marcan
deuda de convención.
