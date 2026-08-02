# lab-corrector

Framework de laboratorios corregibles para materias de UTN FRM. Dos
piezas que se distribuyen distinto:

- **La app de corrección** ([`app/`](app/)) — Streamlit + Claude SDK.
  Opera sobre un *workdir* con las entregas de Moodle y produce un
  `grupo_NN.txt` por grupo. Vive y corre acá.
- **El contrato de autoría** ([`plugins/lab-notebook/`](plugins/lab-notebook/))
  — qué tiene que cumplir un notebook para que la app pueda corregirlo.
  Se distribuye como plugin de Claude Code a los repos de cada materia.

Los repos de materia (RNP, Análisis de Señales y Sistemas, ...) no
dependen de este repo: instalan el plugin y quedan con el contrato, el
compilador y el validador disponibles en cualquier carpeta.

## Instalar el contrato en un repo de materia

```
/plugin marketplace add git@github.com:javovelez/lab-corrector.git
/plugin install lab-notebook
```

A partir de ahí, Claude conoce la convención de `cell_id`, el formato
`.lab.md` y la guía de estilo, y tiene a mano el compilador y el
validador.

Mientras iterás el contrato conviene el symlink, que evita el ciclo
commit → `/plugin update`:

```
ln -s "$PWD/plugins/lab-notebook/skills/lab-notebook" ~/.claude/skills/lab-notebook
```

## Correr la app

```
app/.venv/bin/streamlit run app/main.py
```

Instalación y dependencias: [docs/11-instalacion.md](docs/11-instalacion.md).

## Compilar y validar un lab

Desde el repo de la materia:

```
python <plugin>/scripts/lab_build.py sources/Laboratorio_X.lab.md
python <plugin>/scripts/lab_validate.py Laboratorios/Laboratorio_X.ipynb
```

`lab_build.py` es stdlib pura. `lab_validate.py` aplica el mismo regex y
la misma agrupación que la app, así que si pasa el validador, se corrige.

## Registro de materias

[`materias/registry.yaml`](materias/registry.yaml) lista las materias que
consumen el contrato y con qué versión está alineada cada una. Cuando
cambiás el contrato:

```
python materias/check.py              # qué materias quedaron atrasadas
python materias/check.py --validar    # + validar sus notebooks
```

El reporte sale del `impacto` declarado en
[el CHANGELOG del contrato](plugins/lab-notebook/skills/lab-notebook/reference/CHANGELOG.md).

## Estructura

```
app/                                  la app Streamlit
docs/                                 doc de la app (flujo, UI, IA, instalación)
materias/registry.yaml                materias que consumen el contrato
materias/check.py                     reporte de impacto
plugins/lab-notebook/
  scripts/lab_contract.py             fuente única del contrato
  scripts/lab_build.py                .lab.md → dos .ipynb
  scripts/lab_validate.py             valida un notebook contra el contrato
  skills/lab-notebook/SKILL.md        lo que lee Claude al escribir un lab
  skills/lab-notebook/reference/      cell-ids, lab-md, estilo, rúbrica, changelog
```

`lab_contract.py` es la autoridad: lo importan el validador y
[`app/rubric_gen.py`](app/rubric_gen.py). La documentación en prosa
explica y justifica, pero no define.
