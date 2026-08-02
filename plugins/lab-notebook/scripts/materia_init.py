#!/usr/bin/env python3
"""
Arranca una materia nueva con todo lo necesario para escribir labs
compatibles con la app de corrección.

Crea el layout, el `.labconfig.yaml`, un `CLAUDE.md` con lo mínimo que
Claude necesita saber, y una copia del `.lab.md` de ejemplo para arrancar.

Uso:
    python materia_init.py <directorio> --nombre "Procesamiento del Lenguaje Natural" --id pln
    python materia_init.py . --nombre "..." --id pln --dry-run

Nunca pisa un archivo que ya exista: si el repo ya tiene un CLAUDE.md o un
.labconfig.yaml, los deja intactos y te avisa.

Opciones de layout (por defecto, el de RNP):
    --sources sources --enunciados Laboratorios --soluciones Soluciones
    --prefijo "Laboratorio_" --sufijo-solucion "_Solucion"
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
TEMPLATES = SCRIPTS.parent / "templates"


LABCONFIG = """\
# Layout de esta materia para el compilador de labs (`lab_build.py`).
#
# Lo lee el plugin `lab-notebook`, que aporta el contrato de autoría, el
# compilador y el validador.
#
# Los paths son relativos a la raíz del repo (donde vive este archivo).

materia: "{nombre}"
materia_id: {mid}

layout:
  sources:         {sources}
  enunciados:      {enunciados}
  soluciones:      {soluciones}
  prefijo:         "{prefijo}"
  sufijo_solucion: "{sufijo}"
"""


CLAUDE_MD = """\
# CLAUDE.md — {nombre}

Material didáctico de **{nombre}**. Este repo tiene contenido: clases,
laboratorios, soluciones y rúbricas.

## Cómo se escriben los labs

Los laboratorios de esta materia siguen el contrato del plugin
**`lab-notebook`**, que aporta la convención de `cell_id`, el formato
`.lab.md`, la guía de estilo, el compilador y el validador. Es lo que hace
que la app de corrección pueda corregirlos.

Si el plugin no está instalado:

```
/plugin marketplace add git@github.com:javovelez/lab-corrector.git
/plugin install lab-notebook
```

**No dupliques la documentación del contrato acá.** La convención de
`cell_id` y el formato de la fuente viven en el plugin; este archivo solo
lleva lo propio de esta materia.

El layout de salida está en [`.labconfig.yaml`](.labconfig.yaml).

Ciclo de trabajo:

1. Editar `{sources}/Laboratorio_X.lab.md`.
2. Compilar: `python <plugin>/scripts/lab_build.py {sources}/Laboratorio_X.lab.md`
3. Validar: `python <plugin>/scripts/lab_validate.py {enunciados}/Laboratorio_X.ipynb`
   — hasta cero errores.
4. Abrir la solución en Jupyter/Colab, "Reiniciar y ejecutar todo", guardar
   con outputs.
5. Probar el enunciado como si fueras el alumno.

**No edites un `.ipynb` a mano** si existe su `.lab.md`: se regenera en cada
compilación y perdés el cambio.

Hay un ejemplo completo del formato en `{sources}/_ejemplo_formato.lab.md`.

## Las rúbricas de este repo son de testeo

`{rubricas}/` guarda las rúbricas para probar un lab mientras se desarrolla.
La rúbrica con la que se corrige una cursada vive en el workdir de esa
corrección y **es otra**: divergen a propósito.

## Convenciones de contenido

- **Idioma:** español rioplatense, voseo ("creá", "usá", "respondé").
- **Sin emoticones.**
- Evitar el spanglish conjugado; término técnico en inglés en cursiva, o
  castellano técnico estándar.
- Explicar el porqué antes que el cómo.

El resto de la guía de estilo está en el plugin, en `reference/estilo.md`.

## Estado

<!-- Anotá acá qué labs existen, cuáles están publicados y qué falta. -->

- Sin labs todavía.
"""


def main():
    ap = argparse.ArgumentParser(
        description="Arranca una materia nueva compatible con la app de corrección."
    )
    ap.add_argument("destino", type=Path, help="directorio del repo de la materia")
    ap.add_argument("--nombre", required=True, help='p. ej. "Procesamiento del Lenguaje Natural"')
    ap.add_argument("--id", dest="mid", required=True, help="id corto, p. ej. pln")
    ap.add_argument("--sources", default="sources")
    ap.add_argument("--enunciados", default="Laboratorios")
    ap.add_argument("--soluciones", default="Soluciones")
    ap.add_argument("--rubricas", default="rubricas")
    ap.add_argument("--prefijo", default="Laboratorio_")
    ap.add_argument("--sufijo-solucion", dest="sufijo", default="_Solucion")
    ap.add_argument("--dry-run", action="store_true", help="mostrar qué haría, sin escribir")
    args = ap.parse_args()

    destino = args.destino.expanduser().resolve()
    campos = dict(
        nombre=args.nombre, mid=args.mid,
        sources=args.sources, enunciados=args.enunciados,
        soluciones=args.soluciones, rubricas=args.rubricas,
        prefijo=args.prefijo, sufijo=args.sufijo,
    )

    carpetas = [args.sources, args.enunciados, args.soluciones, args.rubricas]
    archivos = {
        ".labconfig.yaml": LABCONFIG.format(**campos),
        "CLAUDE.md":       CLAUDE_MD.format(**campos),
    }
    ejemplo_dst = Path(args.sources) / "_ejemplo_formato.lab.md"

    accion = "[dry-run] " if args.dry_run else ""
    print(f"{accion}Materia: {args.nombre}  [{args.mid}]")
    print(f"{accion}Destino: {destino}\n")

    if not destino.is_dir():
        if args.dry_run:
            print(f"  crearía el directorio {destino}")
        else:
            destino.mkdir(parents=True)
            print(f"  creado {destino}")

    for c in carpetas:
        p = destino / c
        if p.is_dir():
            print(f"  ya existe   {c}/")
        elif args.dry_run:
            print(f"  crearía     {c}/")
        else:
            p.mkdir(parents=True, exist_ok=True)
            (p / ".gitkeep").touch()
            print(f"  creado      {c}/")

    for nombre, contenido in archivos.items():
        p = destino / nombre
        if p.exists():
            print(f"  YA EXISTE, no lo toco: {nombre}")
        elif args.dry_run:
            print(f"  escribiría  {nombre}")
        else:
            p.write_text(contenido, encoding="utf-8")
            print(f"  escrito     {nombre}")

    src_ejemplo = TEMPLATES / "ejemplo.lab.md"
    dst_ejemplo = destino / ejemplo_dst
    if dst_ejemplo.exists():
        print(f"  YA EXISTE, no lo toco: {ejemplo_dst}")
    elif not src_ejemplo.is_file():
        print(f"  falta la plantilla {src_ejemplo}", file=sys.stderr)
    elif args.dry_run:
        print(f"  copiaría    {ejemplo_dst}")
    else:
        shutil.copy2(src_ejemplo, dst_ejemplo)
        print(f"  copiado     {ejemplo_dst}")

    registrado = registrar_path_local(args.mid, destino, args.dry_run)

    print(f"\nSiguiente paso:\n")
    if not registrado:
        print(f"  1. Registrá la ruta local en el repo lab-corrector:")
        print(f"       materias/paths.local.yaml → {args.mid}: \"{destino}\"")
        print(f"     y completá `repo:` en materias/registry.yaml cuando exista"
              f" el remoto.\n")
    else:
        print(f"  1. Completá `repo:` en materias/registry.yaml cuando exista"
              f" el remoto.\n")
    print(f"""  2. Escribí tu primer lab copiando el ejemplo:
       cp {ejemplo_dst} {args.sources}/{args.prefijo}1.lab.md

  3. Compilá y validá:
       python {SCRIPTS}/lab_build.py {args.sources}/{args.prefijo}1.lab.md
       python {SCRIPTS}/lab_validate.py {args.enunciados}/{args.prefijo}1.ipynb
""")


def registrar_path_local(mid: str, destino: Path, dry_run: bool) -> bool:
    """Anota la ruta local en lab-corrector si este script corre desde ahí.

    Cuando el plugin está instalado por copia (`/plugin install`), el repo
    no está al alcance y no hay nada que registrar: se devuelve False y el
    usuario lo hace a mano.
    """
    repo = SCRIPTS.parent.parent.parent          # …/lab-corrector
    paths_local = repo / "materias" / "paths.local.yaml"
    if not (repo / "materias" / "registry.yaml").is_file():
        return False

    linea = f'  {mid}: "{destino}"\n'
    if paths_local.is_file():
        texto = paths_local.read_text(encoding="utf-8")
        if re.search(rf'^\s*{re.escape(mid)}\s*:', texto, re.M):
            print(f"\n  ya registrado en paths.local.yaml: {mid}")
            return True
        nuevo = texto.rstrip("\n") + "\n" + linea
    else:
        nuevo = "# Paths locales de esta máquina. Ignorado por git.\npaths:\n" + linea

    if dry_run:
        print(f"\n  registraría {mid} en {paths_local}")
    else:
        paths_local.write_text(nuevo, encoding="utf-8")
        print(f"\n  registrado {mid} en materias/paths.local.yaml")
    return True


if __name__ == "__main__":
    main()
