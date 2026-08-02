#!/usr/bin/env python3
"""
Compilador de archivos .lab.md (fuente única de un laboratorio) a dos .ipynb:
un enunciado para el alumno y una solución para el docente.

Uso:
    python lab_build.py <archivo.lab.md>

Convención de paths por defecto (relativa al .lab.md, que vive en sources/):
    <base>/sources/Laboratorio_<N>.lab.md
        -> <base>/Laboratorios/Laboratorio_<N>.ipynb
        -> <base>/Soluciones/Laboratorio_<N>_Solucion.ipynb

El layout se puede redefinir por materia con un `.labconfig.yaml` en
cualquier directorio ancestro del .lab.md (típicamente la raíz del repo):

    layout:
      enunciados: _TPS/Laboratorios      # relativo a la raíz del repo
      soluciones: _TPS/Soluciones
      prefijo:    "Laboratorio_"         # nombre de archivo: <prefijo><lab>
      sufijo_solucion: "_Solucion"

Todos los campos son opcionales; los que falten usan el default. Un
`.lab.md` individual puede además pisar el destino con las claves
`out_enunciado` / `out_solucion` en su frontmatter (paths relativos a la
raíz del repo).

Formato de entrada (ver _TPS/sources/_ejemplo_formato.lab.md):

    ---
    lab: "3"
    title: "..."
    ...
    ---

    ::::cell{#id type=markdown|code role=<role>}
    <contenido>
    [```python solution      <-- solo en celdas code con placeholder]
    [codigo solucion]
    [```]
    [```markdown solution    <-- solo en celdas markdown con respuesta]
    [respuesta]
    [```]
    ::::

Reglas de render:
- Celdas code con un solo bloque ```python  : mismo en ambos notebooks (setup).
- Celdas code con dos bloques (python / python solution): primero = enunciado,
  segundo = solucion.
- Celdas markdown sin bloque 'markdown solution': mismo en ambos.
- Celdas markdown con bloque 'markdown solution': el texto fuera del bloque es
  el enunciado; el texto del bloque reemplaza a la celda en la solucion.
"""
import json
import os
import re
import sys

# ─── Regex ────────────────────────────────────────────────────────────────────
RE_FRONT       = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
RE_CELL_HEAD   = re.compile(
    r"^::::cell\{#(?P<id>[^\s}]+)\s+type=(?P<type>\w+)\s+role=(?P<role>[\w-]+)\}\s*$",
    re.MULTILINE,
)
RE_CELL_END    = re.compile(r"^::::\s*$", re.MULTILINE)
RE_FENCE_OPEN  = re.compile(r"^```(\w+)(?:\s+(\w+))?\s*$")
RE_FENCE_CLOSE = re.compile(r"^```\s*$")


# ─── Layout por materia ──────────────────────────────────────────────────────
LAYOUT_DEFAULT = {
    "enunciados":      "Laboratorios",
    "soluciones":      "Soluciones",
    "prefijo":         "Laboratorio_",
    "sufijo_solucion": "_Solucion",
}

CONFIG_NAME = ".labconfig.yaml"


def find_config(start_dir):
    """Busca .labconfig.yaml hacia arriba. Devuelve (path_config, raiz) o (None, None)."""
    d = os.path.abspath(start_dir)
    while True:
        cand = os.path.join(d, CONFIG_NAME)
        if os.path.isfile(cand):
            return cand, d
        parent = os.path.dirname(d)
        if parent == d:
            return None, None
        d = parent


def parse_layout(config_path):
    """Lee la sección `layout:` del .labconfig.yaml. Parser mínimo, sin dependencias."""
    layout = {}
    in_layout = False
    with open(config_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line[0].isspace():
                in_layout = line.split(":", 1)[0].strip() == "layout"
                continue
            if in_layout and ":" in line:
                k, v = line.split(":", 1)
                v = v.split("#", 1)[0].strip().strip('"').strip("'")
                if v:
                    layout[k.strip()] = v
    return layout


def resolve_paths(src_path, front):
    """Decide dónde escribir los dos .ipynb. Devuelve (enun_path, sol_path)."""
    src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    lab_num = front.get("lab", "X")

    config_path, root = find_config(src_dir)
    layout = dict(LAYOUT_DEFAULT)
    if config_path:
        layout.update(parse_layout(config_path))
    else:
        # Sin config: layout heredado, relativo al padre de sources/.
        root = os.path.dirname(src_dir)

    base_name = f"{layout['prefijo']}{lab_num}"
    sol_name = f"{base_name}{layout['sufijo_solucion']}"

    # El frontmatter del lab puede pisar el destino de forma explícita.
    if "out_enunciado" in front:
        enun_path = os.path.join(root, front["out_enunciado"])
    else:
        enun_path = os.path.join(root, layout["enunciados"], f"{base_name}.ipynb")

    if "out_solucion" in front:
        sol_path = os.path.join(root, front["out_solucion"])
    else:
        sol_path = os.path.join(root, layout["soluciones"], f"{sol_name}.ipynb")

    return enun_path, sol_path


# ─── Parser ──────────────────────────────────────────────────────────────────
def parse_frontmatter(text):
    """Devuelve (dict_de_frontmatter, resto_del_texto)."""
    m = RE_FRONT.match(text)
    if not m:
        return {}, text
    front_text = m.group(1)
    rest = text[m.end():]
    front = {}
    for line in front_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        front[k.strip()] = v.strip().strip('"')
    return front, rest


def split_cells(body):
    """Divide el cuerpo en una lista de celdas (dicts id/type/role/content)."""
    heads = list(RE_CELL_HEAD.finditer(body))
    cells = []
    for h in heads:
        start = h.end()
        rest = body[start:]
        m_end = RE_CELL_END.search(rest)
        if not m_end:
            raise ValueError(f"Celda sin cierre: {h.group('id')}")
        content = rest[:m_end.start()].strip("\n")
        cells.append({
            "id":      h.group("id"),
            "type":    h.group("type"),
            "role":    h.group("role"),
            "content": content,
        })
    return cells


def extract_fenced_blocks(content):
    """
    Extrae todos los bloques con fence de triple-backtick.
    Retorna lista de (lang, modifier, body_str).
    """
    lines = content.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        m = RE_FENCE_OPEN.match(lines[i])
        if m:
            lang = m.group(1)
            mod  = m.group(2)
            i += 1
            code_lines = []
            while i < len(lines) and not RE_FENCE_CLOSE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            blocks.append((lang, mod, "\n".join(code_lines)))
        i += 1
    return blocks


def split_code_cell(content):
    """
    Para una celda code, devuelve (enunciado, solucion).
    Si hay un solo bloque python -> igual en ambos.
    Si hay dos bloques python (normal + solution) -> cada uno va a su lado.
    """
    blocks = extract_fenced_blocks(content)
    py_blocks = [b for b in blocks if b[0] == "python"]
    if len(py_blocks) == 1:
        src = py_blocks[0][2]
        return src, src
    if len(py_blocks) == 2:
        # Asumimos orden: primero enunciado, luego solución (con modifier=solution)
        return py_blocks[0][2], py_blocks[1][2]
    raise ValueError(f"Celda code con {len(py_blocks)} bloques python; esperaba 1 o 2")


def split_markdown_cell(content):
    """
    Para una celda markdown, devuelve (enunciado, solucion).
    Si hay un bloque 'markdown solution', su contenido reemplaza a la celda
    en la solución; el resto del texto (fuera del bloque) es el enunciado.
    Si no hay bloque, el contenido es el mismo en ambos.
    """
    lines = content.splitlines()
    in_block = False
    current_lang = current_mod = None
    block_body = []
    outside = []
    for line in lines:
        if not in_block:
            m = RE_FENCE_OPEN.match(line)
            if m and m.group(1) == "markdown" and m.group(2) == "solution":
                in_block = True
                current_lang, current_mod = m.group(1), m.group(2)
                continue
            outside.append(line)
        else:
            if RE_FENCE_CLOSE.match(line):
                in_block = False
                continue
            block_body.append(line)

    enun = "\n".join(outside).strip("\n")
    sol  = "\n".join(block_body).strip("\n") if block_body else enun
    return enun, sol


# ─── Render a ipynb ──────────────────────────────────────────────────────────
def source_to_lines(source):
    """Convierte un string en la lista de lineas con \\n al final (salvo la ultima)."""
    if source is None:
        return [""]
    if source.endswith("\n"):
        source = source[:-1]
    if source == "":
        return [""]
    parts = source.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def make_cell(cell_id, cell_type, source):
    cell = {
        "cell_type": cell_type,
        "id":       cell_id,
        "metadata": {},
        "source":   source_to_lines(source),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_ipynb(parsed_cells, is_solution):
    cells = []
    for c in parsed_cells:
        src = c["solucion"] if is_solution else c["enunciado"]
        cells.append(make_cell(c["id"], c["type"], src))
    return {
        "cells":    cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat":       4,
        "nbformat_minor": 5,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Uso: python lab_build.py <archivo.lab.md>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        text = f.read()

    front, body = parse_frontmatter(text)
    raw_cells = split_cells(body)

    parsed = []
    for c in raw_cells:
        if c["type"] == "code":
            enun, sol = split_code_cell(c["content"])
        elif c["type"] == "markdown":
            enun, sol = split_markdown_cell(c["content"])
        else:
            raise ValueError(f"Tipo de celda desconocido: {c['type']} ({c['id']})")
        parsed.append({
            "id":        c["id"],
            "type":      c["type"],
            "role":      c["role"],
            "enunciado": enun,
            "solucion":  sol,
        })

    # Variante para la solucion: agregar "-- SOLUCION" al titulo.
    parsed_sol = [dict(c) for c in parsed]
    for c in parsed_sol:
        if c["id"] == "header" and c["type"] == "markdown":
            c["solucion"] = re.sub(
                r"^(# Laboratorio[^\n]*)$",
                r"\1 -- SOLUCION",
                c["solucion"],
                count=1,
                flags=re.MULTILINE,
            )

    # Paths de salida (layout por defecto o .labconfig.yaml de la materia).
    enun_path, sol_path = resolve_paths(path, front)

    nb_enun = build_ipynb(parsed,     is_solution=False)
    nb_sol  = build_ipynb(parsed_sol, is_solution=True)

    os.makedirs(os.path.dirname(enun_path), exist_ok=True)
    os.makedirs(os.path.dirname(sol_path),  exist_ok=True)
    with open(enun_path, "w", encoding="utf-8") as f:
        json.dump(nb_enun, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(sol_path, "w", encoding="utf-8") as f:
        json.dump(nb_sol, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Generado: {enun_path}")
    print(f"Generado: {sol_path}")
    print(f"Celdas:   {len(parsed)}")


if __name__ == "__main__":
    main()
