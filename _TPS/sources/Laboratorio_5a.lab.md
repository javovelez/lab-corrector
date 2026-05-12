---
lab: "5a"
title: "Laboratorio n° 5. Parte A: Redes Generativas Adversarias"
subject: "Redes Neuronales Profundas"
block: "5 — Modelos generativos"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 5a — Redes Generativas Adversarias (GANs)
     Fuente única. Compilar con: python tools/lab_build.py _TPS/sources/Laboratorio_5a.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_5a.ipynb
             _TPS/Soluciones/Laboratorio_5a_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 5. Parte A: Redes Generativas Adversarias

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 5 — Modelos generativos

---

## Introducción

Una **Red Generativa Adversaria** (*GAN*) es un modelo generativo que se entrena con un mecanismo distinto al de la mayoría de las redes que viste hasta ahora: en lugar de minimizar una distancia fija entre la salida y un objetivo, hace que **dos redes compitan entre sí**. El **generador** intenta producir muestras que parezcan reales; el **discriminador** intenta separar las verdaderas de las falsas. El gradiente que mejora al generador no proviene de comparar píxeles contra una imagen objetivo, sino del **error del discriminador** al clasificar los falsos. Esa es la idea central de las GANs y el motivo por el cual son inestables pero capaces de generar muestras nítidas.

En este laboratorio vas a entrenar una **DCGAN** sobre niveles del videojuego *Super Mario Bros.* La presentación detallada del dataset y de su representación está en la Sección A, donde primero vamos a verlo y entenderlo antes de armar la red.

Al completar el laboratorio vas a poder:

- Definir los bloques convolucionales que arman generador y discriminador en una arquitectura DCGAN.
- Implementar el **bucle de actualización adversaria**, distinguiendo qué gradientes deben fluir en cada paso (uso correcto de `detach()`).
- Entrenar la GAN y leer las **curvas de pérdida** del adversario, entendiendo qué patrones indican equilibrio sano y qué patrones indican colapso.
- Generar niveles nuevos muestreando del espacio latente.
- Hacer una **interpolación entre dos niveles** y observar cómo se comporta el espacio latente del generador.

> **Importante — GPU:** este laboratorio entrena dos redes convolucionales en simultáneo. **Activá la GPU en Colab** antes de empezar: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU*. Sin GPU el entrenamiento tarda demasiado para iterar.

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para resolver cada ejercicio, consultá el material teórico de la Clase 5.
::::


::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::


::::cell{#imports type=code role=setup}
```python
# ─── Setup: imports y detección de GPU ──────────────────────────────────────
# Estos imports cubren todo el laboratorio:
#   - torch / torch.nn / torch.nn.functional: capas, módulos y funciones de pérdida.
#   - numpy / matplotlib / PIL: visualización del dataset y de los niveles generados.
# Detectamos GPU una sola vez y guardamos el dispositivo en `device`. Esa
# variable es la que vas a usar en todos los `.to(device)` de tus ejercicios.
import os
import json
import glob
import shutil
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Dispositivo:        {device}")
```
::::


::::cell{#setup-dataset type=code role=setup}
```python
# ─── Setup: descarga del dataset de niveles ─────────────────────────────────
# Bajamos los 31 niveles de Super Mario Bros. en formato texto (Video Game
# Level Corpus) más los 13 sprites de tiles que vamos a usar para renderizar
# los niveles generados. Todo vive en el repo público de TPs de la materia.
# El dataset queda en `./data/GANs/` relativo al notebook, y solo se baja la
# primera vez.
DATA_DIR = "data/GANs"

if not os.path.isdir(DATA_DIR):
    os.makedirs("data", exist_ok=True)
    tmp = "/tmp/_tps_RNP_lab5a"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/javovelez/tps_RNP.git", tmp,
    ])
    shutil.copytree(os.path.join(tmp, "GANs"), DATA_DIR)
    shutil.rmtree(tmp)

print(f"Dataset listo en ./{DATA_DIR}/")
```
::::


::::cell{#setup-vocab type=code role=setup}
```python
# ─── Setup: vocabulario de tiles ────────────────────────────────────────────
# Los niveles son grillas de caracteres ASCII. Cada caracter representa un
# tipo de tile (bloque, suelo, enemigo, moneda, etc.) y tiene asociado un
# sprite de 16x16 píxeles para visualizar. Cargamos el mapeo desde vocab.json
# que vino con el dataset.
with open(os.path.join(DATA_DIR, "vocab.json")) as f:
    vocab = json.load(f)

CHARACTER_SET = vocab["characters"]                # lista ordenada de 13 chars
char2idx = {c: i for i, c in enumerate(CHARACTER_SET)}
idx2char = {i: c for c, i in char2idx.items()}
VOCAB_SIZE = len(CHARACTER_SET)

print(f"Vocabulario ({VOCAB_SIZE} tiles): {CHARACTER_SET}")
```
::::


::::cell{#setup-data-loading type=code role=setup}
```python
# ─── Setup: lectura, ventaneo y dataset ─────────────────────────────────────
# Cada nivel es un archivo .txt con una grilla de caracteres de 14 filas
# (todas exactamente del mismo alto, con el suelo en la fila inferior) y
# ancho variable. Como son solo 31 niveles, no podemos entrenar directamente
# sobre niveles enteros: tomamos *segmentos* (ventanas horizontales de
# 14×32) recorriendo cada nivel con paso de 8 columnas. El paso < 32 hace
# que las ventanas se solapen (75% de solape), lo que multiplica los
# ejemplos efectivos.
#
# Además aplicamos *data augmentation* con flip horizontal aleatorio. Como
# los tubos tienen lados (`<`/`>` y `[`/`]`), el flip no es un simple
# espejo: hay que mapear cada lado al opuesto para preservar la semántica.
# El resto de los tiles son simétricos y se reflejan tal cual.
#
# El dataset devuelve cada segmento codificado como un tensor multicanal
# (13, 14, 32): un canal por tipo de tile, con valores en {-1, +1}. La
# elección de {-1, +1} (en lugar del clásico {0, 1} de un one-hot) es lo
# que nos permite tratar el problema como una DCGAN estándar y poner una
# `tanh()` al final del generador, igual que en una GAN sobre imágenes RGB.
FLIP_MAP = {"<": ">", ">": "<", "[": "]", "]": "["}


def flip_segment_horizontal(segment):
    """Flip horizontal preservando la semántica de tubos (lados se permutan)."""
    return [
        "".join(FLIP_MAP.get(ch, ch) for ch in reversed(row))
        for row in segment
    ]


class MarioLevelDataset(torch.utils.data.Dataset):
    """Niveles de SMB ventaneados como imágenes multicanal de (13, 14, 32)."""

    def __init__(self, folder_path, segment_width=32, step=8, use_flip=True):
        super().__init__()
        self.segment_width = segment_width
        self.use_flip = use_flip
        self.segments = []

        for file_path in sorted(glob.glob(os.path.join(folder_path, "*.txt"))):
            with open(file_path) as f:
                lines = [line.rstrip("\n") for line in f.readlines()]
            # Aseguro ancho uniforme dentro del nivel rellenando con cielo a la derecha.
            level_width = max(len(line) for line in lines)
            lines = [line.ljust(level_width, "-") for line in lines]
            # Ventaneo con solapamiento: paso=8, ancho=32 → 75% de solape.
            for start in range(0, level_width - segment_width + 1, step):
                self.segments.append([line[start:start+segment_width] for line in lines])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        # Flip horizontal aleatorio (data augmentation). El mapeo de tubos
        # garantiza que un `<...>` flippeado sigue siendo un tubo válido.
        if self.use_flip and torch.rand(()).item() < 0.5:
            segment = flip_segment_horizontal(segment)
        arr = [[char2idx.get(ch, char2idx["-"]) for ch in row] for row in segment]
        idx_tensor = torch.tensor(arr, dtype=torch.long)
        # one-hot → (13, 14, 32) → reescalado de {0,1} a {-1,+1}.
        one_hot = F.one_hot(idx_tensor, num_classes=VOCAB_SIZE)
        one_hot = one_hot.permute(2, 0, 1).float()
        return one_hot * 2.0 - 1.0


dataset = MarioLevelDataset(os.path.join(DATA_DIR, "levels"))
batch_size = 32
data_iter = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0,
)

print(f"Total de segmentos en el dataset: {len(dataset)}")
print(f"Forma de un segmento: {dataset[0].shape} (canales, alto, ancho)")
print(f"Rango de valores: [{dataset[0].min():.1f}, {dataset[0].max():.1f}]")
```
::::


::::cell{#setup-vis type=code role=setup}
```python
# ─── Setup: helpers de visualización ────────────────────────────────────────
# Funciones que usás varias veces a lo largo del laboratorio:
#   - cargar_sprites()      : devuelve el dict {char: PIL.Image} para renderizar.
#   - segmento_a_imagen()   : convierte un tensor (13, H, W) a una imagen Mario.
#   - mostrar_segmentos()   : pega varios segmentos verticalmente para comparar.
#   - mostrar_vocabulario() : tabla visual de los 13 tiles con sus caracteres.
#   - mostrar_nivel_txt_e_img(): un fragmento de un nivel como texto y como
#                                imagen renderizada, lado a lado.
def cargar_sprites():
    """Devuelve {char: PIL.Image RGBA} con los sprites de cada tipo de tile."""
    sprites = {}
    sprites_dir = os.path.join(DATA_DIR, "sprites")
    for char, fname in vocab["char_to_sprite"].items():
        sprites[char] = Image.open(os.path.join(sprites_dir, fname)).convert("RGBA")
    return sprites


def segmento_a_imagen(tensor, sprites, tile_size=16):
    """Convierte un tensor (13, H, W) en una imagen PIL renderizada con sprites.
    Los valores del tensor pueden estar en {-1, +1} o en {0, 1}: solo se mira
    el canal con valor máximo en cada celda (argmax)."""
    idx_grid = tensor.argmax(dim=0).cpu().numpy()
    rows, cols = idx_grid.shape
    sky = (135, 206, 235, 255)
    out = Image.new("RGBA", (cols * tile_size, rows * tile_size), sky)
    for y in range(rows):
        for x in range(cols):
            char = idx2char[int(idx_grid[y][x])]
            sprite = sprites.get(char)
            if sprite is not None:
                out.paste(sprite, (x * tile_size, y * tile_size), mask=sprite)
    return out


def mostrar_segmentos(tensores, sprites, titulos=None, scale=0.4):
    """Muestra una lista de segmentos en una columna vertical."""
    n = len(tensores)
    imgs = [segmento_a_imagen(t, sprites) for t in tensores]
    w, h = imgs[0].size
    fig, axes = plt.subplots(n, 1, figsize=(w * scale / 16, n * h * scale / 16))
    if n == 1:
        axes = [axes]
    for ax, img, i in zip(axes, imgs, range(n)):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if titulos is not None and i < len(titulos):
            ax.set_title(titulos[i], fontsize=9)
    plt.tight_layout()
    plt.show()


def mostrar_vocabulario(sprites, descripciones, ncols=7, scale=1.4):
    """Tabla visual del vocabulario: cada tile con su caracter y su descripción."""
    chars = list(sprites.keys())
    n = len(chars)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * scale, nrows * scale * 1.4))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for ax, char in zip(axes, chars):
        ax.imshow(sprites[char])
        ax.set_title(f"'{char}'\n{descripciones[char]}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def mostrar_nivel_txt_e_img(file_path, sprites, columnas=64):
    """Imprime el texto del nivel y muestra su render para que se vea la
    correspondencia 1-a-1 entre cada caracter y cada tile de la imagen."""
    with open(file_path) as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    lines = [line[:columnas].ljust(columnas, "-") for line in lines]

    print(f"Texto (primeras {columnas} columnas de {os.path.basename(file_path)}):\n")
    for line in lines:
        print("  " + line)

    arr = [[char2idx.get(ch, char2idx["-"]) for ch in row] for row in lines]
    idx_tensor = torch.tensor(arr, dtype=torch.long)
    one_hot = F.one_hot(idx_tensor, num_classes=VOCAB_SIZE).permute(2, 0, 1).float()
    img = segmento_a_imagen(one_hot, sprites)
    plt.figure(figsize=(columnas * 0.13, 14 * 0.13))
    plt.imshow(img); plt.axis("off")
    plt.title("Render con sprites")
    plt.show()


sprites = cargar_sprites()
print(f"Sprites cargados: {len(sprites)} tiles.")
```
::::


::::cell{#setup-graficar type=code role=setup}
```python
# ─── Setup: helper para graficar las curvas adversarias ─────────────────────
def graficar_perdidas(d_hist, g_hist):
    """Curvas de pérdida del discriminador y del generador a lo largo del entrenamiento."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d_hist, label="Discriminador")
    ax.plot(g_hist, label="Generador")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
    ax.set_title("Curvas de pérdida del entrenamiento adversario")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()
```
::::


::::cell{#secA type=markdown role=section}
---
## Sección A: El dataset y la representación multicanal
::::


::::cell{#intro-secA-corpus type=markdown role=intro}
Antes de definir la red vale la pena ver con cuidado **qué le vamos a dar para aprender**. La forma del dato condiciona toda la arquitectura, así que vamos a recorrerla paso a paso: primero un nivel concreto, después el vocabulario de piezas que lo componen, después por qué entrenamos sobre fragmentos en lugar de niveles enteros, y al final cómo lo codificamos numéricamente para que la red pueda procesarlo.

### Un nivel de Super Mario Bros.

Tenemos **31 niveles** del juego *Super Mario Bros.* extraídos del [Video Game Level Corpus (VGLC)](https://github.com/TheVGLC/TheVGLC). Cada nivel del juego está construido como una **grilla de cuadraditos**: a cada cuadradito se lo llama *tile* (en castellano podríamos decir "celda" o "ficha"), y todos tienen el mismo tamaño fijo — en este caso 16×16 píxeles. Cada *tile* puede ser de un **tipo** distinto: un bloque sólido, un trozo de suelo, un enemigo, una moneda, un pedazo de tubo, etc. La imagen que se dibuja para cada tipo de *tile* se llama *sprite*: el dibujito de 16×16 píxeles asociado a ese tipo.

En el corpus, los niveles vienen guardados **como texto plano**: cada archivo es una grilla de caracteres ASCII de **14 filas** y ancho variable (típicamente 200–400 columnas), con el suelo siempre en la fila inferior y el cielo arriba. Cada caracter (`X`, `o`, `?`, etc.) funciona como **nombre del tipo de *tile*** que va en esa celda. La celda siguiente muestra esa idea: imprime las primeras 64 columnas del primer nivel del corpus en formato texto y, debajo, el render que se obtiene reemplazando cada caracter por su *sprite*. Las dos vistas son **la misma información** presentada de dos maneras distintas — hay una correspondencia exacta carácter a carácter entre las dos.
::::


::::cell{#setup-demo-nivel type=code role=setup}
```python
# ─── Un nivel real: texto plano y render con sprites ────────────────────────
# Mostramos las primeras 64 columnas del primer nivel del VGLC. Cada caracter
# de la grilla de texto se corresponde 1-a-1 con un tile de 16x16 px en la
# imagen renderizada. La GAN va a aprender la distribución de patrones que
# aparecen en este tipo de grilla.
nivel_demo = sorted(glob.glob(os.path.join(DATA_DIR, "levels", "*.txt")))[0]
mostrar_nivel_txt_e_img(nivel_demo, sprites, columnas=64)
```
::::


::::cell{#intro-secA-vocab type=markdown role=intro}
### El vocabulario de tipos de *tile*

En todo el corpus aparecen **13 tipos de *tile*** distintos: bloques sólidos, suelo destructible, partes izquierda/derecha y topes de tubo, bloques pregunta, monedas, enemigos, etc. La celda siguiente muestra la lista completa: cada tipo con su *sprite*, su caracter ASCII y una descripción corta. A partir de acá, **un nivel es simplemente una secuencia de caracteres tomados de este vocabulario de 13 símbolos**.
::::


::::cell{#setup-demo-vocab type=code role=setup}
```python
# ─── Tabla visual de los 13 tipos de tile del vocabulario ───────────────────
mostrar_vocabulario(sprites, vocab["char_descriptions"])
```
::::


::::cell{#intro-secA-segmentos type=markdown role=intro}
### De niveles enteros a segmentos

31 niveles son demasiado pocos ejemplos para entrenar una GAN. En lugar de entrenar sobre niveles enteros, tomamos **ventanas horizontales** de 14 filas por 32 columnas, recorriendo cada nivel con paso de 8 columnas (75% de solape entre ventana y ventana). Además aplicamos **flip horizontal aleatorio** como *data augmentation*: cada vez que el dataset entrega un segmento, lo refleja con probabilidad ½. De esta forma, de los 31 niveles salen cientos de segmentos distintos, cada uno disponible además en su versión reflejada. La GAN va a aprender a generar segmentos de tamaño 14×32 — no niveles enteros.

> **Flip con preservación de semántica.** El flip no es un simple espejo de la grilla: los tubos tienen lados (`<`/`>` para los costados, `[`/`]` para los topes). Al reflejar un segmento esos lados se tienen que **permutar**, para que un tubo flippeado siga siendo un tubo válido (con la curvatura del frente apuntando hacia el lado correcto). El helper `flip_segment_horizontal` se ocupa de hacer ese mapeo además del espejo.
::::


::::cell{#intro-secA-multicanal type=markdown role=intro}
### Cómo lo codificamos para que entre a la GAN

Hasta acá vimos un nivel en dos formas: como **texto** (cómo se guarda) y como **imagen renderizada con *sprites*** (cómo se ve cuando jugás). Pero la GAN no procesa ninguna de las dos directamente — necesita una representación **numérica** adecuada para una red convolucional.

La solución es codificar cada segmento como un **tensor multicanal** de forma `(13, 14, 32)`. La idea es exactamente análoga a una imagen RGB:

- Una imagen RGB de tamaño `H × W` se representa como un tensor `(3, H, W)`: tres canales — rojo, verde, azul — cada uno con la grilla de intensidades.
- Un segmento de 14×32 *tiles* lo representamos como un tensor `(13, 14, 32)`: trece canales, uno por **tipo de *tile***, cada uno con la grilla de "este tipo aparece en esta celda" (1) o "no aparece" (0).

Para alinear el rango de los datos reales con la `tanh()` que usa el generador, escalamos los canales de `{0, 1}` a `{-1, +1}` (una multiplicación por 2 menos 1).

> **Importante: la GAN nunca ve un *sprite*.** El generador produce tensores `(13, 14, 32)` y el discriminador clasifica tensores `(13, 14, 32)`. Los *sprites* no entran nunca a un *forward pass*. La imagen RGB con *sprites* aparece **solo al final**, en las celdas de visualización: tomamos el tensor multicanal (real o generado), hacemos `argmax` sobre los 13 canales en cada celda para recuperar el tipo de *tile* dominante, y reemplazamos cada tipo por su *sprite*. Si entrenaras la GAN sin renderizar nunca, funcionaría igual — los *sprites* son para vos.

> **La analogía "imagen RGB ↔ nivel de Mario" es la idea central del lab.** Cuando definas el generador y el discriminador en las próximas secciones, vas a usar **exactamente la misma arquitectura DCGAN** que viste sobre imágenes RGB en la teoría. Lo único que cambia es el número de canales de entrada/salida (`3` → `13`) y las dimensiones espaciales (`64×64` → `14×32`).
::::


::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Explorar el dataset

**Objetivo:** Familiarizarte con los datos antes de modelar. Vas a mirar varios segmentos reales renderizados como imágenes y vas a ver cómo se ve la representación multicanal por dentro.

**Enunciado:**

**Parte A — Visualizar segmentos reales.**

1. Tomá los **primeros 4 segmentos** del `dataset` (`dataset[0]`, `dataset[1]`, `dataset[2]`, `dataset[3]`).
2. Mostralos en una columna usando `mostrar_segmentos(lista, sprites)`.

**Parte B — Mirar un canal del tensor multicanal.**

1. Tomá el segmento `dataset[0]` (un tensor de forma `(13, 14, 32)`).
2. Quedate con **un solo canal**: el correspondiente al caracter `"X"` (bloques sólidos). Acordate que el índice del caracter está en `char2idx`.
3. Imprimí esa matriz de `14 × 32` con `print()`. Vas a ver una grilla de `-1.0` y `1.0` que indica dónde hay bloques `X` en el segmento.

> **Pista:** un tensor multicanal es como apilar varias "imágenes" del mismo tamaño, una por canal. Para sacar un solo canal se indexa la primera dimensión: `tensor[indice_canal]`.
::::


::::cell{#ej1-code-a type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Parte A: cuatro segmentos reales en una columna ────────────────────────
# El dataset devuelve tensores listos para la GAN (con valores en {-1, +1}).
# Los helpers de visualización aplican argmax internamente para renderizar.
ejemplos = [dataset[i] for i in range(4)]
mostrar_segmentos(ejemplos, sprites, titulos=[f"Segmento {i}" for i in range(4)])
```
::::


::::cell{#ej1-code-b type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Parte B: el canal del caracter "X" como mapa binario ───────────────────
# El canal i-ésimo es 1.0 (en realidad +1 después del reescalado) en cada
# celda donde aparece el caracter idx2char[i], y -1.0 en el resto. Con eso
# se ve la silueta del bloque "X" en el segmento.
segmento = dataset[0]
canal_X = segmento[char2idx["X"]]
print(f"Forma del segmento completo: {segmento.shape}")
print(f"Forma de un canal:           {canal_X.shape}")
print(f"\nCanal del caracter 'X' (bloques sólidos):")
print(canal_X)
```
::::


::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

El dataset tiene **31 niveles** completos, pero el `len(dataset)` da varios cientos de segmentos. Mirá cómo construye `MarioLevelDataset` los segmentos: itera sobre cada nivel y toma ventanas de ancho 32 con paso 8, y a la entrega aplica un flip horizontal aleatorio.

1. ¿Por qué se eligió un paso (`step=8`) menor que el ancho del segmento (`32`)? ¿Qué ganaríamos y qué perderíamos si usáramos `step=32` (sin solape)?
2. El flip horizontal **mapea** los caracteres `<` ↔ `>` y `[` ↔ `]` en lugar de simplemente invertir el orden de la fila. ¿Por qué hace falta ese mapeo? ¿Qué pasaría visualmente si flipparamos sin él?
3. Sabiendo que el dataset es chico y que la GAN va a entrenar sobre estos cientos de segmentos solapados, ¿qué riesgo concreto te imaginás que tiene el modelo en términos de **memorización** vs. **generación de niveles novedosos**?
::::


::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

1. **Por qué `step=8` (con solape de 75%).**
   Tomar ventanas con solape **multiplica los ejemplos efectivos**: el mismo tramo de nivel aparece centrado en distintas posiciones dentro de la ventana, así que la GAN ve la misma estructura desplazada y aprende a generarla en cualquier posición horizontal del segmento. Con `step=32` (sin solape) tendríamos una cuarta parte de los segmentos y la red no tendría esa señal de invariancia traslacional. La contraparte es que segmentos consecutivos comparten 24 columnas — los ejemplos no son del todo independientes.

2. **Por qué el flip mapea los lados de los tubos.**
   Los caracteres `<` y `>` codifican el **lado izquierdo** y el **lado derecho** de un tubo, respectivamente (análogo con `[` y `]` para los topes). Si flipparamos sin mapear, el lado izquierdo del tubo aparecería en la posición derecha del segmento — pero seguiría dibujándose con el sprite de "lado izquierdo", lo que produciría tubos visualmente rotos (con la curvatura del frente apuntando hacia afuera). El mapeo `<` ↔ `>` y `[` ↔ `]` mantiene la consistencia: un tubo flippeado sigue siendo un tubo válido. Es un caso particular de un principio más general en *data augmentation*: las transformaciones tienen que respetar las simetrías reales del dominio, no las del array.

3. **Riesgo de memorización.**
   Con un corpus tan chico, hay riesgo concreto de que el generador termine **memorizando trozos de los niveles originales** y los reproduzca casi tal cual. La diversidad real del modelo está limitada por la variedad estructural de los 31 niveles fuente, no por los cientos de segmentos solapados. En la práctica vamos a ver que los niveles generados se parecen mucho a "remezclas" del corpus: el generador aprende a producir patrones plausibles localmente (un tubo verde, una hilera de monedas, una plataforma flotante) pero combinándolos de manera que tienden a reproducir las composiciones originales. Esto es esperable en GANs sobre datasets chicos y no es un bug — es la realidad de aprender una distribución muy concentrada.
```
::::


::::cell{#secB type=markdown role=section}
---
## Sección B: Bloques convolucionales
::::


::::cell{#intro-secB type=markdown role=intro}
Antes de armar el generador y el discriminador vamos a definir los **dos bloques de construcción** que se repiten en cada uno. La idea es la misma que en una DCGAN clásica:

- `G_block`: bloque de **subida de resolución** que va al generador. Hace `ConvTranspose2d` (con `stride=2`, así duplica alto y ancho) seguido de `BatchNorm2d` y `ReLU`.
- `D_block`: bloque de **bajada de resolución** que va al discriminador. Hace `Conv2d` (con `stride=2`, así reduce a la mitad alto y ancho) seguido de `LeakyReLU` (sin batchnorm).

> **Pista:** la asimetría `BatchNorm + ReLU` (generador) vs `solo LeakyReLU` (discriminador) viene del paper original de DCGAN (Radford et al. 2015). Es una receta empírica que estabiliza el entrenamiento. La `LeakyReLU` en el discriminador asegura que pasen pendientes negativas — sin eso, las neuronas que se vuelven negativas al inicio nunca recuperan gradiente.
::::


::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Bloques `G_block` y `D_block`

**Objetivo:** Implementar dos bloques reutilizables, uno para subir resolución (lo usás en el generador) y otro para bajarla (lo usás en el discriminador).

**Enunciado:**

**Parte A — `G_block` (subida de resolución):**

1. Heredá de `nn.Module`.
2. En `__init__` instanciá tres capas con los argumentos del constructor:
   - `nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=strides, padding=padding, bias=False)`.
   - `nn.BatchNorm2d(out_channels)`.
   - `nn.ReLU()`.
3. En `forward(x)` aplicalas en orden: convolución transpuesta → batchnorm → activación.

**Parte B — `D_block` (bajada de resolución):**

1. Igual al anterior pero con `nn.Conv2d` en lugar de `ConvTranspose2d`, **sin batchnorm**, y con `nn.LeakyReLU(alpha)` en lugar de `nn.ReLU`.
2. En `forward(x)` aplicá: convolución → activación.

> **Pista:** los argumentos por defecto que ya están en la firma (`kernel_size=4`, `strides=2`, `padding=1`) están elegidos para que cada bloque divida o multiplique por 2 las dimensiones espaciales. No hace falta cambiarlos en los ejercicios siguientes.
::::


::::cell{#ej2-code-a type=code role=student-code}
```python
class G_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1):
        super().__init__()
        # Tu código aquí

    def forward(self, x):
        # Tu código aquí
        pass
```

```python solution
class G_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1):
        super().__init__()
        # ─── ConvTranspose → BatchNorm → ReLU ─────────────────────────────
        # ConvTranspose2d con stride=2 duplica alto y ancho. BatchNorm2d
        # estabiliza el entrenamiento del generador (en GANs es casi
        # obligatorio). ReLU es la activación canónica del generador
        # según DCGAN.
        # bias=False: cuando hay BatchNorm la convolución no necesita bias,
        # porque BatchNorm ya tiene un parámetro de desplazamiento aprendible.
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=strides, padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```
::::


::::cell{#ej2-code-b type=code role=student-code}
```python
class D_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1, alpha=0.2):
        super().__init__()
        # Tu código aquí

    def forward(self, x):
        # Tu código aquí
        pass
```

```python solution
class D_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1, alpha=0.2):
        super().__init__()
        # ─── Conv → LeakyReLU ─────────────────────────────────────────────
        # En el discriminador NO usamos BatchNorm (la receta DCGAN dice que
        # estabiliza menos del lado del discriminador y a veces hasta perjudica).
        # LeakyReLU con alpha=0.2 garantiza que pasen pendientes negativas:
        # sin eso, neuronas que arrancan negativas se "mueren" porque la ReLU
        # estándar las deja con gradiente 0 para siempre.
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=strides, padding=padding,
            bias=False,
        )
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.act(self.conv(x))
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

El generador usa `nn.ReLU` y el discriminador usa `nn.LeakyReLU`. ¿Por qué se elige una activación distinta para cada red? ¿Qué problema concreto evita la `LeakyReLU` que la `ReLU` estándar **no** evita?
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La `LeakyReLU` se diferencia de la `ReLU` estándar en que, para entradas negativas, no devuelve `0` sino una pendiente chiquita (`alpha * x`, típicamente `alpha = 0.2`). Eso evita el problema conocido como **"ReLU moribunda"** (*dying ReLU*): si una neurona produce siempre valores negativos durante el entrenamiento, la `ReLU` la deja con activación 0 y, lo que es peor, **gradiente 0**. La neurona nunca recibe señal para corregir sus pesos y queda muerta para siempre.

En el **discriminador** este riesgo es alto. El discriminador empieza recibiendo entradas casi al azar (los falsos del generador no entrenado son ruido) y, si se "mueren" muchas de sus neuronas al inicio, pierde capacidad permanente justo cuando más la necesita. La `LeakyReLU` mantiene un canal de gradiente abierto incluso para entradas negativas, así que toda neurona puede recuperarse.

En el **generador** el problema es menos grave: las activaciones tienden a ser más positivas (el generador aprende a producir samples plausibles, no a discriminar) y la pérdida que recibe es más estable. La receta DCGAN deja `ReLU` estándar ahí — empíricamente funciona mejor.

En resumen: la asimetría no es estética, viene de la dinámica adversaria. El discriminador tiene más riesgo de neuronas muertas, así que se le da una activación con "fuga".
```
::::


::::cell{#secC type=markdown role=section}
---
## Sección C: Generador
::::


::::cell{#intro-secC type=markdown role=intro}
El **generador** es la red que vamos a quedarnos al final del entrenamiento: toma un vector latente `z` (ruido gaussiano) y lo convierte en un nivel sintético. Su trabajo es transformar **ruido en estructura**.

La arquitectura es una pila de `G_block` que va **subiendo la resolución espacial** y **bajando los canales**. Empezamos con un tensor de forma `(100, 1, 4)` (un vector latente de 100 dimensiones reformateado como un mapa de `1 × 4`) y terminamos en `(13, 14, 32)`, que es la forma de un segmento de nivel.

La progresión espacial es:

```
(1, 4)   →   (3, 8)   →   (6, 16)   →   (12, 32)   →   (14, 32)
              ↑              ↑              ↑              ↑
         G_block(1,2)    G_block(2,2)   G_block(2,2)   ConvTranspose
         (kernel       (kernel        (kernel        (kernel (3,1),
          (3,5))         clásico        clásico        stride 1)
                         (4,4))         (4,4))
```

El primer bloque tiene un **kernel rectangular** `(3, 5)` con stride `(1, 2)` para empezar a expandir solo la dirección horizontal. Los dos siguientes son bloques DCGAN clásicos (kernel 4, stride 2, padding 1). El bloque final no es un `G_block`: es solo una `ConvTranspose2d` seguida de **`tanh`**, sin batchnorm ni ReLU. La `tanh` mapea cada salida al rango `[-1, +1]`, que es exactamente el rango en el que codificamos los datos reales.
::::


::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — Generador

**Objetivo:** Implementar el generador completo: pila de bloques que sube resolución desde el ruido latente hasta el segmento generado.

**Enunciado:**

1. Heredá de `nn.Module`.
2. En `__init__` definí `self.net = nn.Sequential(...)` con, en orden:
   - Un `G_block(in_channels=latent_dim, out_channels=n_g*8, kernel_size=(3, 5), strides=(1, 2), padding=(0, 1))`. Lleva de `(latent_dim, 1, 4)` a `(n_g*8, 3, 8)`.
   - Un `G_block(in_channels=n_g*8, out_channels=n_g*4)`. Lleva a `(n_g*4, 6, 16)`.
   - Un `G_block(in_channels=n_g*4, out_channels=n_g*2)`. Lleva a `(n_g*2, 12, 32)`.
   - Una `nn.ConvTranspose2d(n_g*2, vocab_size, kernel_size=(3, 1), stride=(1, 1), padding=(0, 2), bias=False)`. Lleva a `(vocab_size, 14, 32)`.
   - Una `nn.Tanh()`.
3. En `forward(z)` devolvé `self.net(z)`.

> **Pista:** los kernels asimétricos `(3, 5)` al principio y `(3, 1)` al final están elegidos a propósito para que las dimensiones espaciales lleguen exactamente a `14 × 32`. No hace falta entender en detalle la aritmética de la convolución transpuesta para usarlos — pasalos tal como aparecen en el enunciado.
::::


::::cell{#ej3-code type=code role=student-code}
```python
class Generator(nn.Module):
    def __init__(self, vocab_size, latent_dim=100, n_g=64):
        super().__init__()
        # Tu código aquí

    def forward(self, z):
        # Tu código aquí
        pass
```

```python solution
class Generator(nn.Module):
    def __init__(self, vocab_size, latent_dim=100, n_g=64):
        super().__init__()
        # ─── Pila de upsampling: (latent_dim, 1, 4) → (vocab_size, 14, 32) ──
        # Cada G_block duplica las dimensiones espaciales (excepto el primero,
        # que tiene un kernel rectangular para empezar a expandir solo el
        # ancho). El bloque final NO es un G_block: es una ConvTranspose2d
        # seguida de tanh, sin batchnorm ni ReLU. La tanh mapea cada canal
        # al rango [-1, +1], que es el rango en el que codificamos los datos
        # reales en el dataset.
        self.net = nn.Sequential(
            G_block(in_channels=latent_dim, out_channels=n_g * 8,
                    kernel_size=(3, 5), strides=(1, 2), padding=(0, 1)),
            G_block(in_channels=n_g * 8, out_channels=n_g * 4),
            G_block(in_channels=n_g * 4, out_channels=n_g * 2),
            nn.ConvTranspose2d(n_g * 2, vocab_size,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(0, 2), bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# ─── Verificación de forma ──────────────────────────────────────────────────
# Con un ruido (B, 100, 1, 4) la salida tiene que ser (B, 13, 14, 32).
G = Generator(vocab_size=VOCAB_SIZE).to(device)
z_demo = torch.randn(2, 100, 1, 4, device=device)
print(f"Forma de la salida del generador: {G(z_demo).shape}")
```
::::


::::cell{#ej3-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La última capa del generador es una `nn.Tanh`. ¿Por qué es **necesaria** ahí, y por qué la elegimos en lugar de una `nn.Sigmoid` que también acota la salida a un rango finito?
::::


::::cell{#ej3-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Los segmentos reales del dataset están codificados con valores en `{-1, +1}` (one-hot reescalado). Para que el discriminador no pueda separar trivialmente reales de falsos por su rango, el generador tiene que producir salidas en **el mismo rango**. La `tanh` mapea cualquier valor real a `(-1, +1)`, exactamente el rango que necesitamos.

- **Sin activación final**: el generador produciría valores reales sin restricción. Al inicio del entrenamiento, los falsos podrían estar en un rango completamente distinto al de los reales, y el discriminador aprendería a separarlos por rango (no por estructura). El gradiente que llega al generador queda dominado por "ajustar el rango" y la dinámica adversaria no progresa.

- **Con `Sigmoid`**: la salida quedaría en `(0, 1)`. Si los datos reales están en `{-1, +1}`, la asimetría rompe lo mismo que en el punto anterior. Para usar `Sigmoid` deberíamos cambiar también la codificación del dataset a `{0, 1}` clásica.

La elección **`tanh` + datos en `{-1, +1}`** es el patrón canónico de DCGAN. Para imágenes RGB se hace exactamente lo mismo: las imágenes se preprocesan a `[-1, +1]` y el generador termina en `tanh`. Acá replicamos ese patrón con 13 canales en vez de 3.
```
::::


::::cell{#secD type=markdown role=section}
---
## Sección D: Discriminador
::::


::::cell{#intro-secD type=markdown role=intro}
El **discriminador** es la red que clasifica entre real y falso. Su trabajo es exactamente el opuesto al del generador: toma un segmento `(13, 14, 32)` (real o sintético) y produce un único número, el **logit** de la probabilidad de que el segmento sea real.

La arquitectura es una pila de `D_block` que va **bajando la resolución espacial** y **subiendo los canales**, espejando al generador. Después una `Conv2d` final colapsa el mapa a un único valor por muestra.

```
(13, 14, 32)   →   (n_d, 7, 16)   →   (n_d*2, 3, 8)   →   (n_d*4, 1, 4)   →   (1,)
                  ↑                  ↑                    ↑                    ↑
              D_block (4,4)      D_block (4,4)        D_block (4,4)        Conv (1,4)
              stride 2           stride 2             stride 2             stride 1
```

La salida es un **logit** (un número real sin acotar), no una probabilidad. La función de pérdida que vamos a usar (`BCEWithLogitsLoss`) aplica internamente una `sigmoid` antes de calcular la entropía cruzada. Esa combinación es **numéricamente más estable** que aplicar `sigmoid` a mano y después `BCELoss` por separado.
::::


::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — Discriminador

**Objetivo:** Implementar el discriminador como un clasificador binario sobre segmentos.

**Enunciado:**

1. Heredá de `nn.Module`.
2. En `__init__` definí `self.net = nn.Sequential(...)` con, en orden:
   - Un `D_block(in_channels=vocab_size, out_channels=n_d)`. Lleva de `(vocab_size, 14, 32)` a `(n_d, 7, 16)`.
   - Un `D_block(in_channels=n_d, out_channels=n_d*2)`. Lleva a `(n_d*2, 3, 8)`.
   - Un `D_block(in_channels=n_d*2, out_channels=n_d*4)`. Lleva a `(n_d*4, 1, 4)`.
   - Una `nn.Conv2d(n_d*4, 1, kernel_size=(1, 4), stride=1, padding=0)`. Colapsa el mapa a `(1, 1, 1)`.
   - Un `nn.Flatten()`. Aplana a `(1,)` por muestra.
3. En `forward(x)` devolvé `self.net(x)`.

> **Pista:** la última `Conv2d` con `kernel_size=(1, 4)` actúa como una "lineal espacial": como el mapa de entrada ya tiene altura 1 y ancho 4, ese kernel cubre toda la región y produce un solo valor por muestra. Usar `Conv2d` en vez de `nn.Linear` permite mantener todo el discriminador como una pila convolucional uniforme.
::::


::::cell{#ej4-code type=code role=student-code}
```python
class Discriminator(nn.Module):
    def __init__(self, vocab_size, n_d=64):
        super().__init__()
        # Tu código aquí

    def forward(self, x):
        # Tu código aquí
        pass
```

```python solution
class Discriminator(nn.Module):
    def __init__(self, vocab_size, n_d=64):
        super().__init__()
        # ─── Pila de downsampling: (vocab_size, 14, 32) → (1,) ──────────────
        # Cada D_block reduce a la mitad las dimensiones espaciales y duplica
        # los canales. Después de tres bloques llegamos a (n_d*4, 1, 4). La
        # Conv2d final con kernel (1, 4) colapsa el mapa a (1, 1, 1) y el
        # Flatten lo aplana a un escalar por muestra. La salida es un LOGIT
        # (sin sigmoide) — la sigmoide se aplica adentro de BCEWithLogitsLoss
        # cuando calculemos las pérdidas.
        self.net = nn.Sequential(
            D_block(in_channels=vocab_size, out_channels=n_d),
            D_block(in_channels=n_d, out_channels=n_d * 2),
            D_block(in_channels=n_d * 2, out_channels=n_d * 4),
            nn.Conv2d(n_d * 4, 1, kernel_size=(1, 4), stride=1, padding=0),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


# ─── Verificación de forma ──────────────────────────────────────────────────
# Con un batch (B, 13, 14, 32) la salida tiene que ser (B, 1).
D = Discriminator(vocab_size=VOCAB_SIZE).to(device)
x_demo = torch.randn(2, VOCAB_SIZE, 14, 32, device=device)
print(f"Forma de la salida del discriminador: {D(x_demo).shape}")
```
::::


::::cell{#ej4-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

El discriminador devuelve un **logit** (un número real sin acotar) en lugar de una **probabilidad** en `[0, 1]`. Si quisiéramos interpretar la salida del discriminador como "qué tan real cree que es el segmento", ¿qué tendríamos que hacerle al logit? ¿Y por qué se prefiere mantener la salida como logit y aplicar `BCEWithLogitsLoss` en vez de aplicar `sigmoid` adentro de la red y usar `BCELoss`?
::::


::::cell{#ej4-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Para interpretar el logit como probabilidad hay que aplicarle la función **sigmoide**: `p = 1 / (1 + exp(-logit))`. Logits cerca de `0` corresponden a `p ≈ 0.5` (el discriminador no sabe), logits muy positivos a `p ≈ 1` ("creo que es real") y logits muy negativos a `p ≈ 0` ("creo que es falso").

La razón para mantener la salida como logit y combinarla con `BCEWithLogitsLoss` es **numérica**, no conceptual. La fórmula de la entropía cruzada binaria contiene términos como `log(sigmoid(logit))` y `log(1 - sigmoid(logit))`. Si calculamos la sigmoide primero y después aplicamos `log`, los logits muy negativos hacen que `sigmoid(logit)` sea muy chiquito (cerca de `0`), y `log(0)` desborda a `-inf`. Lo mismo en el extremo opuesto con `1 - sigmoid(logit)`.

`BCEWithLogitsLoss` evita este problema usando una identidad matemática (la fórmula del *log-sum-exp* estable) que combina la sigmoide y el logaritmo en una sola expresión sin valores extremos. Como las pérdidas en GANs pueden producir logits grandes (sobre todo cuando el discriminador "gana" mucho), esa estabilidad numérica es crítica para que el entrenamiento no produzca `nan`.
```
::::


::::cell{#secE type=markdown role=section}
---
## Sección E: Funciones de actualización adversaria
::::


::::cell{#intro-secE type=markdown role=intro}
Lo que distingue a una GAN de cualquier otra red que viste es **cómo se actualizan los pesos**. En lugar de un único `loss.backward()` por batch, hay **dos pasos** alternados:

1. **Paso del discriminador.** Se calcula la pérdida del discriminador sobre un batch que mezcla reales y falsos, se hace backward, y se actualizan **solo** los pesos de `D`. Los pesos de `G` no se modifican.
2. **Paso del generador.** Se calcula la pérdida del generador (qué tan bien engañó al discriminador), se hace backward, y se actualizan **solo** los pesos de `G`. Los pesos de `D` no se modifican.

La separación se logra con dos optimizadores distintos (`opt_D` solo registra los parámetros de `D`, `opt_G` solo los de `G`) y con un truco más sutil: durante el paso de `D`, los falsos del generador se pasan por `D` con un `.detach()` que **corta el grafo de gradientes**. Sin ese detach, el `loss.backward()` del paso de `D` también acumularía gradientes en los pesos de `G` (consumiendo cómputo y posiblemente interfiriendo con el siguiente paso).

> **Detalle clave:** durante el paso del **generador** sí queremos que los gradientes fluyan a través de `D` hasta los pesos de `G` — pero `opt_G.step()` actualiza solo los parámetros de `G`. Los gradientes que se acumulan en `D` durante ese backward se descartan en el siguiente `opt_D.zero_grad()`.

La pérdida en ambos casos es la **entropía cruzada binaria** (`BCEWithLogitsLoss`). La etiqueta cambia según el caso:

| Paso | Entrada al discriminador | Etiqueta objetivo |
|---|---|---|
| `update_D` (rama real) | `D(real)` | `1` (real) |
| `update_D` (rama falsa) | `D(fake.detach())` | `0` (falso) |
| `update_G` | `D(fake)` (sin detach) | `1` (real, queremos engañar) |
::::


::::cell{#ej5-enunciado type=markdown role=enunciado}
### Ejercicio 5 — Funciones `update_D` y `update_G`

**Objetivo:** Implementar las dos rutinas de actualización del entrenamiento adversario.

**Enunciado:**

**Parte A — `update_D(real, Z, G, D, opt_D)`:**

1. Hacé `opt_D.zero_grad()` para limpiar gradientes acumulados.
2. Pasá `real` por el discriminador: `d_real = D(real)`.
3. Calculá la pérdida sobre los reales con `F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))`.
4. Generá los falsos: `fake = G(Z)`.
5. Pasá los falsos por el discriminador **con `.detach()`** para cortar el grafo: `d_fake = D(fake.detach())`.
6. Calculá la pérdida sobre los falsos con `F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))`.
7. Sumá las dos pérdidas, hacé `loss_D.backward()` y `opt_D.step()`.
8. Devolvé `loss_D`.

**Parte B — `update_G(Z, G, D, opt_G)`:**

1. Hacé `opt_G.zero_grad()`.
2. Generá los falsos: `fake = G(Z)`.
3. Pasá los falsos por el discriminador **sin `.detach()`**: `d_fake = D(fake)`.
4. Calculá la pérdida con etiqueta `1` (queremos que `D` los confunda con reales): `F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))`.
5. Hacé `loss_G.backward()` y `opt_G.step()`.
6. Devolvé `loss_G`.

> **Pista:** la única diferencia operativa entre el paso de `D` y el paso de `G` (más allá del optimizador y la etiqueta) es **el detach** en `D(fake.detach())` durante el paso de `D`. Esa línea es el corazón del entrenamiento adversario.
::::


::::cell{#ej5-code-a type=code role=student-code}
```python
def update_D(real, Z, G, D, opt_D):
    # Tu código aquí
    pass
```

```python solution
def update_D(real, Z, G, D, opt_D):
    # ─── Limpiar gradientes acumulados de iteraciones previas ───────────────
    opt_D.zero_grad()

    # ─── Rama real: queremos D(real) → 1 ────────────────────────────────────
    d_real = D(real)
    loss_real = F.binary_cross_entropy_with_logits(
        d_real, torch.ones_like(d_real)
    )

    # ─── Rama falsa: queremos D(fake) → 0 ───────────────────────────────────
    # G(Z) construye un grafo de gradientes a través del generador. Si NO
    # cortamos con .detach(), el loss_D.backward() también acumularía
    # gradientes en G — innecesarios y potencialmente perjudiciales (el paso
    # de G recalcula su propio forward y no quiere que opt_D.step() ya haya
    # tocado nada que dependa de G).
    fake = G(Z)
    d_fake = D(fake.detach())
    loss_fake = F.binary_cross_entropy_with_logits(
        d_fake, torch.zeros_like(d_fake)
    )

    # ─── Backward y paso del optimizador (solo afecta a D) ──────────────────
    loss_D = loss_real + loss_fake
    loss_D.backward()
    opt_D.step()
    return loss_D
```
::::


::::cell{#ej5-code-b type=code role=student-code}
```python
def update_G(Z, G, D, opt_G):
    # Tu código aquí
    pass
```

```python solution
def update_G(Z, G, D, opt_G):
    # ─── Limpiar gradientes del generador ───────────────────────────────────
    opt_G.zero_grad()

    # ─── Forward "engañar al discriminador" ─────────────────────────────────
    # Acá NO usamos detach: queremos que el gradiente fluya a través de D
    # hasta los pesos de G. La etiqueta es 1 (real) porque G es exitoso si
    # hace que D crea que sus salidas son reales. Aunque el backward acumula
    # gradientes también en los parámetros de D, el opt_G.step() solo actualiza
    # los parámetros de G — los de D se descartan en el próximo opt_D.zero_grad().
    fake = G(Z)
    d_fake = D(fake)
    loss_G = F.binary_cross_entropy_with_logits(
        d_fake, torch.ones_like(d_fake)
    )

    loss_G.backward()
    opt_G.step()
    return loss_G
```
::::


::::cell{#ej5-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

En `update_D` usamos `D(fake.detach())` y en `update_G` usamos `D(fake)` (sin `detach`). ¿Qué pasaría en cada caso si nos olvidamos del `.detach()` en `update_D`? ¿Y qué pasaría si por error agregamos un `.detach()` en `update_G`?
::::


::::cell{#ej5-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

**Si nos olvidamos del `.detach()` en `update_D`:**
Cuando llamamos `loss_D.backward()`, el grafo computacional incluye también el forward del generador (porque `fake = G(Z)` está conectado a `D(fake)`). PyTorch calcula y acumula gradientes en los pesos de `G`. Esos gradientes son **espurios**: `loss_D` está medida desde el punto de vista del discriminador (penaliza a `D` cuando confunde fake con real), así que el gradiente que llega a `G` es exactamente lo opuesto a lo que `G` quiere — empuja a `G` a producir falsos *más fáciles de detectar*. En la siguiente llamada a `update_G`, el `opt_G.step()` opera sobre gradientes contaminados por la suma de la señal correcta más este efecto adverso. El entrenamiento se vuelve inestable o directamente diverge.

(Hay otra consecuencia más sutil: el grafo del forward de `G` se mantiene "vivo" más tiempo del necesario, consumiendo memoria. En batches grandes esto puede causar OOM.)

**Si por error agregamos un `.detach()` en `update_G`:**
`fake.detach()` corta el grafo: `D(fake.detach())` calcula la pérdida pero el gradiente no puede volver a `G`. `loss_G.backward()` solo acumula gradientes en los pesos de `D` (que `opt_G.step()` ignora). El generador recibe **gradiente cero** y nunca se mueve. El entrenamiento queda con `G` congelado mientras `D` sigue mejorando, y termina en una situación trivial donde `D` separa perfectamente reales de falsos (que son siempre los mismos, porque `G` no aprende).

En resumen, el `.detach()` solo va en el paso de `D`. Es un detalle chico de implementación con consecuencia grande: cualquiera de los dos errores rompe el entrenamiento.
```
::::


::::cell{#secF type=markdown role=section}
---
## Sección F: Entrenamiento
::::


::::cell{#intro-secF type=markdown role=intro}
Con el generador, el discriminador y las funciones de actualización ya definidas, lo único que falta es el **loop de entrenamiento**: la función que recorre épocas y batches, llama a `update_D` y `update_G` en cada paso, y registra la pérdida promedio por época para poder graficar al final.

A diferencia de las redes supervisadas que viste antes, acá hay **dos optimizadores**: uno solo para los parámetros del discriminador (`opt_D`) y otro solo para los del generador (`opt_G`). Cada uno actualiza los pesos de su propia red en su correspondiente paso. Para el optimizador `Adam`, la receta DCGAN recomienda `lr = 2e-4` y `betas = (0.5, 0.999)` — el `beta1` bajo evita que el momentum amplifique las oscilaciones propias del adversario.

> **Tiempos esperados:** con GPU, **200 épocas** sobre este dataset toman entre 1 y 2 minutos. Si el dispositivo detectado es `cpu`, el entrenamiento puede llevar varias horas — volvé al menú de Colab y activá la GPU antes de continuar.

> **Cómo leer las curvas de una GAN:** en una GAN **las pérdidas no bajan monótonamente**. El equilibrio sano es una "danza" donde las dos curvas oscilan dentro de un rango razonable: si `D` tiende a `0` el discriminador está ganando todo el tiempo y `G` no aprende; si `G` tiende a `0` y `D` se dispara, suele ser señal de mode collapse. Lo que querés ver es a las dos curvas **moverse cerca de `log(2) ≈ 0.69`** (el valor de equilibrio teórico para `BCE` con dos clases balanceadas), con oscilaciones pero sin dispararse.
::::


::::cell{#ej6-enunciado type=markdown role=enunciado}
### Ejercicio 6 — Loop de entrenamiento y ejecución

**Objetivo:** Implementar el loop de entrenamiento adversario, instanciar las redes y lanzar el entrenamiento.

**Enunciado:**

**Parte A — Implementar `train_gan(G, D, dataloader, epochs, lr, latent_dim)`:**

1. Crear los dos optimizadores Adam:
   - `opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))`
   - `opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))`
2. Poner las dos redes en modo entrenamiento: `G.train()`, `D.train()`.
3. Inicializar dos listas vacías para guardar la historia de pérdidas: `d_hist = []`, `g_hist = []`.
4. Por cada época en `range(1, epochs + 1)`:
   - Inicializar acumuladores `d_losses = []`, `g_losses = []`.
   - Por cada `real` en el `dataloader`:
     - Mover `real` al `device`.
     - Muestrear ruido latente: `Z = torch.randn(B, latent_dim, 1, 4, device=device)`, donde `B = real.size(0)`.
     - Llamar a `update_D(real, Z, G, D, opt_D)` y a `update_G(Z, G, D, opt_G)` y guardar las pérdidas en los acumuladores con `.item()`.
   - Al terminar la época, agregar el promedio de cada acumulador a `d_hist` y `g_hist`.
   - Imprimir el progreso cada 20 épocas (y en la primera).
5. Devolver `d_hist, g_hist`.

**Parte B — Ejecutar:**

1. Instanciar `G = Generator(vocab_size=VOCAB_SIZE).to(device)` y `D = Discriminator(vocab_size=VOCAB_SIZE).to(device)`.
2. Llamar a `train_gan(G, D, data_iter, epochs=200, lr=2e-4, latent_dim=100)` y guardar las dos listas en `d_hist`, `g_hist`.
3. Graficar las curvas con `graficar_perdidas(d_hist, g_hist)`.

> **Pista:** la "coreografía" de cada batch es siempre la misma — un paso de `D`, un paso de `G`. Las dos funciones `update_*` se ocupan internamente del `zero_grad`, del `backward` y del `step`, así que en `train_gan` solo las llamás y guardás el valor que devuelven.
::::


::::cell{#ej6-code-a type=code role=student-code}
```python
def train_gan(G, D, dataloader, epochs=200, lr=2e-4, latent_dim=100):
    # Tu código aquí
    pass
```

```python solution
def train_gan(G, D, dataloader, epochs=200, lr=2e-4, latent_dim=100):
    # ─── Optimizadores Adam con betas DCGAN ─────────────────────────────────
    # beta1=0.5 (en lugar del 0.9 de Adam por defecto) atenúa el momentum y
    # evita que las oscilaciones del juego adversario se amplifiquen.
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    G.train(); D.train()
    d_hist, g_hist = [], []

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        for real in dataloader:
            real = real.to(device)
            B = real.size(0)
            # ─── Ruido latente con la forma rectangular que espera G ────────
            # G está diseñado para tomar un tensor (B, 100, 1, 4) y expandirlo
            # progresivamente hasta (B, 13, 14, 32). La forma del ruido define
            # la forma de la salida.
            Z = torch.randn(B, latent_dim, 1, 4, device=device)

            d_loss = update_D(real, Z, G, D, opt_D)
            g_loss = update_G(Z, G, D, opt_G)

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        d_hist.append(float(np.mean(d_losses)))
        g_hist.append(float(np.mean(g_losses)))

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"D loss: {d_hist[-1]:.4f} | G loss: {g_hist[-1]:.4f}")

    return d_hist, g_hist
```
::::


::::cell{#ej6-code-b type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Instanciación y entrenamiento ──────────────────────────────────────────
G = Generator(vocab_size=VOCAB_SIZE).to(device)
D = Discriminator(vocab_size=VOCAB_SIZE).to(device)

d_hist, g_hist = train_gan(G, D, data_iter, epochs=200, lr=2e-4, latent_dim=100)

graficar_perdidas(d_hist, g_hist)
```
::::


::::cell{#ej6-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirá las curvas de pérdida del entrenamiento. A diferencia de una red supervisada normal, las dos curvas no bajan monótonamente — oscilan, suben, bajan.

1. ¿Qué patrón general ves en las dos curvas? ¿Se acercan a algún valor estable?
2. Si en algún momento del entrenamiento vieras que la pérdida del discriminador `D loss` cae rápidamente cerca de `0` y se queda ahí, ¿qué estaría pasando con el equilibrio adversario? ¿Qué efecto tendría sobre el aprendizaje del generador?
::::


::::cell{#ej6-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

1. **Patrón general.**
   Las dos curvas oscilan, no bajan monótonamente. En las primeras épocas suele haber una fase de transitorio donde `D loss` está alta y `G loss` baja (el discriminador todavía no distingue, el generador "engaña" trivialmente). A medida que `D` aprende, `D loss` baja un poco y `G loss` sube — empieza el balance adversario. En régimen estable las dos curvas se mantienen oscilando alrededor de valores cercanos a `log(2) ≈ 0.69` (valor de equilibrio teórico cuando `D` está cerca de `0.5` para cualquier entrada). Es esperable que sigan oscilando hasta el final del entrenamiento: en GANs no existe el concepto de "convergencia" como en regresión.

2. **Si `D loss` colapsa cerca de `0`.**
   Eso significa que el discriminador clasifica casi perfectamente entre reales y falsos. Cuando eso pasa, los logits de `D(fake)` son muy negativos (muy seguro de que son falsos), y `sigmoid(logit) ≈ 0`. La pérdida del generador `BCE(logit, y=1)` sigue dando un número finito, pero el **gradiente** que llega a `G` es muy chico — la función sigmoide está saturada en su parte plana. En la práctica, el generador deja de aprender: no recibe señal útil para mejorar sus salidas. Es lo que se llama el **problema del gradiente desvaneciente del generador**.

   Hay varias técnicas para evitarlo (label smoothing, mejorar la regularización del discriminador, usar pérdidas alternativas como Wasserstein), pero todas apuntan a lo mismo: **mantener al discriminador imperfecto**. Una GAN con un discriminador "demasiado bueno" no entrena.
```
::::


::::cell{#secG type=markdown role=section}
---
## Sección G: Generación e interpolación latente
::::


::::cell{#intro-secG type=markdown role=intro}
Con el generador entrenado, ya podemos hacer dos cosas:

- **Muestrear nuevos niveles**: tomamos vectores latentes `z` aleatorios del prior `N(0, I)`, los pasamos por `G`, y obtenemos segmentos sintéticos.
- **Interpolar entre dos niveles**: tomamos dos vectores latentes `z_1` y `z_2`, generamos una secuencia de vectores intermedios `z_t = (1 - t) z_1 + t z_2` con `t ∈ [0, 1]`, y decodificamos toda la secuencia. Si el espacio latente es **continuo** (lo cual no es para nada obvio en una GAN — no hay un término KL forzándolo, como en un VAE), la trayectoria visual debería transicionar de manera **suave** entre los dos extremos.

La interpolación es la prueba cualitativa más interesante de una GAN: muestra qué aprendió el generador sobre la **estructura del espacio latente**, más allá de la calidad puntual de cada muestra.
::::


::::cell{#ej7-enunciado type=markdown role=enunciado}
### Ejercicio 7 — Generación de niveles e interpolación latente

**Objetivo:** Usar el generador entrenado para producir niveles nuevos y para interpolar entre dos puntos del espacio latente.

**Enunciado:**

**Parte A — Muestras del prior.**

1. Poné el generador en modo evaluación: `G.eval()`.
2. Sin construir el grafo de gradientes (`with torch.no_grad():`), generá **4 niveles** desde ruido aleatorio: `z = torch.randn(4, 100, 1, 4, device=device)` y `samples = G(z)`.
3. Mostralos con `mostrar_segmentos(list(samples.cpu()), sprites)`.

**Parte B — Interpolación entre dos niveles.**

1. Generá **dos vectores latentes** distintos `z_a` y `z_b`, cada uno con forma `(1, 100, 1, 4)`.
2. Construí una **trayectoria de 8 pasos** entre los dos: para cada `t` en `torch.linspace(0, 1, 8)`, calculá `z_t = (1 - t) * z_a + t * z_b` y juntalos en un único tensor de forma `(8, 100, 1, 4)`.
3. Decodificá toda la trayectoria de una sola pasada por el generador.
4. Mostrá la secuencia con `mostrar_segmentos(list(traj.cpu()), sprites)`.

> **Pista:** para construir la trayectoria de una sola vez, podés usar `torch.linspace(0, 1, 8).view(-1, 1, 1, 1).to(device)` y aprovechar broadcasting: `(1 - alphas) * z_a + alphas * z_b`.
::::


::::cell{#ej7-code-a type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Parte A: cuatro niveles muestreados del prior ──────────────────────────
G.eval()
with torch.no_grad():
    z = torch.randn(4, 100, 1, 4, device=device)
    samples = G(z)

mostrar_segmentos(list(samples.cpu()), sprites,
                  titulos=[f"Muestra {i}" for i in range(4)])
```
::::


::::cell{#ej7-code-b type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Parte B: trayectoria latente entre dos niveles ─────────────────────────
# Tomamos dos puntos cualesquiera del espacio latente y generamos una
# trayectoria lineal de 8 pasos entre ellos. Si el espacio latente es
# continuo, los segmentos intermedios son híbridos plausibles entre los
# extremos.
G.eval()
with torch.no_grad():
    z_a = torch.randn(1, 100, 1, 4, device=device)
    z_b = torch.randn(1, 100, 1, 4, device=device)

    alphas = torch.linspace(0, 1, 8, device=device).view(-1, 1, 1, 1)
    z_traj = (1 - alphas) * z_a + alphas * z_b   # broadcasting → (8, 100, 1, 4)

    traj = G(z_traj)

mostrar_segmentos(list(traj.cpu()), sprites,
                  titulos=[f"t = {a.item():.2f}" for a in alphas[:, 0, 0, 0]])
```
::::


::::cell{#ej7-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirá la secuencia de la interpolación. La transición entre `t=0` y `t=1` puede ser **suave** (los elementos del nivel se transforman gradualmente: aparecen tubos donde antes había bloques, se mueve la línea de suelo, etc.) o **abrupta** (los segmentos intermedios se ven distorsionados o no parecen niveles plausibles).

1. ¿Qué tan suave fue tu trayectoria? ¿Las muestras intermedias parecen niveles válidos o aparecen artefactos visuales?
2. A diferencia de un VAE, **una GAN no tiene ningún término en la pérdida que la fuerce a tener un espacio latente continuo**. ¿Por qué entonces podemos esperar (y a menudo observar) interpolaciones razonables? ¿Qué propiedad del entrenamiento adversario contribuye a esto?
::::


::::cell{#ej7-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

1. **Suavidad observada.**
   En general la trayectoria es razonablemente suave: los elementos del nivel se modifican gradualmente entre los dos extremos. Pueden aparecer artefactos (transiciones donde un tubo "se rompe" por unas columnas, regiones donde el suelo desaparece o aparece de a saltos), sobre todo si el dataset es chico y el modelo no exploró todo el espacio latente. La calidad de la interpolación es heterogénea: algunos pares `z_a, z_b` dan transiciones limpias y otros producen un par de fotogramas raros en el medio.

2. **Por qué interpola razonable sin un término KL.**
   El generador es una red neuronal **continua y diferenciable** desde su entrada (el latente) hasta su salida. Si dos puntos `z_a, z_b` están razonablemente cerca en el espacio latente, sus salidas `G(z_a)` y `G(z_b)` también van a estar cerca en el espacio de pixeles (continuidad de la función). Esa continuidad es matemática, no entrenada: vale para cualquier red neuronal con activaciones continuas.

   Lo que el entrenamiento adversario aporta es algo más sutil: el discriminador presiona al generador a que **toda la región del prior produzca salidas plausibles** (cualquier `z` que el discriminador encuentre raro lo penaliza). Esa presión hace que `G` no tenga grandes "desiertos" en el latente donde produce basura. Como la trayectoria interpolada vive enteramente dentro del soporte del prior `N(0, I)`, casi todos sus puntos están en la región donde `G` aprendió a producir niveles plausibles.

   Sin embargo, a diferencia del VAE, no hay ninguna garantía de que distancias en el latente reflejen distancias semánticas en el espacio de salida — y por eso las GANs tienden a tener interpolaciones "menos suaves" que los VAEs entrenados sobre el mismo dataset. La contraparte es que cada muestra individual es más nítida.
```
::::


::::cell{#caminos type=markdown role=info}
---
## Caminos para profundizar

Lo que entrenaste es una **DCGAN vainilla** sobre un dataset chico. Funciona y produce niveles plausibles, pero cualquiera que se ponga a generar contenido de juegos con GANs sabe que se puede ir bastante más lejos. Si te quedaste con ganas de explorar, estos son los caminos más productivos, ordenados por relación impacto/complejidad:

**1. Mejor pérdida adversaria.** Vanilla DCGAN con BCE es la versión "histórica" de las GANs. Las alternativas modernas — **WGAN-GP** (*Wasserstein con gradient penalty*) y **hinge loss** — son notablemente más estables, casi eliminan el mode collapse y mantienen los gradientes vivos durante todo el entrenamiento. Cambian dos o tres líneas del código de pérdidas. WGAN-GP es lo que usaba el paper original de Volz et al. (2018) sobre Mario.

**2. Regularización del discriminador.** Envolver cada `Conv2d` del discriminador con `nn.utils.spectral_norm()` agrega una restricción de Lipschitz que estabiliza el entrenamiento. Combina muy bien con hinge loss. Cambio mínimo, mejora consistente.

**3. Self-attention layers (SAGAN).** Insertar una capa de atención entre dos bloques convolucionales (en `G` y en `D`) le da al modelo capacidad de mirar **estructura global**, no solo vecindad local. Para Mario eso se traduce en suelos más continuos, tubos completos en lugar de "rotos a mitad de columna" y plataformas que respetan distancias jugables.

**4. Two-Time Scale Update Rule (TTUR).** En lugar de un mismo `lr` para `G` y `D`, usar tasas distintas — típicamente `lr_D ≈ 4·lr_G` — estabiliza el balance adversario. Un cambio de un parámetro.

**5. Búsqueda en el espacio latente (post-training).** Una vez entrenada la GAN, podés usar un optimizador como **CMA-ES** o gradient descent para encontrar `z` que maximicen una métrica de interés: cantidad de monedas, dificultad estimada, densidad de enemigos, jugabilidad (medida con un solver simple). Es lo que convierte a una GAN en un *generador controlable*. La idea aparece en Volz et al. y en gran parte de la literatura de PCGML (*procedural content generation via machine learning*).

**6. Output discreto correctamente.** El truco `tanh + argmax` que usamos es una aproximación: durante el entrenamiento el discriminador ve salidas continuas en `[-1, +1]`, no las elecciones discretas finales. La forma "matemáticamente honesta" de generar discretos con gradientes es la **Gumbel-softmax con straight-through estimator**. En la práctica para Mario el truco funciona casi tan bien, pero conocer la versión correcta es útil para extender el método a otros dominios discretos (texto, secuencias).

**7. Reformulación: VAE.** Si lo que más te interesa de un modelo generativo es el **espacio latente continuo y muestreable**, los autoencoders variacionales lo logran de manera más directa que las GANs (ver el Laboratorio 5b). Para problemas como interpolación entre dos niveles, un VAE típicamente da resultados más suaves; para nitidez puntual de cada muestra, la GAN gana.

**Lectura recomendada:** [Volz et al. 2018, "Evolving Mario Levels in the Latent Space of a Deep Convolutional GAN"](https://arxiv.org/abs/1805.00728) — el paper de referencia para GANs sobre Mario. Y [Radford et al. 2015, "Unsupervised Representation Learning with Deep Convolutional GANs"](https://arxiv.org/abs/1511.06434) — el paper de DCGAN, todavía la receta empírica de base que cualquier extensión termina respetando.
::::


::::cell{#checklist type=markdown role=checklist}
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Ejecuté el laboratorio con la **GPU activa** (`device == "cuda"`).
- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] El entrenamiento corrió las 200 épocas y las dos curvas de pérdida oscilan dentro de un rango razonable (ninguna se dispara ni colapsa a 0).
- [ ] Los niveles generados en el Ej 7A se ven como segmentos plausibles de Mario (con piso, bloques o tubos reconocibles).
- [ ] La interpolación del Ej 7B muestra una transición visualmente coherente entre los dos extremos.
- [ ] No modifiqué ninguna celda fuera de las de actividad.
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Construiste una GAN de punta a punta: bloques convolucionales para subir y bajar resolución, generador, discriminador, las dos funciones de actualización adversaria con el detalle clave del `.detach()`, el loop de entrenamiento, y dos formas de explorar el espacio latente aprendido (muestreo del prior e interpolación entre puntos).

Lo más importante del laboratorio es probablemente **el flujo de gradientes en el entrenamiento adversario** — la disciplina de qué red recibe gradiente en qué paso, y la coreografía con `detach()`. Esa idea aparece en muchos otros lugares (entrenamientos auto-supervisados, modelos con teacher fijo, distillation), no solo en GANs.

En la próxima parte de este bloque vas a ver cómo los **autoencoders variacionales** abordan el mismo problema (modelado generativo de imágenes) desde un ángulo distinto: en lugar de entrenar dos redes que compiten, entrenan una sola red que aprende a comprimir y descomprimir, con un término de regularización explícito sobre el espacio latente.
::::
