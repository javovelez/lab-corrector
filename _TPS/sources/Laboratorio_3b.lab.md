---
lab: "3b"
title: "Laboratorio n° 3. Parte B: Transferencia de estilos rápida"
subject: "Redes Neuronales Profundas"
block: "3 — Transferencia de conocimiento"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 3b — Transferencia de estilos rápida
     Fuente única. Compilar con:
         python tools/lab_build.py _TPS/sources/Laboratorio_3b.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_3b.ipynb
             _TPS/Soluciones/Laboratorio_3b_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 3. Parte B: Transferencia de estilos rápida

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 3 — Transferencia de conocimiento

---

## Introducción

En la clase vimos que una red convolucional preentrenada no solo sirve para clasificar: sus capas intermedias son **extractores de features** reutilizables para tareas muy distintas a la original. Una de las aplicaciones más vistosas de esta idea es la **transferencia de estilos**: combinar el *contenido* de una foto con el *estilo* de una pintura en una sola imagen sintetizada.

Hay dos familias de métodos para hacerlo:

- **Gatys et al. 2015** — optimizar directamente los píxeles de una imagen sintetizada, usando una VGG preentrenada para medir la distancia de contenido y de estilo. Es el método que estudiamos en la teoría. Es flexible (cualquier par contenido/estilo) pero **lento**: cada imagen nueva requiere muchas de iteraciones de descenso de gradiente.
- **Johnson et al. 2016** — entrenar **una red feed-forward** (TransformerNet) especializada en un único estilo. Una vez entrenada, estilizar una imagen nueva es un solo forward pass: *tiempo real*. El precio: una red entrenada por estilo.

En este laboratorio vas a implementar el método de Johnson. Vas a:

- Construir las piezas que ambos métodos comparten: matriz de Gram, pérdida de contenido y pérdida de estilo sobre features de VGG.
- Ensamblar una **TransformerNet** encoder-decoder con bloques residuales.
- Entrenarla sobre COCO para que aprenda a reproducir el estilo de una única imagen.
- Usarla para estilizar imágenes nuevas y analizar el resultado.

### Cómo se entrena la red: vista general

Antes de entrar en las piezas conviene tener el mapa del entrenamiento. En cada iteración pasa lo siguiente:

1. Tomamos una imagen cualquiera del dataset (una foto de COCO, sin importar qué contiene) y la usamos como **imagen de contenido**.
2. La pasamos por la **TransformerNet** y obtenemos la **imagen generada** — la versión estilizada que produce la red con sus pesos actuales.
3. Medimos dos cosas con ayuda de una VGG preentrenada (que nunca se entrena, solo mide distancias perceptuales):
   - **Pérdida de contenido**: cuánto se parece la generada a la imagen de contenido *en estructura* (features de una capa intermedia de VGG).
   - **Pérdida de estilo**: cuánto se parece la generada a la **imagen de estilo** —fija durante todo el entrenamiento— *en textura y color* (matrices de Gram sobre features de varias capas).
4. Sumamos las dos pérdidas con pesos, backpropagamos y actualizamos los pesos de la TransformerNet. La VGG queda intacta: es la regla de medición, no el modelo que aprende.
5. Repetimos sobre miles de imágenes. Como la imagen de estilo es siempre la misma pero la de contenido cambia todo el tiempo, la red termina aprendiendo una *función* que aplica ese estilo concreto a **cualquier** foto — no memoriza un par contenido/estilo puntual.

Las piezas que implementás en el lab encajan en ese esquema: las pérdidas perceptuales (Sección A), la red generadora (Sección B), y el loop que las engancha (Sección C). Si en algún momento te perdés en los detalles, volvé a este mapa.

> **Nota sobre la VGG — convención de capas respecto de la teoría:**
>
> En la teoría trabajamos con el método de Gatys sobre **VGG19**, seleccionando las capas por su índice numérico dentro de `vgg.features` (por ejemplo `[0, 5, 10, 19, 28]` para estilo y `[25]` para contenido) y quedándonos con la salida **pre-activación**, es decir, el resultado de la convolución antes de aplicar la no-linealidad ReLU.
>
> En este laboratorio seguimos la convención del paper de Johnson, que usa **VGG16** y extrae la salida **post-activación** (después de la ReLU) de la última capa ReLU de los primeros cuatro bloques convolucionales. Por eso las features aparecen nombradas como `relu1_2`, `relu2_2`, `relu3_3` y `relu4_3`: el nombre `reluN_M` indica "salida del M-ésimo ReLU del bloque N".
>
> La diferencia más relevante frente a la teoría es qué capa se usa para **contenido**: Gatys toma una capa profunda (`conv4_4`, índice 25 en VGG19), mientras que Johnson toma una capa temprana (`relu2_2`). La razón es práctica: al tratarse de un método feedforward que entrena una red generadora, una capa de contenido temprana obliga a la TransformerNet a preservar la estructura espacial de la foto de entrada con fidelidad. Una capa de contenido profunda deja demasiada libertad al generador y produce salidas inestables o muy distorsionadas.

> **Importante — GPU:** este laboratorio entrena una red convolucional sobre ~5.000 imágenes. **Activá la GPU en Colab** antes de empezar: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4)*. Sin GPU el entrenamiento es impracticable (horas en lugar de minutos).

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para resolver cada ejercicio, consultá el material teórico de la clase (en particular la sección de transferencia de estilos).
::::


::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::


::::cell{#imports type=code role=setup}
```python
# ─── Setup: imports y detección de GPU ──────────────────────────────────────
# Estos son los imports que usa el laboratorio de punta a punta:
#   - torch / torchvision: red preentrenada (VGG-16), capas, optimizadores y
#     transformaciones de imágenes.
#   - cv2 / PIL / matplotlib: cargar, convertir y mostrar imágenes.
#   - os / urllib: descargar dataset (COCO val2017) e imagen de estilo.
# También detectamos si hay GPU disponible y la guardamos en `device`: esa
# variable la usás en todos los `.to(device)` del laboratorio.
import os
import random
import time
import urllib.request

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()        else
    "cpu"
)
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Dispositivo:        {device}")
if device == "cpu":
    print("ADVERTENCIA: sin GPU el entrenamiento va a ser inviable. "
          "Activá la GPU en Colab (T4) o corré local con CUDA/MPS antes de continuar.")
```
::::


::::cell{#setup-coco type=code role=setup}
```python
# ─── Setup: descarga del dataset de contenido (COCO val2017) ────────────────
# COCO (Common Objects in Context, Microsoft) es un dataset estándar de
# imágenes naturales con anotaciones de detección y segmentación. Acá lo
# usamos solo como fuente variada de imágenes de contenido (ignoramos las
# anotaciones): la TransformerNet necesita ver muchas fotos distintas para
# aprender a aplicar un estilo sobre "cualquier" imagen.
# Usamos el split val2017 (~5.000 imágenes, ~780 MB) en lugar de train2014
# (13 GB) del paper original: alcanza para que la red aprenda un estilo
# razonable en una sola pasada de ~5 minutos sobre T4.
os.makedirs('/content/dataset', exist_ok=True)
if not os.path.exists('/content/dataset/val2017'):
    !wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip -O /content/val2017.zip
    !unzip -q /content/val2017.zip -d /content/dataset/
    !rm /content/val2017.zip
else:
    print('Dataset ya descargado')
print(f'Imágenes disponibles: {len(os.listdir("/content/dataset/val2017"))}')
```
::::


::::cell{#setup-style type=code role=setup}
```python
# ─── Setup: imagen de estilo ────────────────────────────────────────────────
# Por defecto bajamos una imagen de estilo clásica (mosaic.jpg) desde el repo
# oficial pytorch/examples. Podés cambiar STYLE_NAME a 'candy', 'rain_princess'
# o 'udnie' para probar otros estilos.
STYLES = {
    'mosaic':        'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/mosaic.jpg',
    'candy':         'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/candy.jpg',
    'rain_princess': 'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/rain-princess.jpg',
    'udnie':         'https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/udnie.jpg',
}

STYLE_NAME = 'mosaic'
style_path = f'/content/style_images/{STYLE_NAME}.jpg'
os.makedirs('/content/style_images', exist_ok=True)

if not os.path.exists(style_path):
    req = urllib.request.Request(STYLES[STYLE_NAME],
                                 headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as r, open(style_path, 'wb') as f:
        f.write(r.read())

plt.figure(figsize=(8, 5))
plt.imshow(Image.open(style_path))
plt.title(f'Imagen de estilo: {STYLE_NAME}')
plt.axis('off')
plt.show()
```
::::


::::cell{#setup-helpers type=code role=setup}
```python
# ─── Setup: helpers de imagen ───────────────────────────────────────────────
# Funciones auxiliares para leer imágenes en BGR (formato de cv2, coincide
# con el preprocesamiento Caffe que espera la VGG que usamos más abajo),
# convertirlas a tensores en rango [0, 255] con dimensión de batch, y volver.
def load_image(path):
    """Devuelve una imagen BGR (numpy, shape H, W, 3)."""
    return cv2.imread(path)


def itot(img, max_size=None):
    """Convierte una imagen numpy BGR en un tensor (1, 3, H, W) en [0, 255]."""
    if max_size is None:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ])
    else:
        H, W, _ = img.shape
        size = tuple(int((float(max_size) / max(H, W)) * x) for x in [H, W])
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ])
    return t(img).unsqueeze(0)


def ttoi(tensor):
    """Convierte un tensor (1, 3, H, W) en un array numpy (H, W, 3)."""
    return tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     SECCIÓN A: Fundamentos compartidos (Gatys y Johnson)
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#secA type=markdown role=section}
---
## Sección A: Fundamentos compartidos (Gatys y Johnson)
::::


::::cell{#intro-secA type=markdown role=intro}
### Una misma idea para dos métodos

Tanto Gatys como Johnson usan una VGG preentrenada para calcular dos pérdidas:

- **Pérdida de contenido**: MSE entre los *feature maps* de una capa intermedia de la imagen generada y de la imagen de contenido. Mide cuánto se parecen **qué cosas hay** y **dónde están**.
- **Pérdida de estilo**: MSE entre las **matrices de Gram** de los feature maps en varias capas. La matriz de Gram captura **correlaciones entre canales** e ignora la posición espacial — justamente lo que asociamos con "estilo".

En esta sección vas a implementar esas dos piezas y después las vas a reutilizar en el loop de entrenamiento del método de Johnson. Son las mismas funciones que usarías en el método de Gatys.
::::


::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Matriz de Gram

**Objetivo:** Implementar la operación central para medir estilo: la matriz de Gram de un tensor de feature maps.

**Contexto:**

Dado un tensor de activaciones de forma `(B, C, H, W)` (lote, canales, alto, ancho), la matriz de Gram se obtiene aplanando el eje espacial y haciendo el producto externo entre canales:

$$G = \frac{X \, X^T}{C \cdot H \cdot W}$$

donde $X$ es el tensor al que se aplica `reshape` para llevarlo a la forma `(B, C, H*W)`. El resultado tiene forma `(B, C, C)`: el elemento `G[b, i, j]` es la correlación entre los canales $i$ y $j$ en el ejemplo $b$. Dividimos por `C*H*W` para que la escala no dependa del tamaño de entrada.

**Generalización a un lote de tamaño `B`:**

En la teoría viste una versión de `gram` pensada para una sola imagen: el tensor se lleva con `reshape` a una matriz 2D de forma `(C, H*W)` y se multiplica por su traspuesta con `X.T`. En este laboratorio necesitamos la misma operación pero sobre un **lote de tamaño `B`**, porque durante el entrenamiento de Johnson (Sección B) procesamos varias imágenes a la vez y `gram` se aplica sobre features con `B > 1`. La adaptación es mínima, pero como el tensor pasa a ser 3D hay dos detalles a cuidar al trasponer y al multiplicar.

**Enunciado:**

1. Implementá la función `gram(X)` que reciba un tensor de forma `(B, C, H, W)` y devuelva su matriz de Gram normalizada de forma `(B, C, C)`.
2. Probá la función con un tensor aleatorio de forma `(2, 4, 8, 8)` e imprimí:
   - La forma del resultado.
   - Si la matriz resultante es simétrica (una matriz de Gram siempre lo es).

> **Pistas:**
>
> - Aplicá `reshape` para llevar el tensor a la forma `(B, C, H*W)`, sin tocar el eje del lote. Cada canal queda como un vector de largo `H*W`.
> - Para el producto `X · X^T` por cada elemento del lote usá `torch.matmul`: sobre tensores 3D se comporta como una multiplicación por lote, es decir, para cada `b` hace el producto matricial de `X[b]` con su traspuesta.
> - Para obtener la traspuesta dentro de cada elemento del lote usá `X.transpose(1, 2)`: intercambia **filas (eje 1 = canales)** por **columnas (eje 2 = posiciones espaciales)** y deja el eje del lote intacto. `X.T` no sirve acá porque en un tensor 3D invertiría los tres ejes.
> - Para chequear simetría: `torch.allclose(G, G.transpose(1, 2))`.
::::


::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Implementación de la matriz de Gram ─────────────────────────────────────
def gram(X):
    """
    Calcula la matriz de Gram de un tensor de feature maps.

    Parámetros:
    X (Tensor): activaciones de forma (B, C, H, W).

    Retorna:
    G (Tensor): matriz de Gram normalizada, shape (B, C, C).
    """
    # Misma receta que en la teoría, generalizada a un lote de tamaño B.
    B, C = X.shape[0], X.shape[1]
    n = X.numel() // (B * C)           # n = H*W (producto de los ejes espaciales)
    # Aplicamos reshape para aplanar el eje espacial sin tocar el eje del lote:
    # (B, C, H, W) → (B, C, n). Cada canal queda como un vector de largo n.
    X = X.reshape((B, C, n))
    # transpose(1, 2) intercambia filas (eje 1 = canales) por columnas
    # (eje 2 = posiciones espaciales) dentro de cada elemento del lote.
    # No se puede usar .T en un tensor 3D: invertiría los tres ejes.
    # matmul sobre tensores 3D multiplica por lote (equivale a torch.bmm):
    # para cada b hace X[b] (C, n) · X[b].T (n, C) → matriz C×C de correlaciones
    # entre canales. Dividimos por C*n para que la magnitud no dependa del
    # tamaño espacial.
    return torch.matmul(X, X.transpose(1, 2)) / (C * n)


# ─── Test rápido ─────────────────────────────────────────────────────────────
torch.manual_seed(0)
test = torch.randn(2, 4, 8, 8)
G = gram(test)
print(f"Forma de entrada:  {tuple(test.shape)}")
print(f"Forma de Gram:     {tuple(G.shape)}  (esperado: (2, 4, 4))")
print(f"¿Es simétrica?     {torch.allclose(G, G.transpose(1, 2))}")
```
::::


::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La matriz de Gram se construye justamente colapsando la dimensión espacial (`H, W`) y quedándose con la correlación entre canales. ¿Por qué eso captura "estilo" y descarta "contenido"? Pensalo imaginando qué pasa si tomás una imagen y la desplazás o la rotás un poco: ¿cambia mucho su matriz de Gram? ¿Cambian mucho sus feature maps directos?
::::


::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Los **feature maps directos** guardan información posicional: cada valor `feat[c, i, j]` dice cuánto se activa el filtro `c` **en la posición `(i, j)`**. Si desplazás la imagen 20 píxeles a la derecha, los feature maps se desplazan parecido — la representación cambia mucho aunque la imagen sea, intuitivamente, "la misma cosa corrida".

La **matriz de Gram** promedia el producto de canales sobre todas las posiciones espaciales:

$$G_{ij} \approx \sum_{\mathrm{pos}} \mathrm{feat}_i(\mathrm{pos}) \cdot \mathrm{feat}_j(\mathrm{pos})$$

Al integrar sobre `(H, W)`, pierde la información de **dónde** está cada patrón y se queda con **qué patrones co-ocurren** en la imagen. Esa co-ocurrencia es precisamente lo que asociamos con estilo: "en esta pintura, el filtro que detecta diagonales rojas se activa junto con el que detecta azules planos" — una propiedad global que se mantiene aunque recortes, desplaces o rotes la imagen.

Por eso minimizar MSE entre matrices de Gram transfiere estilo (patrones globales) y minimizar MSE entre feature maps transfiere contenido (patrones locales en la posición correcta).
```
::::


::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Pérdidas de contenido y estilo sobre features de VGG

**Objetivo:** Definir las dos pérdidas perceptuales que usan Gatys y Johnson, operando sobre **dicts de features** extraídos de distintas capas de VGG.

**Contexto:**

Cuando calculamos estas pérdidas, tenemos dict de features `{'relu1_2': ..., 'relu2_2': ..., 'relu3_3': ..., 'relu4_3': ...}` para la imagen generada y para la(s) imagen(es) de referencia. El convenio del paper original es:

- **Contenido**: MSE sobre `relu2_2` (una sola capa, intermedia).
- **Estilo**: suma de MSE entre matrices de Gram sobre las **cuatro** capas.

**Enunciado:**

Al comienzo de la celda instanciá `mse = nn.MSELoss()` para reutilizarlo en las dos pérdidas. Después implementá dos funciones (`gram` ya está disponible del ejercicio 1):

1. `content_loss(feats_gen, feats_content)`: MSE entre `feats_gen['relu2_2']` y `feats_content['relu2_2']`.
2. `style_loss(feats_gen, style_gram)`: suma de MSE entre matrices de Gram sobre todas las capas. Los dos argumentos entran a la función en estados distintos:
   - `feats_gen` son las **features crudas** de la imagen generada (una por capa, sin Gram calculada todavía). La imagen generada cambia en cada iteración del entrenamiento, así que su Gram hay que calcularla adentro de la función.
   - `style_gram` son las **matrices de Gram ya calculadas** de la imagen de estilo (una por capa). Se calculan una sola vez fuera de la función, antes de empezar a entrenar, porque la imagen de estilo es fija durante todo el entrenamiento.

   Para cada clave `k` de `feats_gen`, aplicá `gram` a `feats_gen[k]` y acumulá el MSE contra `style_gram[k]`. Devolvé la suma.

Probá las dos funciones con tensores dummy de la forma correcta y verificá que devuelven escalares no negativos.

> **Pista:** para el test dummy podés generar tensores de shape `(4, 64, 32, 32)` y similares — la forma exacta no importa, solo que tenga 4 ejes.
::::


::::cell{#ej2-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Instancia de MSE que vamos a reutilizar ─────────────────────────────────
mse = nn.MSELoss()


def content_loss(feats_gen, feats_content):
    """
    Pérdida de contenido: MSE sobre la capa relu2_2 de la VGG.

    Parámetros:
    feats_gen (dict): features de la imagen generada (una entrada por capa).
    feats_content (dict): features de la imagen de contenido.

    Retorna:
    loss (Tensor escalar): MSE entre relu2_2 generado y relu2_2 contenido.
    """
    return mse(feats_gen['relu2_2'], feats_content['relu2_2'])


def style_loss(feats_gen, style_gram):
    """
    Pérdida de estilo: suma de MSE entre matrices de Gram sobre todas las
    capas presentes en feats_gen.

    Parámetros:
    feats_gen (dict): features de la imagen generada (una entrada por capa).
    style_gram (dict): matrices de Gram precomputadas de la imagen de estilo.

    Retorna:
    loss (Tensor escalar): suma de MSEs capa a capa.
    """
    total = 0.0
    for k, v in feats_gen.items():
        # Gram del generado — se recalcula en cada iteración porque el generado
        # cambia; la del estilo viene precomputada.
        total = total + mse(gram(v), style_gram[k])
    return total


# ─── Test dummy ──────────────────────────────────────────────────────────────
torch.manual_seed(0)
dummy_gen = {
    'relu1_2': torch.randn(4, 64, 64, 64),
    'relu2_2': torch.randn(4, 128, 32, 32),
    'relu3_3': torch.randn(4, 256, 16, 16),
    'relu4_3': torch.randn(4, 512, 8, 8),
}
dummy_content = {k: torch.randn_like(v) for k, v in dummy_gen.items()}
# Precalcular Gram de la "imagen de estilo"
dummy_style_gram = {k: gram(torch.randn_like(v)) for k, v in dummy_gen.items()}

c_loss = content_loss(dummy_gen, dummy_content)
s_loss = style_loss(dummy_gen, dummy_style_gram)
print(f"content_loss: {c_loss.item():.4f}  (esperado: escalar ≥ 0)")
print(f"style_loss:   {s_loss.item():.4f}  (esperado: escalar ≥ 0)")
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Una alternativa ingenua para medir cuánto se parecen dos imágenes es calcular el MSE **píxel a píxel** entre ellas. Sin embargo, ni Gatys ni Johnson usan eso: ambos miden pérdidas sobre **features de VGG** (una red preentrenada para clasificar ImageNet). ¿Por qué? Pensalo en términos de qué querés que la pérdida "considere cercano" y qué no.
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

El MSE píxel a píxel mide **diferencia de intensidad en la misma coordenada**. Con eso, dos fotos idénticas desplazadas un píxel tendrían un MSE altísimo, mientras que dos imágenes completamente distintas pero con la misma intensidad media tendrían MSE bajo. Es una métrica que ignora la semántica.

Las features de VGG son distintas: cada capa detecta patrones cada vez más abstractos (bordes → texturas → partes → objetos). El MSE sobre `relu2_2` dice "se parecen en qué patrones medianos aparecen y dónde" — tolera pequeñas variaciones de apariencia mientras preserve la estructura, que es exactamente lo que entendemos por "mismo contenido".

Para estilo el argumento es análogo pero más fuerte: la matriz de Gram sobre features de VGG captura correlaciones de alto nivel entre canales semánticamente significativos (el canal "diagonales" vs. el canal "superficie azul"), mientras que la Gram píxel a píxel serían apenas estadísticas de color. Por eso las "perceptual losses" (pérdidas perceptuales) — así se llama la familia — funcionan y las pixel losses no.
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     SECCIÓN B: TransformerNet
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#secB type=markdown role=section}
---
## Sección B: TransformerNet — la red que transforma imágenes
::::


::::cell{#intro-secB type=markdown role=intro}
### Una red feed-forward para estilizar

En el método de Gatys, cada imagen nueva requiere un proceso de optimización por descenso de gradiente sobre sus píxeles (cientos de iteraciones). Johnson propone reemplazar ese proceso por una **única red** que aprenda a hacer la transformación en un solo forward pass.

La arquitectura elegida es una CNN encoder-decoder con **bloques residuales** en el medio:

- **Encoder**: 3 convoluciones que reducen la resolución 4× y suben los canales (3 → 32 → 64 → 128).
- **5 bloques residuales** de 128 canales: aprenden *correcciones estilísticas* sin destruir la estructura del contenido (los skip connections garantizan que la identidad sea una solución trivial).
- **Decoder**: 2 deconvoluciones (upsampling) + 1 convolución final que restauran la resolución original y vuelven a 3 canales.

> **Aclaración importante sobre el nombre:** la clase se llama `TransformerNetwork`, pero **no tiene nada que ver con la arquitectura Transformer** (la de *Attention is All You Need*, Vaswani et al. 2017) que probablemente conozcan de modelos de lenguaje o ViT. Acá "transformer" se usa en su sentido literal del inglés —"algo que transforma"— porque esta red **transforma** una imagen de contenido en una imagen estilizada. No hay atención, no hay tokens, no hay positional encodings: es una CNN convencional (encoder + bloques residuales + decoder) entrenada end-to-end.
>
> La coincidencia de nombres es puramente histórica: Johnson publica este trabajo en 2016, un año **antes** que Vaswani. Cuando años después "Transformer" pasó a ser sinónimo de arquitectura con self-attention, el nombre original de Johnson quedó como un falso amigo. En algunos repositorios se lo renombra a `StyleTransferNet` o `ImageTransformNet` para evitar la confusión, pero mantenemos `TransformerNetwork` porque es el nombre que vas a encontrar en el paper y en la mayoría de las implementaciones de referencia.
::::


::::cell{#setup-layers type=code role=setup}
```python
# ─── Setup: bloques de construcción de la TransformerNet ───────────────────
# Las tres clases de abajo son piezas estándar y no es parte del objetivo
# pedagógico que las escribas. Leé los docstrings para saber qué hace cada una
# — las vas a usar en el ejercicio 3 para ensamblar la red completa.
class ConvLayer(nn.Module):
    """
    Convolución con reflection padding + InstanceNorm (o BatchNorm, o sin norm).
    El reflection padding evita los bordes negros que genera el padding cero
    convencional; la instance norm estabiliza el entrenamiento de redes de
    generación de imágenes (Ulyanov et al. 2016).
    """
    def __init__(self, in_c, out_c, kernel, stride, norm='instance'):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel // 2)
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride)
        self.norm_type = norm
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_c, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_c, affine=True)

    def forward(self, x):
        x = self.conv(self.reflection_pad(x))
        return x if self.norm_type == 'None' else self.norm(x)


class ResidualLayer(nn.Module):
    """
    Bloque residual (He et al. 2015): dos ConvLayer 3x3 con ReLU entre medio y
    un skip connection aditivo. Permite apilar muchos bloques sin degradar el
    entrenamiento (el gradiente tiene un "atajo" a través del skip).
    """
    def __init__(self, channels=128, kernel=3):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel, 1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel, 1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class DeconvLayer(nn.Module):
    """
    Convolución traspuesta (upsampling aprendido) + InstanceNorm. Es la
    operación inversa de una Conv con stride: duplica la resolución espacial.
    """
    def __init__(self, in_c, out_c, kernel, stride, output_padding,
                 norm='instance'):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel, stride,
                                         kernel // 2, output_padding)
        self.norm_type = norm
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_c, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_c, affine=True)

    def forward(self, x):
        x = self.deconv(x)
        return x if self.norm_type == 'None' else self.norm(x)
```
::::


::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — Ensamblar la TransformerNet

**Objetivo:** Usar los bloques del setup (`ConvLayer`, `ResidualLayer`, `DeconvLayer`) para armar la red completa encoder → residuales → decoder.

**Enunciado:**

Completá la clase `TransformerNetwork(nn.Module)`. En `__init__` construí tres bloques como `nn.Sequential` —`self.ConvBlock`, `self.ResidualBlock` y `self.DeconvBlock`— y en `forward` encadenálos en ese orden.

Para orientarte, ésta es la forma `(canales, alto, ancho)` del tensor en cada punto de la red, suponiendo una entrada de referencia de `256×256`:

```
entrada           →  (  3, 256, 256)
ConvBlock         →  (128,  64,  64)     # reduce la resolución 4×, sube canales
ResidualBlock     →  (128,  64,  64)     # no cambia la forma
DeconvBlock       →  (  3, 256, 256)     # restaura la resolución, vuelve a 3 canales
```

Es decir: el encoder comprime espacio e infla canales, los residuales trabajan en esa representación comprimida, y el decoder deshace la compresión para devolver una imagen del mismo tamaño que la entrada.

---

**`self.ConvBlock` (encoder):** tres `ConvLayer` seguidas cada una por `nn.ReLU()`.

| Capa | Canales in → out | Kernel | Stride | Cambio de resolución |
|---|---|---|---|---|
| 1 | 3 → 32 | 9 | 1 | la preserva |
| 2 | 32 → 64 | 3 | 2 | la divide a la mitad |
| 3 | 64 → 128 | 3 | 2 | la divide otra vez a la mitad |

La primera usa un kernel grande (`9`) para tomar contexto espacial amplio al entrar a la red; las otras dos hacen reducción de resolución con kernel `3` y `stride=2`.

---

**`self.ResidualBlock`:** cinco `ResidualLayer(128, 3)` apilados, todos idénticos. No cambian la forma del tensor. El rol de los skip connections es que la identidad sea una solución trivial: así la red puede aprender una *corrección estilística* sobre las features que le llegan del encoder, sin destruir la estructura de contenido que contienen.

---

**`self.DeconvBlock` (decoder):** dos `DeconvLayer` seguidas cada una por `nn.ReLU()`, y al final una `ConvLayer` de cierre sin activación posterior.

| Capa | Tipo | Canales in → out | Configuración | Sigue con |
|---|---|---|---|---|
| 1 | `DeconvLayer` | 128 → 64 | `kernel=3, stride=2, output_padding=1` | `nn.ReLU()` |
| 2 | `DeconvLayer` | 64 → 32 | `kernel=3, stride=2, output_padding=1` | `nn.ReLU()` |
| 3 | `ConvLayer` | 32 → 3 | `kernel=9, stride=1, norm='None'` | nada |

Las dos `DeconvLayer` son la contraparte de las convs `stride=2` del encoder: cada una duplica la resolución. La `ConvLayer` final vuelve a 3 canales y va con `norm='None'` y **sin** ReLU posterior, porque queremos que la salida sea una imagen en rango libre (valores reales que luego se recortan al mostrar), no features activadas por una no-linealidad.
::::


::::cell{#ej3-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class TransformerNetwork(nn.Module):
    """
    Red encoder → 5 bloques residuales → decoder (Johnson et al. 2016).

    Arquitectura:
      Encoder:   3 convs (3→32→64→128 canales, reduce 4x la resolución).
      Residual:  5 bloques residuales de 128 canales (no cambian shape).
      Decoder:   2 deconvs (128→64→32 canales, duplica 2x dos veces) + 1 conv
                 final a 3 canales.
    La salida tiene la misma forma que la entrada.
    """
    def __init__(self):
        super().__init__()
        # ─── Encoder: sube canales, baja resolución ─────────────────────────
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1), nn.ReLU(),
            ConvLayer(32, 64, 3, 2), nn.ReLU(),
            ConvLayer(64, 128, 3, 2), nn.ReLU(),
        )
        # ─── Bloques residuales: mismo shape adentro y afuera ───────────────
        # Los skip connections permiten que la identidad sea una solución
        # trivial: la red aprende una "corrección estilística" sobre el
        # contenido que ya pasó el encoder.
        self.ResidualBlock = nn.Sequential(
            *[ResidualLayer(128, 3) for _ in range(5)]
        )
        # ─── Decoder: baja canales, sube resolución ─────────────────────────
        # La última conv NO tiene norm ni ReLU: queremos que devuelva una
        # imagen en rango libre (después se recorta a [0, 255] al mostrar).
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1), nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1), nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm='None'),
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        return self.DeconvBlock(x)
```
::::


::::cell{#ej3-cierre type=markdown role=intro}
**Cerrando el ejercicio — por qué esta arquitectura funciona:**

Vale la pena detenerse en dos decisiones arquitectónicas que explican que la TransformerNet resuelva bien el problema.

**1. Por qué encoder → residuales → decoder, y no simplemente convoluciones en cascada a resolución completa.**

El encoder comprime la imagen a una representación de `(128, H/4, W/4)`. En ese espacio, cada posición "ve" un parche amplio de la imagen original (gran campo receptivo efectivo) y los 128 canales codifican patrones más abstractos y semánticos que los píxeles crudos. Operar ahí tiene dos ventajas: es mucho más barato en cómputo y memoria que hacerlo a resolución completa, y es más expresivo para aprender patrones de estilo que involucran texturas extendidas (pinceladas, paletas, estructuras repetitivas). Los bloques residuales hacen el trabajo de "reestilización" sobre esa representación compacta, y el decoder la devuelve al tamaño original.

**2. Por qué el bloque del medio usa conexiones residuales y no convoluciones simples.**

Cada bloque residual calcula `x + F(x)` en vez de `F(x)`. Esto tiene dos consecuencias importantes para el entrenamiento:

- Inicializar los pesos cerca de cero equivale a la **función identidad**, que es una muy buena inicialización para esta tarea: al arrancar el entrenamiento, la red casi no toca las features que le llegan del encoder (que ya contienen toda la información de contenido) y el gradiente solo la empuja hacia el estilo objetivo. Sin residuales, la red tendría que aprender desde cero a reproducir contenido **y** estilo a la vez.
- El skip connection le da al gradiente un **atajo** que atraviesa el bloque sin pasar por las convs. Con 5 bloques apilados eso es crítico: sin atajos el gradiente se atenuaría capa a capa y el entrenamiento se volvería mucho más difícil.
::::


::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — Verificar la arquitectura

**Objetivo:** Instanciar la red, contar sus parámetros y verificar que un tensor de entrada sale con la misma forma que entró.

**Enunciado:**

1. Instanciá `TransformerNetwork()` y guardala en `transformerNet`.
2. Contá el total de parámetros entrenables (usá `sum(p.numel() for p in transformerNet.parameters())`) e imprimilo. Esperamos ~1.68 millones.
3. Generá un tensor de prueba `x = torch.randn(1, 3, 256, 256)`, pasalo por la red con `y = transformerNet(x)` e imprimí la forma de la salida. Tiene que ser **idéntica** a la de la entrada.

> **Pista:** Por ahora no movemos nada al GPU — este ejercicio es solo de verificación estructural.
::::


::::cell{#ej4-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Instanciar y contar parámetros ──────────────────────────────────────────
transformerNet = TransformerNetwork()
n_params = sum(p.numel() for p in transformerNet.parameters())
print(f"Parámetros totales: {n_params:,}  (~{n_params * 4 / 1e6:.1f} MB en float32)")

# ─── Verificar que la forma espacial se preserva ─────────────────────────────
x = torch.randn(1, 3, 256, 256)
y = transformerNet(x)
print(f"Entrada:  {tuple(x.shape)}")
print(f"Salida:   {tuple(y.shape)}")
assert x.shape == y.shape, "La salida debe tener la misma forma que la entrada"
```
::::


::::cell{#ej4-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La red reduce la resolución espacial 4× en el encoder, opera en esa resolución reducida durante los bloques residuales, y la recupera en el decoder. ¿Por qué es importante que la **salida final tenga exactamente la misma forma que la entrada**?
::::


::::cell{#ej4-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Queremos que la red sea una **función de estilización de imágenes**: en la entrada entra una imagen arbitraria y en la salida sale esa misma imagen estilizada. Si la salida tuviera otra resolución, no podríamos compararla pixel-a-pixel con la entrada en la pérdida de contenido (ni siquiera con perceptual loss tiene sentido comparar features de un tensor `256×256` con uno `64×64`). Además, a nivel de uso práctico, el alumno o el usuario espera poder mandar una imagen y recibir la misma imagen pintada.
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     SECCIÓN C: Entrenamiento (método de Johnson)
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#secC type=markdown role=section}
---
## Sección C: Entrenamiento
::::


::::cell{#intro-secC type=markdown role=intro}
### El loop

Ya tenés todas las piezas:

- `gram`, `content_loss`, `style_loss` — las pérdidas perceptuales (Sección A).
- `TransformerNetwork` — la red que vamos a entrenar (Sección B).

Falta una última pieza preescrita: la **VGG-16** que calcula las features sobre las que medimos las pérdidas. Usamos los pesos Caffe-convertidos de Justin Johnson, que son los que esperan ser alimentados con imágenes BGR en rango [0, 255] menos la media de ImageNet (`[-103.939, -116.779, -123.68]` por canal). Ese preprocesamiento es el que asumimos en toda la sección.

> **Nota — qué es el "preprocesamiento Caffe":** Caffe (Berkeley, ~2014) es un framework de deep learning anterior a PyTorch donde se entrenaron originalmente muchos modelos clásicos (incluida la VGG-16). Sus pesos asumen una convención distinta a la de PyTorch "moderno": canales en orden **BGR** (no RGB), valores en rango **[0, 255]** (no [0, 1]), y la media de ImageNet restada por canal. Cuando cargamos esos pesos en PyTorch tenemos que respetar esa receta o la red "ve" imágenes distintas de las que fue entrenada y las features salen basura. Por eso todo el laboratorio trabaja en BGR (leemos con `cv2`, que ya devuelve en BGR) y en escala [0, 255].
::::


::::cell{#setup-vgg type=code role=setup}
```python
# ─── Setup: descarga y definición de la VGG-16 perceptual ──────────────────
# Los pesos de Justin Johnson son la versión Caffe-convertida de VGG-16. Son
# consistentes con el preprocesamiento BGR + mean-subtraction que usamos en
# el resto del laboratorio (itot/ttoi están en BGR por la lectura con cv2).
VGG_PATH = '/content/vgg16-00b39a1b.pth'
if not os.path.exists(VGG_PATH):
    !wget -q --show-progress https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth -O {VGG_PATH}


class VGG16(nn.Module):
    """
    VGG-16 congelada que devuelve, en un dict, las activaciones de relu1_2,
    relu2_2, relu3_3 y relu4_3 — las capas que el paper usa para medir
    contenido y estilo.
    """
    def __init__(self, vgg_path=VGG_PATH):
        super().__init__()
        vgg = models.vgg16(weights=None)
        vgg.load_state_dict(torch.load(vgg_path, weights_only=False),
                            strict=False)
        self.features = vgg.features
        for p in self.features.parameters():
            p.requires_grad = False  # red frozen: nunca se actualiza

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2',
                  '15': 'relu3_3', '22': 'relu4_3'}
        out = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                out[layers[name]] = x
                if name == '22':
                    break
        return out


print('VGG16 lista')
```
::::


::::cell{#setup-hparams type=code role=setup}
```python
# ─── Setup: hiperparámetros del entrenamiento ──────────────────────────────
# Los valores por defecto vienen del repo oficial del paper y funcionan bien
# sobre COCO val2017. Si querés experimentar, los pesos content/style son
# los más influyentes: subir STYLE_WEIGHT da un estilo más agresivo.
TRAIN_IMAGE_SIZE = 256
DATASET_PATH     = '/content/dataset'
STYLE_IMAGE_PATH = style_path

NUM_EPOCHS     = 1
BATCH_SIZE     = 4
CONTENT_WEIGHT = 50
STYLE_WEIGHT   = 50
ADAM_LR        = 1e-3

PRINT_EVERY    = 25
SAVE_MODEL_PATH = '/content/models/'
SAVE_IMAGE_PATH = '/content/samples/'
SEED = 35

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)

# Media de ImageNet (BGR) que se resta a las imágenes antes de pasarlas a VGG.
imagenet_neg_mean = torch.tensor(
    [-103.939, -116.779, -123.68], dtype=torch.float32
).reshape(1, 3, 1, 1).to(device)

# Semilla para reproducibilidad.
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed(SEED)
elif device == "mps":
    torch.mps.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```
::::


::::cell{#ej5-enunciado type=markdown role=enunciado}
### Ejercicio 5 — Completar el loop de entrenamiento

**Objetivo:** Ensamblar todas las piezas en el loop de entrenamiento de Johnson y correrlo sobre COCO.

**Contexto:**

El loop hace, por cada batch de imágenes de contenido:

1. Pasar el batch por la **TransformerNet** → imagen generada.
2. Pasar contenido y generado por **VGG** → dos dicts de features.
3. Calcular `content_loss` entre los dos dicts.
4. Calcular `style_loss` entre el dict del generado y la Gram precomputada del estilo.
5. Combinar con los pesos `CONTENT_WEIGHT` y `STYLE_WEIGHT`, backward, step.

Las partes de infraestructura (DataLoader, precomputar la Gram del estilo, el chequeo de `cur_bs`, los prints periódicos, la conversión RGB→BGR con `[:, [2, 1, 0]]`) ya están resueltas. Tenés que completar el cuerpo del loop.

**Enunciado:**

Completá los `TODO` marcados en el código. Tres pasos:

1. **Forward de la red generadora** sobre el batch de contenido para producir la imagen estilizada de cada iteración.
2. **Features perceptuales**: extraé las features de VGG del batch de contenido y de la salida generada. Acordate de que esta VGG asume el preprocesamiento Caffe (BGR + media de ImageNet restada); la media ya la tenés precalculada entre los hiperparámetros.
3. **Pérdidas ponderadas**: reutilizá las dos funciones del Ejercicio 2 y combiná cada una con su peso correspondiente entre los hiperparámetros del entrenamiento.

Después corré la celda. Con `NUM_EPOCHS=1` y `BATCH_SIZE=4`, en T4 tarda ~5 minutos.

> **Pistas:**
> - La Gram del estilo se precalcula **antes** del loop (no depende del contenido, así que calcularla adentro sería trabajo repetido).
> - La pérdida de estilo espera una Gram precomputada con el mismo tamaño de batch que la imagen generada. Como la imagen de estilo se expandió al `BATCH_SIZE` nominal, en la última iteración puede sobrar una fila: por eso ves el corte `{k: v[:cur_bs] for k, v in style_gram.items()}` ya resuelto.
::::


::::cell{#ej5-code type=code role=student-code}
```python
def train():
    # ─── Dataset y dataloader ────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    total_iters = NUM_EPOCHS * len(train_loader)
    print(f'Dataset: {len(train_dataset)} imgs, '
          f'{len(train_loader)} iters/epoch, total={total_iters}')

    # ─── Modelos ────────────────────────────────────────────────────────────
    net = TransformerNetwork().to(device)
    VGG = VGG16().to(device)

    # ─── Gram de la imagen de estilo (precomputada) ─────────────────────────
    style_image = load_image(STYLE_IMAGE_PATH)
    style_tensor = itot(style_image).to(device).add(imagenet_neg_mean)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {k: gram(v) for k, v in style_features.items()}

    # ─── Optimizador ────────────────────────────────────────────────────────
    optimizer = optim.Adam(net.parameters(), lr=ADAM_LR)

    c_hist, s_hist, t_hist = [], [], []
    c_sum = s_sum = t_sum = 0.0
    batch_count = 1
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\n======== Epoch {epoch + 1}/{NUM_EPOCHS} ========')
        for content_batch, _ in train_loader:
            cur_bs = content_batch.shape[0]
            optimizer.zero_grad()

            # Las imágenes vienen en RGB; VGG Caffe espera BGR → intercambiamos canales.
            content_batch = content_batch[:, [2, 1, 0]].to(device)

            # ─── TODO (1): obtené la imagen estilizada de esta iteración ─────
            generated_batch = None

            # ─── TODO (2): features perceptuales de contenido y generado ─────
            # Necesitás las activaciones de VGG para ambas imágenes, con el
            # preprocesamiento que la red espera (ver hiperparámetros).
            content_features = None
            generated_features = None

            # ─── TODO (3): combiná las dos pérdidas con sus pesos ────────────
            # sg_cur: Gram del estilo ajustada al tamaño real de este batch
            # (ya resuelto, no lo toques).
            sg_cur = {k: v[:cur_bs] for k, v in style_gram.items()}
            c_loss = None
            s_loss = None

            # ─── Backward y step (ya resuelto) ──────────────────────────────
            total_loss = c_loss + s_loss
            total_loss.backward()
            optimizer.step()

            c_sum += c_loss.item(); s_sum += s_loss.item()
            t_sum += total_loss.item()

            if batch_count % PRINT_EVERY == 0 or batch_count == total_iters:
                elapsed = time.time() - start
                eta = elapsed / batch_count * (total_iters - batch_count)
                print(f'[{batch_count:5d}/{total_iters}] '
                      f'content={c_sum / batch_count:.1f}  '
                      f'style={s_sum / batch_count:.1f}  '
                      f'total={t_sum / batch_count:.1f}  '
                      f'elapsed={elapsed:.0f}s  eta={eta:.0f}s')
                c_hist.append(c_sum / batch_count)
                s_hist.append(s_sum / batch_count)
                t_hist.append(t_sum / batch_count)

            batch_count += 1

    print(f'\nEntrenamiento terminado en {(time.time() - start) / 60:.1f} min')

    net.eval().cpu()
    final_path = f'{SAVE_MODEL_PATH}final.pth'
    torch.save(net.state_dict(), final_path)
    print(f'Pesos finales guardados: {final_path}')

    return c_hist, s_hist, t_hist


content_hist, style_hist, total_hist = train()
```

```python solution
def train():
    # ─── Dataset y dataloader ────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    total_iters = NUM_EPOCHS * len(train_loader)
    print(f'Dataset: {len(train_dataset)} imgs, '
          f'{len(train_loader)} iters/epoch, total={total_iters}')

    # ─── Modelos ────────────────────────────────────────────────────────────
    net = TransformerNetwork().to(device)
    VGG = VGG16().to(device)

    # ─── Gram de la imagen de estilo (precomputada) ─────────────────────────
    style_image = load_image(STYLE_IMAGE_PATH)
    style_tensor = itot(style_image).to(device).add(imagenet_neg_mean)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {k: gram(v) for k, v in style_features.items()}

    # ─── Optimizador ────────────────────────────────────────────────────────
    optimizer = optim.Adam(net.parameters(), lr=ADAM_LR)

    c_hist, s_hist, t_hist = [], [], []
    c_sum = s_sum = t_sum = 0.0
    batch_count = 1
    start = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\n======== Epoch {epoch + 1}/{NUM_EPOCHS} ========')
        for content_batch, _ in train_loader:
            cur_bs = content_batch.shape[0]
            optimizer.zero_grad()

            content_batch = content_batch[:, [2, 1, 0]].to(device)  # RGB→BGR

            # ─── (1) Forward de la TransformerNet ──────────────────────────
            # La red recibe una imagen de contenido y devuelve su versión estilizada.
            generated_batch = net(content_batch)

            # ─── (2) Features con VGG ──────────────────────────────────────
            # VGG Caffe espera imágenes BGR centradas en la media de ImageNet.
            # Le restamos la media (imagenet_neg_mean ya es negativa; add la suma).
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            # ─── (3) Pérdidas ──────────────────────────────────────────────
            sg_cur = {k: v[:cur_bs] for k, v in style_gram.items()}
            # Reutilizamos las funciones del Ejercicio 2.
            c_loss = CONTENT_WEIGHT * content_loss(generated_features, content_features)
            s_loss = STYLE_WEIGHT   * style_loss(generated_features, sg_cur)

            # ─── Backward y step ───────────────────────────────────────────
            total_loss = c_loss + s_loss
            total_loss.backward()
            optimizer.step()

            c_sum += c_loss.item(); s_sum += s_loss.item()
            t_sum += total_loss.item()

            if batch_count % PRINT_EVERY == 0 or batch_count == total_iters:
                elapsed = time.time() - start
                eta = elapsed / batch_count * (total_iters - batch_count)
                print(f'[{batch_count:5d}/{total_iters}] '
                      f'content={c_sum / batch_count:.1f}  '
                      f'style={s_sum / batch_count:.1f}  '
                      f'total={t_sum / batch_count:.1f}  '
                      f'elapsed={elapsed:.0f}s  eta={eta:.0f}s')
                c_hist.append(c_sum / batch_count)
                s_hist.append(s_sum / batch_count)
                t_hist.append(t_sum / batch_count)

            batch_count += 1

    print(f'\nEntrenamiento terminado en {(time.time() - start) / 60:.1f} min')

    net.eval().cpu()
    final_path = f'{SAVE_MODEL_PATH}final.pth'
    torch.save(net.state_dict(), final_path)
    print(f'Pesos finales guardados: {final_path}')

    return c_hist, s_hist, t_hist


content_hist, style_hist, total_hist = train()
```
::::


::::cell{#ej5-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Durante todo el entrenamiento, la **VGG está congelada** (sus pesos no cambian) y la **TransformerNet se entrena**. ¿Por qué usamos la VGG solo como **extractor de features** y no la entrenamos también? ¿Qué pasaría si la entrenáramos junto con la TransformerNet, usando la misma pérdida total?
::::


::::cell{#ej5-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La VGG funciona en este montaje como una **métrica perceptual** — un "ojo" que sabe reconocer patrones semánticos gracias a haber visto ImageNet. Las pérdidas `content_loss` y `style_loss` tienen sentido precisamente porque sus features son estables, interpretables y comparables entre imágenes distintas. Si entrenáramos la VGG junto con la TransformerNet, sus features perderían ese anclaje: la red podría aprender a **deformar el espacio de features** para que la pérdida baje sin que las imágenes realmente se parezcan. Un caso extremo trivial: si VGG "colapsa" y devuelve el mismo vector para cualquier entrada, `content_loss` y `style_loss` dan 0 y el gradiente deja de empujar a la TransformerNet hacia un comportamiento útil. En otras palabras, la pérdida perdería significado y todo el sistema se degeneraría a una solución trivial.

Formalmente: estamos usando la VGG como **función objetivo**, no como modelo a optimizar. El papel de modelo a optimizar lo juega solo la TransformerNet. Mantener la VGG fija es lo que transforma "minimizar una distancia perceptual bien definida" en un problema de optimización bien planteado.
```
::::


::::cell{#ej6-enunciado type=markdown role=enunciado}
### Ejercicio 6 — Curvas de pérdida

**Objetivo:** Visualizar cómo evolucionan las tres pérdidas (contenido, estilo, total) a lo largo del entrenamiento.

**Enunciado:**

1. Armá una figura `plt.subplots(1, 3, figsize=(15, 4))`.
2. Ploteá `content_hist` en el primer eje, `style_hist` en el segundo, `total_hist` en el tercero.
3. A cada eje ponele título (`"Content loss"`, `"Style loss"`, `"Total loss"`), etiqueta `xlabel = f"checkpoint (cada {PRINT_EVERY} iters)"` y grilla.
4. `plt.tight_layout()` y `plt.show()`.
::::


::::cell{#ej6-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Curvas de pérdida ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(content_hist); axes[0].set_title('Content loss')
axes[1].plot(style_hist);   axes[1].set_title('Style loss')
axes[2].plot(total_hist);   axes[2].set_title('Total loss')
for ax in axes:
    ax.set_xlabel(f'checkpoint (cada {PRINT_EVERY} iters)')
    ax.grid(True)
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej6-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirá las tres curvas con cuidado. Vas a ver que la **style loss baja** fuertemente, la **total loss baja** también, pero la **content loss sube** durante la primera mitad del entrenamiento antes de estabilizarse. ¿Qué está pasando? ¿Qué nos dice esto sobre el tradeoff intrínseco del problema que estamos resolviendo?
::::


::::cell{#ej6-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La TransformerNet arranca con pesos aleatorios, pero por la estructura encoder-residuales-decoder (y los skip connections en el bloque residual) su salida inicial se parece bastante a la entrada — los bloques residuales aportan poco al principio. Eso es óptimo para la content loss: la content loss arranca **muy baja** (la imagen generada ≈ la imagen de contenido). El problema es que la style loss también arranca **enorme**: la imagen no tiene nada del estilo.

El optimizador, que mira la suma `CONTENT_WEIGHT * content + STYLE_WEIGHT * style`, encuentra que el gradiente de style es mucho más grande y empuja a la red a **deformar la imagen para acercar su Gram a la del estilo**. Eso inevitablemente aleja las features de contenido de las originales: la content loss **sube**. El proceso se estabiliza cuando las dos pérdidas están balanceadas por los pesos: bajar más la style loss costaría más content del que la suma ponderada tolera.

Esto es el **tradeoff fundamental** de la transferencia de estilos: contenido y estilo son objetivos en tensión, no hay un óptimo que minimice ambos simultáneamente. Los pesos `CONTENT_WEIGHT` y `STYLE_WEIGHT` son el knob con el que elegimos en qué punto de esa curva de Pareto queremos quedarnos — más estilo agresivo o más fidelidad al contenido.
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     SECCIÓN D: Inferencia y comparación
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#secD type=markdown role=section}
---
## Sección D: Inferencia y análisis
::::


::::cell{#intro-secD type=markdown role=intro}
### El pago: un solo forward pass

Ya entrenaste el modelo. La promesa del método de Johnson es que estilizar una imagen nueva cuesta **un solo forward pass** — sin optimización, sin gradientes, sin iteraciones. Lo vamos a comprobar ahora sobre 3 imágenes random del dataset.
::::


::::cell{#ej7-enunciado type=markdown role=enunciado}
### Ejercicio 7 — Estilizar imágenes nuevas

**Objetivo:** Cargar los pesos entrenados, escribir una función de inferencia, y aplicarla a 3 imágenes random de val2017 para visualizar los resultados.

**Enunciado:**

1. Cargá los pesos finales (`{SAVE_MODEL_PATH}final.pth`) en una nueva instancia de `TransformerNetwork` y mandala al `device`. Ponela en `.eval()`.
2. Definí la función `stylize(image_path, max_size=512)`:
   - Cargá la imagen con `load_image` (devuelve BGR).
   - Convertila a tensor con `itot(img, max_size=max_size)` y mandala al `device`.
   - Pasala por la red **dentro de `torch.no_grad()`** y recortá la salida a `[0, 255]` con `.clamp(0, 255)`.
   - Devolvé la tupla `(original_bgr, estilizada_numpy)` donde la segunda es un array numpy `H×W×3`.
3. Elegí 3 archivos random de `/content/dataset/val2017` y mostralos en una grilla `3 × 2` (original a la izquierda, estilizada a la derecha). Para mostrar tenés que convertir BGR → RGB con `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` (la estilizada conviene castearla a `uint8` primero: `styled.clip(0, 255).astype('uint8')`).

> **Pista:** `torch.load(..., map_location=device, weights_only=True)` te carga los pesos en el dispositivo correcto en una línea.
::::


::::cell{#ej7-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Cargar los pesos entrenados ─────────────────────────────────────────────
model = TransformerNetwork().to(device)
model.load_state_dict(torch.load(
    f'{SAVE_MODEL_PATH}final.pth', map_location=device, weights_only=True
))
model.eval()


def stylize(image_path, max_size=512):
    """Aplica la TransformerNet entrenada a una imagen. Devuelve (orig_bgr, styled_np)."""
    img = load_image(image_path)  # BGR desde cv2
    tensor = itot(img, max_size=max_size).to(device)
    with torch.no_grad():
        out = model(tensor).cpu().clamp(0, 255)
    return img, ttoi(out)


# ─── Aplicar a 3 imágenes random ─────────────────────────────────────────────
val_dir = '/content/dataset/val2017'
samples = random.sample(os.listdir(val_dir), 3)

fig, axes = plt.subplots(3, 2, figsize=(12, 15))
for i, fname in enumerate(samples):
    orig, styled = stylize(f'{val_dir}/{fname}')
    # BGR → RGB para matplotlib.
    axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f'Original: {fname}')
    axes[i, 0].axis('off')
    styled_u8 = styled.clip(0, 255).astype('uint8')
    axes[i, 1].imshow(cv2.cvtColor(styled_u8, cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title('Estilizada')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej7-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirando las 3 imágenes estilizadas, respondé:

- ¿Qué se **preserva** del contenido original?
- ¿Qué se **transfiere** del estilo?
::::


::::cell{#ej7-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- **Se preserva**: la estructura global de las escenas — las siluetas de personas y objetos, sus posiciones relativas, las zonas de luz y sombra, las divisiones entre objeto y fondo. Todo lo que la content loss sobre `relu2_2` mide: qué hay y dónde.
- **Se transfiere**: la paleta de colores del estilo (en mosaic, tonos cálidos y azulados contrastantes), la textura tipo teselas pequeñas, los bordes duros entre regiones, la estética pictórica que elimina gradientes suaves. Todo lo que las matrices de Gram capturan como correlaciones entre canales.

En general, cuanto más alejada está una imagen del dominio de COCO (por ejemplo, una foto de satélite o un documento) o más fuerte sea el desbalance entre sus estadísticas y las de COCO, peor se va a comportar la red. La red aprendió un estilo, no un traductor universal de texturas.
```
::::


::::cell{#reflexion-cierre type=markdown role=intro}
### Reflexión de cierre: Gatys vs. Johnson

Implementaste el método de Johnson completo, pero la teoría giró en torno al método de Gatys. Vale la pena cerrar el lab comparando los dos enfoques sobre las mismas cuatro dimensiones — no son alternativas equivalentes, cada uno domina en un régimen distinto.

**Tiempo por imagen nueva.** Gatys estiliza una imagen con cientos de iteraciones de descenso de gradiente: del orden de segundos a minutos en GPU, según resolución y número de pasos. Johnson hace la misma tarea en un solo forward pass — milisegundos. El costo no desaparece, se mueve al entrenamiento: la TransformerNet te llevó unos 5 minutos sobre COCO val2017 con T4, pero es un costo único que se paga una sola vez por estilo.

**Flexibilidad frente a estilos nuevos.** Probar un estilo nuevo con Gatys es gratis: cambiás la imagen de estilo, volvés a optimizar y listo. Con Johnson, cada estilo nuevo exige entrenar una red completa desde cero. Existen variantes multi-estilo que aflojan esta restricción — Dumoulin et al. 2017 (instance normalization condicional) y Huang & Belongie 2017 (AdaIN) son las más conocidas — pero el Johnson vanilla es estrictamente mono-estilo.

**Calidad perceptual.** Gatys puede ganar en casos específicos: al optimizar directamente sobre cada imagen logra un match perceptual más fino, sobre todo con estilos muy marcados o contenidos inusuales. Retocar hiperparámetros y re-optimizar es además cuestión de segundos. Johnson tiende a ganar en consistencia —por ejemplo, entre frames consecutivos de un video— y en robustez: aprendió una función suave sobre todo COCO, así que no cae en los mínimos locales feos que a veces produce una optimización por píxeles.

**Cuándo elegir cada uno.** Dos escenarios típicos cierran la comparación. Si sos un artista experimentando con muchos estilos sobre una misma foto, Gatys es el método natural: el costo dominante es cambiar de estilo, no estilizar, y entrenar una red por cada estilo que vas a usar una sola vez sería un desperdicio. Si en cambio estás construyendo una app móvil que estiliza video en tiempo real, Johnson es la respuesta clara: cada frame tiene que salir en milisegundos, y la cantidad de estilos suele ser chica y fija (cinco filtros, por ejemplo). El costo de entrenar esas cinco redes una vez se amortiza sin esfuerzo. Con AdaIN incluso se pueden meter decenas de estilos intercambiables en una sola red.

La regla general que te llevás: si el costo dominante de tu aplicación es el **tiempo por imagen**, Johnson; si es la **variedad de estilos**, Gatys. El resto son matices.
::::


<!-- ══════════════════════════════════════════════════════════════════════
     CHECKLIST + CIERRE
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#checklist type=markdown role=checklist}
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Reinicié el entorno con GPU activada y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] El entrenamiento corrió sobre `val2017` completo (1.250 iteraciones) y la curva de `total_loss` baja de forma estable.
- [ ] Las 3 imágenes estilizadas del Ejercicio 7 muestran claramente el estilo transferido.
- [ ] Los valores numéricos que imprimo son razonables (no hay infinitos ni `NaN`).
- [ ] Todos los gráficos tienen título, etiquetas en los ejes y grilla.
- [ ] No modifiqué ninguna celda fuera de las de actividad.
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Entrenaste una red de transferencia de estilos desde cero. Implementaste las piezas que comparten los métodos de Gatys y Johnson (matriz de Gram y pérdidas perceptuales sobre features de VGG), ensamblaste una TransformerNet encoder-decoder con bloques residuales, y la entrenaste sobre COCO para reproducir un estilo específico en un solo forward pass.

En el próximo bloque vamos a ver modelos generativos más generales (autoencoders variacionales y GANs), donde la idea de "una red que genera imágenes" se profundiza hasta convertirse en el objetivo central, no solo un medio para transferir estilo.
::::
