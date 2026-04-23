---
lab: "3c"
title: "Laboratorio n° 3c: Transferencia de estilos"
subject: "Redes Neuronales Profundas"
block: "3 — Transferencia de conocimiento"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 3c — Transferencia de estilos (BACKUP)

     Este archivo preserva la sección de transferencia de estilos que
     originalmente formaba parte del Laboratorio 3 (ver Laboratorio_3.lab.md).
     Se extrajo a un lab aparte porque todavía no está decidido si queda como
     ejercicio definitivo o se reemplaza por otro. Mantiene la estructura de
     `.lab.md` para que pueda compilarse con `tools/lab_build.py` si en algún
     momento se promociona a lab final.

     Compilar con: python tools/lab_build.py _TPS/sources/Laboratorio_3c.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_3c.ipynb
             _TPS/Soluciones/Laboratorio_3c_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 3c: Transferencia de estilos

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 3 — Transferencia de conocimiento

---

## Introducción

En los laboratorios anteriores usamos los feature maps de una red preentrenada **como punto de partida** para clasificación e inspección. Acá los vamos a usar de otra manera: congelados, como **funciones de pérdida** aplicadas sobre los píxeles de una imagen que sintetizamos desde cero.

La idea, del paper de *Gatys, Ecker y Bethge (2015)*, es descomponer una imagen en dos cosas independientes:

- Su **contenido**: la estructura espacial de objetos y regiones. Se captura midiendo la distancia entre los feature maps de una capa relativamente profunda, comparando la imagen sintetizada con la imagen de contenido.
- Su **estilo**: las correlaciones entre los canales de los feature maps, promediadas sobre el espacio. Se captura con la **matriz de Gram**, que mide "qué filtros se encienden juntos". La imagen sintetizada debe tener Grams parecidas a las de una imagen de estilo.

Sumando ambas pérdidas (más un pequeño término de suavidad) y haciendo descenso de gradiente **sobre los píxeles** de la imagen sintetizada, obtenemos una imagen que conserva la geometría de una y la paleta y texturas de la otra.

En este laboratorio vas a implementar las cuatro piezas de la pérdida (`content_loss`, `gram_matrix`, `style_loss`, `tv_loss`) y luego usarlas con la función `style_transfer` que te damos armada.

> **Importante — GPU:** este laboratorio hace descenso de gradiente sobre imágenes. **Activá la GPU en Colab** antes de empezar: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU*.

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para resolver cada ejercicio, consultá el material teórico de la Clase 3.

## IMPORTANTE: qué celdas podés modificar

- Solo modificá celdas que contengan el placeholder `# Tu código aquí` o `*(Escribí tu respuesta acá)*`.
- No modifiques las celdas de setup ni los tests: rompen la trazabilidad de la corrección.
::::


::::cell{#imports type=code role=setup}
```python
# ─── Setup: imports y detección de GPU ──────────────────────────────────────
# Estos son los imports que usa el laboratorio de punta a punta:
#   - torch / torchvision: red extractora preentrenada y transformaciones.
#   - PIL / matplotlib: cargar y mostrar imágenes.
#   - os / requests: descargar las imágenes de contenido y estilo.
# También detectamos si hay GPU disponible y la guardamos en `device`.
import os
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Dispositivo:        {device}")
```
::::


::::cell{#setup-preprocess type=code role=setup}
```python
# ─── Setup: helpers de preprocesamiento y desnormalización ──────────────────
# Toda red preentrenada de torchvision espera imágenes RGB estandarizadas con
# las medias y desvíos de ImageNet. Definimos eso una sola vez acá, junto con
# dos funciones que usás en todos los ejercicios:
#
#   - preprocess(pil_img, size): PIL → tensor normalizado listo para la red.
#   - deprocess(tensor):          tensor normalizado → array (H, W, 3) listo
#                                 para plt.imshow.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def preprocess(pil_img, size=224):
    """
    Convierte una imagen PIL en un tensor listo para entrar a una red preentrenada
    de torchvision. Redimensiona, normaliza con medias/desvíos de ImageNet y
    agrega la dimensión de batch.

    Parámetros:
    pil_img (PIL.Image): imagen a procesar.
    size (int o tuple): tamaño de salida. Si es int se usa (size, size).

    Retorna:
    tensor (Tensor): tensor de forma (1, 3, H, W).
    """
    if isinstance(size, int):
        size = (size, size)
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(),
                             std=IMAGENET_STD.tolist()),
    ])
    return tf(pil_img).unsqueeze(0)


def deprocess(tensor):
    """
    Invierte la normalización de ImageNet y devuelve un array numpy (H, W, 3)
    con valores recortados a [0, 1], listo para plt.imshow.

    Parámetros:
    tensor (Tensor): tensor normalizado, shape (1, 3, H, W) o (3, H, W).

    Retorna:
    arr (np.ndarray): array de forma (H, W, 3) en [0, 1].
    """
    img = tensor.detach().cpu()
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img * IMAGENET_STD.view(3, 1, 1) + IMAGENET_MEAN.view(3, 1, 1)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     Preparación de imágenes + red extractora
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#intro-tus-imagenes type=markdown role=intro}
### Preparación: tus imágenes de contenido y estilo

En este laboratorio vas a aplicar transferencia de estilos a **dos imágenes que vos elegís**:

- Una **imagen de contenido**: una foto cuya estructura (objetos, escena, composición) querés preservar.
- Una **imagen de estilo**: una pintura, ilustración o foto cuyas texturas y paleta de color querés transferir a la primera.

**Cómo dárselas al notebook.** Subí cada imagen a un host público que sirva el archivo directo (no una página HTML con la imagen embebida) y pegá las dos URLs en la celda siguiente. Opciones que funcionan:

- **Imgur:** subí la imagen y usá la URL que termina en `.jpg` o `.png` (clic derecho sobre la imagen → *Copiar dirección de imagen*).
- **GitHub (gist o repo público):** usá la URL que empieza con `raw.githubusercontent.com/...`.
- **Google Drive:** compartí con "cualquiera con el link", copiá el ID del archivo y armá `https://drive.google.com/uc?export=download&id=ID`.

**Sugerencias prácticas:**

- Elegí imágenes de resolución moderada (entre 300 y 1000 px por lado). Imágenes muy grandes hacen la optimización lenta.
- Para que el estilo se "note", conviene que la imagen de estilo tenga textura marcada (pinceladas, tramas, patrones) y colores saturados.
- Verificá que la URL devuelva la imagen directamente pegándola en una pestaña nueva del navegador: tenés que ver solo la imagen, sin marco de página web alrededor.
::::


::::cell{#tus-imagenes type=code role=student-code}
```python
# Completá las dos URLs con los links públicos a tus imágenes.
# URL_CONTENIDO: imagen cuyo contenido (estructura) querés preservar.
# URL_ESTILO:    imagen cuyo estilo (texturas, paleta) querés transferir.
URL_CONTENIDO = ""
URL_ESTILO    = ""
```

```python solution
# Ejemplo con el par clásico del paper de Gatys: foto de Tübingen (contenido)
# y La noche estrellada de Van Gogh (estilo). Reemplazá por tus imágenes.
URL_CONTENIDO = "https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/tubingen.jpg"
URL_ESTILO    = "https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night.jpg"
```
::::


::::cell{#setup-estilo type=code role=setup}
```python
# ─── Setup: descarga de tus imágenes y red extractora ───────────────────────
# Esta celda prepara todo lo que necesita el laboratorio:
#   - Descarga las dos imágenes que definiste en URL_CONTENIDO y URL_ESTILO.
#   - Carga `cnn_estilo`: una SqueezeNet1_1 preentrenada, congelada y en
#     modo eval, que vas a usar como extractor de features en todos los
#     ejercicios.
#   - Define `extract_features` y `features_from_img`, que recorren la red
#     capa por capa y devuelven la lista de feature maps.
if not URL_CONTENIDO or not URL_ESTILO:
    raise ValueError(
        "Completá URL_CONTENIDO y URL_ESTILO en la celda anterior antes de ejecutar esta celda."
    )

urls_estilo = {
    "content.jpg": URL_CONTENIDO,
    "style.jpg":   URL_ESTILO,
}
for nombre, url in urls_estilo.items():
    # Siempre re-descargar: si cambiás la URL, queremos traer la imagen nueva
    # en vez de servir un cache viejo de una corrida anterior.
    r = requests.get(url)
    r.raise_for_status()
    with open(nombre, "wb") as f:
        f.write(r.content)

content_img = Image.open("content.jpg").convert("RGB")
style_img   = Image.open("style.jpg").convert("RGB")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(content_img); axes[0].set_title("Contenido"); axes[0].axis('off')
axes[1].imshow(style_img);   axes[1].set_title("Estilo");    axes[1].axis('off')
plt.show()

# SqueezeNet1_1: pequeña (4.7 MB) y rápida, ideal para estilos.
# Usamos solo la parte .features (las capas convolucionales).
cnn_estilo = models.squeezenet1_1(weights='DEFAULT').features.to(device).eval()
for p in cnn_estilo.parameters():
    p.requires_grad_(False)  # red congelada: solo optimizamos la imagen


def extract_features(x, cnn):
    """
    Pasa `x` por la CNN capa por capa y retorna la lista de feature maps
    producidos por cada submódulo.

    Parámetros:
    x (Tensor): tensor de entrada, forma (1, 3, H, W).
    cnn (nn.Sequential): CNN a evaluar (aquí squeezenet.features).

    Retorna:
    features (list[Tensor]): lista de feature maps, uno por capa.
    """
    features = []
    for modulo in cnn.children():
        x = modulo(x)
        features.append(x)
    return features


def features_from_img(pil_img, size, cnn):
    """Preprocesa una imagen y extrae sus feature maps de la CNN."""
    x = preprocess(pil_img, size=size).to(device)
    return extract_features(x, cnn), x
```
::::


::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Pérdida de contenido

**Objetivo:** Implementar la pérdida de contenido, que mide cuánto difiere el mapa de activación de la imagen sintetizada del mapa de activación de una imagen de referencia en una capa elegida.

**Contexto matemático:**

Sea $F^\ell$ el mapa de activación de la imagen sintetizada en la capa $\ell$, y $P^\ell$ el mapa correspondiente de la imagen de contenido. La pérdida de contenido es la **suma de los cuadrados de las diferencias** elemento a elemento, ponderada por un escalar $w_c$:

$$L_c = w_c \sum_{i,j,k} (F^\ell_{ijk} - P^\ell_{ijk})^2$$

**Enunciado:**

Implementá `content_loss(content_weight, content_current, content_original)`:

- `content_weight` (float): el escalar $w_c$.
- `content_current` (Tensor): $F^\ell$, shape `(1, C, H, W)`.
- `content_original` (Tensor): $P^\ell$, shape `(1, C, H, W)`.

Devolvé un escalar con la pérdida, **sin bucles**.

> **Pista:** `((a - b) ** 2).sum()` te da la suma de diferencias cuadráticas elemento a elemento.
::::


::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
def content_loss(content_weight, content_current, content_original):
    """
    Calcula la pérdida de contenido para la transferencia de estilo.

    Parámetros:
    content_weight (float): peso escalar w_c de la pérdida.
    content_current (Tensor): feature map de la imagen sintetizada, shape (1, C, H, W).
    content_original (Tensor): feature map de la imagen de contenido, shape (1, C, H, W).

    Retorna:
    loss (Tensor): escalar con la pérdida.
    """
    return content_weight * ((content_current - content_original) ** 2).sum()


# ─── Test: la pérdida entre un tensor y sí mismo debe ser cero ───────────────
tensor_a = torch.randn(1, 8, 16, 16)
assert content_loss(1.0, tensor_a, tensor_a).item() == 0.0
tensor_b = tensor_a + 0.1
assert content_loss(1.0, tensor_a, tensor_b).item() > 0.0
print("content_loss OK")
```
::::


::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La imagen sintetizada se parecerá más o menos a la imagen de contenido según qué **capa** usemos para calcular `content_loss`. Si usamos una capa muy temprana (ej. la primera conv), la imagen resultante va a parecerse mucho a la de contenido píxel a píxel. Si usamos una capa muy profunda, la imagen puede parecerse **conceptualmente** al contenido pero diferir mucho en los detalles. ¿Por qué?
::::


::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Porque cada capa codifica un nivel distinto de abstracción:

- **Capa temprana:** los feature maps son, en esencia, versiones filtradas de la imagen que preservan bordes, colores y posiciones exactas. Hacer coincidir esos feature maps obliga casi a reconstruir la imagen original píxel a píxel.
- **Capa profunda:** los feature maps codifican "qué cosa hay y más o menos dónde" (una estructura de partes), pero no la apariencia pixelada exacta. Hacer coincidir esos feature maps permite muchas imágenes distintas a nivel de píxel que representan el mismo contenido abstracto — justo el margen de libertad que queremos para que el estilo pueda insertarse.

Por eso en la práctica se elige una capa **intermedia-profunda** para la pérdida de contenido: lo bastante alta para permitir variación estilística, pero no tanto como para perder el contenido reconocible.
```
::::


::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Matriz de Gram

**Objetivo:** Implementar la matriz de Gram, que captura las **correlaciones entre canales** de un feature map. Es la piedra angular de la pérdida de estilo.

**Contexto matemático:**

Sea $F^\ell \in \mathbb{R}^{C_\ell \times H_\ell \times W_\ell}$ un feature map. Si lo aplanamos espacialmente a $\tilde F^\ell \in \mathbb{R}^{C_\ell \times M_\ell}$ con $M_\ell = H_\ell W_\ell$, la matriz de Gram es

$$G^\ell = \tilde F^\ell (\tilde F^\ell)^\top \in \mathbb{R}^{C_\ell \times C_\ell}, \qquad G^\ell_{ij} = \sum_k \tilde F^\ell_{ik}\, \tilde F^\ell_{jk}$$

Cada elemento $G_{ij}$ mide cuánto co-activan los canales $i$ y $j$ sobre la imagen, promediado espacialmente. Las magnitudes dependen del tamaño del feature map, así que opcionalmente se **normaliza** dividiendo por $C \cdot H \cdot W$.

**Enunciado:**

Implementá `gram_matrix(features, normalize=True)`:

- `features` (Tensor): tensor de forma `(N, C, H, W)`.
- `normalize` (bool): si es `True`, divide el resultado por `C * H * W`.

Devolvé un tensor de forma `(N, C, C)`. **Sin bucles.**

> **Pistas:**
> - `features.reshape(N, C, H * W)` te deja el tensor listo para hacer la multiplicación.
> - `torch.bmm(A, B)` multiplica matrices por lotes (*batch matrix multiply*).
> - Para transponer las últimas dos dimensiones: `.transpose(1, 2)`.
::::


::::cell{#ej2-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
def gram_matrix(features, normalize=True):
    """
    Calcula la matriz de Gram de un feature map.

    Parámetros:
    features (Tensor): feature map, shape (N, C, H, W).
    normalize (bool): si True, divide el resultado por C*H*W.

    Retorna:
    gram (Tensor): matriz de Gram, shape (N, C, C).
    """
    N, C, H, W = features.shape
    F_plano = features.reshape(N, C, H * W)                # (N, C, H*W)
    gram = torch.bmm(F_plano, F_plano.transpose(1, 2))     # (N, C, C)
    if normalize:
        gram = gram / (C * H * W)
    return gram


# ─── Test: forma esperada y propiedad de simetría ────────────────────────────
feats_prueba = torch.randn(2, 4, 8, 8)
G = gram_matrix(feats_prueba)
assert G.shape == (2, 4, 4), f"shape esperada (2,4,4), obtenida {tuple(G.shape)}"
assert torch.allclose(G, G.transpose(1, 2), atol=1e-6), "la matriz de Gram no es simétrica"
print("gram_matrix OK")
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La matriz de Gram es una matriz **`C × C`**, donde `C` es el número de canales del feature map. ¿Por qué **no depende** de la resolución espacial `H × W`? ¿Qué nos dice eso sobre qué información se conserva y qué se pierde al computar el Gram?
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Al multiplicar $\tilde F^\ell (\tilde F^\ell)^\top$, la sumatoria $\sum_k \tilde F_{ik}\, \tilde F_{jk}$ recorre **todas** las posiciones espaciales $k$ y las colapsa en un único número por par de canales. La dimensión espacial se consume en la suma.

Esto quiere decir que la matriz de Gram **descarta dónde se activó cada cosa** y retiene solo **con qué otros canales se co-activa**. Dos imágenes con las mismas texturas pero en posiciones distintas tienen Grams casi iguales. Por eso la matriz de Gram es una buena representación del *estilo*: dos pinturas de Van Gogh tendrán patrones de co-activación similares (pinceladas gruesas, paleta azul-amarilla) aunque representen escenas completamente diferentes.

La contraparte es que la matriz de Gram **no** captura la estructura geométrica de la escena — justo por eso la combinamos con la pérdida de contenido, que sí la captura.
```
::::


::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — Pérdida de estilo

**Objetivo:** Combinar `gram_matrix` para medir la diferencia de estilo en varias capas simultáneamente.

**Contexto matemático:**

Dado un conjunto de capas $\mathcal{L}$ que elegimos como "capas de estilo", calculamos la matriz de Gram $G^\ell$ de la imagen sintetizada y $A^\ell$ de la imagen de estilo en cada capa $\ell \in \mathcal{L}$, y sumamos las distancias cuadráticas ponderadas:

$$L_s = \sum_{\ell \in \mathcal{L}} w_\ell \sum_{i,j} \left(G^\ell_{ij} - A^\ell_{ij}\right)^2$$

**Enunciado:**

Implementá `style_loss(feats, style_layers, style_targets, style_weights)`:

- `feats` (list[Tensor]): feature maps de la imagen sintetizada en todas las capas (lo que devuelve `extract_features`).
- `style_layers` (list[int]): índices de las capas a usar para la pérdida de estilo.
- `style_targets` (list[Tensor]): ya son **matrices de Gram** precalculadas sobre la imagen de estilo, una por capa en `style_layers`.
- `style_weights` (list[float]): pesos $w_\ell$ por capa.

Está **permitido usar un bucle `for`** sobre las capas de estilo.

> **Pista:** Para cada índice `i`, calculá `gram_matrix(feats[style_layers[i]])` y compará con `style_targets[i]` usando distancia cuadrática ponderada por `style_weights[i]`.
::::


::::cell{#ej3-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Calcula la pérdida de estilo sumando contribuciones de varias capas.

    Parámetros:
    feats (list[Tensor]): feature maps de la imagen sintetizada, uno por capa.
    style_layers (list[int]): índices de las capas a considerar.
    style_targets (list[Tensor]): matrices de Gram precalculadas de la imagen
        de estilo para las capas indicadas.
    style_weights (list[float]): peso por capa.

    Retorna:
    loss (Tensor): escalar con la pérdida de estilo.
    """
    loss = torch.tensor(0.0, device=feats[0].device)
    for i, idx in enumerate(style_layers):
        G = gram_matrix(feats[idx])
        loss = loss + style_weights[i] * ((G - style_targets[i]) ** 2).sum()
    return loss
```
::::


::::cell{#ej3-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

`style_layers` suele incluir capas de **varias profundidades** (unas tempranas y otras más profundas), mientras que `content_layer` es típicamente una sola. ¿Por qué el estilo se mide en varias capas y el contenido en una? Pensalo en términos de qué aspectos del "estilo" captura cada capa.
::::


::::cell{#ej3-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

El estilo de una imagen no es un concepto único: es la combinación de muchas escalas de textura. La paleta de colores y las pinceladas finas viven en capas tempranas (patrones de bajo nivel), mientras que las composiciones repetidas y las formas características (por ejemplo, los remolinos de *La noche estrellada*) solo aparecen en capas más profundas. Si midiéramos el estilo en una sola capa, capturaríamos solo una escala y la imagen sintetizada reflejaría solo ese aspecto.

El contenido, en cambio, es en gran medida una noción única: *qué cosa hay y más o menos cómo está ordenada*. Una capa intermedia-profunda la codifica bien, y agregar más capas solo haría que la imagen sintetizada se pareciera más píxel a píxel a la imagen de contenido — que es justo lo que no queremos.
```
::::


::::cell{#setup-style-transfer type=code role=setup}
```python
# ─── Setup: función de transferencia de estilos ─────────────────────────────
# `style_transfer` es la función que usás en el Ej 4 para ejecutar la
# transferencia completa: combina los cuatro términos de pérdida que
# implementás (content_loss, style_loss, gram_matrix, tv_loss) y corre el
# descenso de gradiente sobre los píxeles de la imagen sintetizada durante
# num_iters pasos. Asume que esas cuatro funciones ya están definidas antes
# de llamarla, así que ejecutá las celdas de los Ej 1-4A antes de esta.

def style_transfer(content_image, style_image, image_size, style_size,
                   content_layer, content_weight,
                   style_layers, style_weights, tv_weight,
                   init_random=False, num_iters=200, cnn=None):
    """
    Ejecuta la transferencia de estilo sobre una imagen.

    Parámetros:
    content_image, style_image (PIL.Image): imágenes de contenido y estilo.
    image_size (int): lado de la imagen sintetizada.
    style_size (int): tamaño al que se redimensiona la imagen de estilo.
    content_layer (int): índice de capa para la pérdida de contenido.
    content_weight (float): peso de la pérdida de contenido.
    style_layers (list[int]): índices de capas para la pérdida de estilo.
    style_weights (list[float]): pesos de las capas de estilo.
    tv_weight (float): peso de la regularización de variación total.
    init_random (bool): si True, inicia desde ruido; si no, desde la imagen de contenido.
    num_iters (int): iteraciones de optimización.
    cnn (nn.Module): red extractora de features (por defecto, cnn_estilo).
    """
    if cnn is None:
        cnn = cnn_estilo

    # ─── Features objetivo (se calculan una sola vez) ────────────────────────
    c_feats, content_x = features_from_img(content_image, image_size, cnn)
    content_target = c_feats[content_layer].clone()

    s_feats, _ = features_from_img(style_image, style_size, cnn)
    style_targets = [gram_matrix(s_feats[i].clone()) for i in style_layers]

    # ─── Imagen inicial ──────────────────────────────────────────────────────
    if init_random:
        img = torch.rand_like(content_x) * 2 - 1   # rango aprox [-1, 1]
    else:
        img = content_x.clone()
    img.requires_grad_(True)

    # ─── Optimizador con decay manual ────────────────────────────────────────
    lr_inicial, lr_decaido = 3.0, 0.1
    paso_decay = int(num_iters * 0.9)
    optimizer = torch.optim.Adam([img], lr=lr_inicial)

    # ─── Mostrar imágenes fuente ─────────────────────────────────────────────
    style_x = preprocess(style_image, style_size).to(device)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(deprocess(content_x)); axes[0].set_title("Contenido"); axes[0].axis('off')
    axes[1].imshow(deprocess(style_x));   axes[1].set_title("Estilo");    axes[1].axis('off')
    plt.show()

    # ─── Bucle de optimización ───────────────────────────────────────────────
    for t in range(num_iters):
        if t == paso_decay:
            optimizer = torch.optim.Adam([img], lr=lr_decaido)
        if t < int(num_iters * 0.95):
            # Clamp suave: evita que los píxeles escapen del rango normalizado.
            img.data.clamp_(-1.5, 1.5)

        optimizer.zero_grad()
        feats = extract_features(img, cnn)
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(img, tv_weight)
        loss = c_loss + s_loss + t_loss
        loss.backward()
        optimizer.step()

        if (t + 1) % (num_iters // 4) == 0 or t == num_iters - 1:
            print(f"iter {t+1:3d}  loss={loss.item():.2f}")
            plt.figure(figsize=(5, 5))
            plt.imshow(deprocess(img.data))
            plt.axis('off'); plt.title(f"iter {t+1}")
            plt.show()
```
::::


::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — Variación total + transferencia completa

**Objetivo:** Implementar la última pieza de la pérdida —la regularización de variación total— y ejecutar el pipeline completo en dos configuraciones distintas.

**Contexto matemático:**

La **variación total** suma las diferencias al cuadrado entre píxeles vecinos, penalizando imágenes con ruido de alta frecuencia. Para una imagen $x \in \mathbb{R}^{1 \times 3 \times H \times W}$,

$$L_{tv} = w_t \left( \sum_{c, i, j} (x_{c, i+1, j} - x_{c, i, j})^2 + \sum_{c, i, j} (x_{c, i, j+1} - x_{c, i, j})^2 \right)$$

es decir, diferencias cuadráticas en sentido vertical más diferencias cuadráticas en sentido horizontal.

**Parte A — Implementación:**

Implementá `tv_loss(img, tv_weight)` **sin bucles**, usando slicing de tensores.

> **Pistas:**
> - Diferencias verticales: `img[:, :, 1:, :] - img[:, :, :-1, :]` — el primer slice empieza desde la fila 1, el segundo termina en la anteúltima.
> - Análogamente para horizontales con la última dimensión.
> - Sumá los cuadrados de ambos y multiplicá por `tv_weight`.

**Parte B — Transferencia completa:**

Ejecutá `style_transfer` con los parámetros dados y observá cómo evoluciona la imagen.

**Parte C — Inversión de features:**

Repetí la llamada pero con dos cambios: `style_weights=[0, 0, 0, 0]` (sin pérdida de estilo) e `init_random=True` (imagen inicial aleatoria). El resultado es una **reconstrucción** de la imagen de contenido a partir solo de los feature maps intermedios, partiendo de ruido puro.
::::


::::cell{#ej4-code type=code role=student-code}
```python
# Tu código aquí (Parte A: tv_loss + test)
```

```python solution
def tv_loss(img, tv_weight):
    """
    Pérdida de variación total: suma de diferencias cuadráticas entre píxeles
    vecinos, vertical y horizontalmente.

    Parámetros:
    img (Tensor): imagen, shape (1, 3, H, W).
    tv_weight (float): peso de la regularización.

    Retorna:
    loss (Tensor): escalar con la pérdida.
    """
    diff_vertical   = ((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum()
    diff_horizontal = ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum()
    return tv_weight * (diff_vertical + diff_horizontal)


# ─── Test: imagen constante → tv_loss = 0; imagen con ruido → tv_loss > 0 ────
img_constante = torch.ones(1, 3, 32, 32)
assert tv_loss(img_constante, 1.0).item() == 0.0
img_ruido = torch.randn(1, 3, 32, 32)
assert tv_loss(img_ruido, 1.0).item() > 0.0
print("tv_loss OK")
```
::::


::::cell{#ej4-code-b type=code role=student-code}
```python
# Tu código aquí (Parte B: ejecutar style_transfer; Parte C: feature inversion)
```

```python solution
# ─── Parte B: transferencia con los parámetros sugeridos ─────────────────────
params = {
    'content_image':  content_img,
    'style_image':    style_img,
    'image_size':     192,
    'style_size':     192,
    'content_layer':  3,
    'content_weight': 6e-2,
    'style_layers':   [1, 4, 6, 7],
    'style_weights':  [300000, 1000, 15, 3],
    'tv_weight':      2e-2,
}
style_transfer(**params)

# ─── Parte C: feature inversion — reconstruir desde ruido, sin estilo ────────
# Apagamos la pérdida de estilo (pesos en 0) y arrancamos desde ruido.
# La imagen resultante muestra qué se puede reconstruir a partir solo de los
# feature maps de la capa de contenido.
params_inversion = dict(params)
params_inversion['style_weights'] = [0, 0, 0, 0]
params_inversion['init_random']   = True
style_transfer(**params_inversion)
```
::::


::::cell{#ej4-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

En la Parte C apagás la pérdida de estilo y arrancás desde ruido. El bucle de optimización **reconstruye** una imagen que se parece mucho a la de contenido original, pero con diferencias perceptibles.

- ¿Qué información **sí** conserva esa imagen reconstruida? ¿Qué información **no** conserva?
- Relacioná este experimento con la visualización de features por optimización (maximizar la activación de un canal): en ambos casos usamos la red para ir "hacia atrás" de un objetivo a una imagen. ¿En qué se parecen y en qué difieren?
::::


::::cell{#ej4-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- La imagen reconstruida conserva **todo lo que esté codificado en los feature maps de la capa de contenido**: la estructura general de la escena, la disposición de los objetos, sus bordes. No conserva necesariamente los píxeles exactos — los colores y los detalles finos pueden variar — porque una capa intermedia-profunda es invariante a esas variaciones. Cuanto más profunda sea la capa de contenido, más "ideas generales" quedan y menos píxeles exactos.
- La semejanza con la visualización de features es directa: en ambos casos congelamos la red y hacemos descenso de gradiente **sobre los píxeles** de una imagen hasta que ésta satisface un objetivo definido sobre activaciones internas. La diferencia es qué objetivo elegimos:
  - En visualización por activación, maximizar la activación de un canal específico → la imagen busca el patrón que "enciende" ese filtro.
  - En la Parte C de este ejercicio, igualar los feature maps de una capa entera al de una imagen dada → la imagen se convierte en una representación equivalente a esa imagen en ese nivel de abstracción.

En conjunto, estos experimentos muestran que una red preentrenada no es solo un clasificador: es un mapeo diferenciable entre el espacio de píxeles y un espacio de features, y podemos aprovechar esa diferenciabilidad para navegar en cualquiera de las dos direcciones.
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     CHECKLIST + CIERRE
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#checklist type=markdown role=checklist}
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Activé la GPU en Colab antes de empezar.
- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] Las celdas de test (`content_loss OK`, `gram_matrix OK`, `tv_loss OK`) imprimen correctamente.
- [ ] Las imágenes de transferencia de estilos se muestran (no quedaron cortadas ni en blanco).
- [ ] No modifiqué ninguna celda fuera de las de actividad.
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Completaste el Laboratorio 3c. Implementaste las cuatro piezas de la pérdida de transferencia de estilos (`content_loss`, `gram_matrix`, `style_loss`, `tv_loss`) y las usaste para sintetizar imágenes combinando el contenido de una foto con el estilo de una pintura. Usar una red preentrenada **como función de pérdida sobre los píxeles** es una de las formas más creativas de reutilizarla: la red no se entrena, se aprovecha su mapeo diferenciable entre píxeles y features para navegarlo en sentido inverso.
::::
