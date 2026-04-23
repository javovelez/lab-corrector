---
lab: "3a"
title: "Laboratorio n° 3: Transferencia de conocimiento"
subject: "Redes Neuronales Profundas"
block: "3 — Transferencia de conocimiento"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 3 — Transferencia de conocimiento
     Fuente única. Compilar con: python tools/lab_build.py _TPS/sources/Laboratorio_3.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_3.ipynb
             _TPS/Soluciones/Laboratorio_3_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 3 Parte A: Transferencia de conocimiento

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 3 — Transferencia de conocimiento

---

## Introducción

Entrenar una red profunda desde cero requiere millones de imágenes etiquetadas. En la práctica, rara vez tenemos esa cantidad de datos. La **transferencia de conocimiento** (*transfer learning*) resuelve este problema reutilizando las representaciones que una red ya aprendió sobre un dataset grande (típicamente ImageNet) para resolver nuevas tareas con muy pocos datos.

Para que esta idea funcione hay que entender qué es exactamente lo que una red preentrenada "sabe". Este laboratorio está organizado en tres bloques que avanzan desde la **interpretación** hacia la **aplicación**:

- Inspeccionar una red preentrenada y sus mapas de activación para ver qué captura cada capa.
- Visualizar un filtro individual optimizando la imagen que lo activa al máximo.
- Hacer *fine-tuning* de una red preentrenada sobre un dataset chico de imágenes satelitales (EuroSAT, 10 clases de uso del suelo).

> **Importante — GPU:** este laboratorio entrena redes y hace descenso de gradiente sobre imágenes. **Activá la GPU en Colab** cuando tengas que realizar un entrenamiento: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU*.

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para resolver cada ejercicio, consultá el material teórico de la Clase 3.
::::


::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::


::::cell{#imports type=code role=setup}
```python
# ─── Setup: imports y detección de GPU ──────────────────────────────────────
# Estos son los imports que usa el laboratorio de punta a punta, agrupados
# por función:
#   - torch / torchvision: modelos preentrenados, capas, optimizadores y
#     transformaciones de imágenes (todas las secciones).
#   - PIL / matplotlib: cargar y mostrar imágenes.
#   - os / requests: descargar la imagen de prueba de la Sección A desde
#     Internet. El dataset EuroSAT de la Sección C lo baja torchvision
#     automáticamente.
# También detectamos si hay GPU disponible y la guardamos en `device`:
# esa variable es la que usás en todos los `.to(device)` de tus ejercicios.
import os
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

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
# dos funciones que usás en casi todo el laboratorio:
#
#   - preprocess(pil_img, size): PIL → tensor normalizado listo para la red.
#   - deprocess(tensor):          tensor normalizado → array (H, W, 3) listo
#                                 para plt.imshow.
#
# Aparecen en los Ej 1 y 2 (forward normal) y Ej 3-5 (visualización por
# optimización). Las constantes IMAGENET_MEAN / IMAGENET_STD además se
# usan directamente en el Ej 6 para armar los pipelines de data augmentation.
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
     SECCIÓN A: Inspección de una red preentrenada
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#secA type=markdown role=section}
---
## Sección A: Un vistazo a una red preentrenada
::::


::::cell{#intro-secA type=markdown role=intro}
### ¿Qué guarda una red preentrenada?

Una ResNet-18 preentrenada sobre ImageNet es una función que mapea imágenes RGB a vectores de 1000 números (uno por clase). Pero el grueso de su valor no está en esa capa final, sino en las **capas intermedias**: miles de filtros convolucionales que aprendieron a detectar bordes, texturas, partes de objetos y composiciones visuales que generalizan mucho más allá de las 1000 clases de ImageNet.

En esta sección vamos a cargar la red, usarla tal cual viene, y espiar qué hacen sus mapas de activación sobre una imagen real.
::::


::::cell{#setup-imagen-prueba type=code role=setup}
```python
# ─── Setup: imagen de prueba y clases de ImageNet ───────────────────────────
# Usamos la foto de un Labrador que mantiene el repo oficial pytorch/hub y la
# lista de 1000 clases que espera cualquier red preentrenada en ImageNet.
# Dejamos cargadas `img_prueba` y `imagenet_classes` — las usás en los Ej 1
# y 2 para probar la red y leer los nombres de las clases predichas.
url_imagen  = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
url_classes = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
ruta_imagen = "dog.jpg"

if not os.path.exists(ruta_imagen):
    with open(ruta_imagen, "wb") as f:
        f.write(requests.get(url_imagen).content)

img_prueba = Image.open(ruta_imagen).convert("RGB")
imagenet_classes = requests.get(url_classes).text.strip().split("\n")

plt.figure(figsize=(4, 4))
plt.imshow(img_prueba); plt.axis('off')
plt.title("Imagen de prueba")
plt.show()
print(f"Total de clases de ImageNet: {len(imagenet_classes)}")
```
::::


::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Cargar ResNet-18 y clasificar una imagen

**Objetivo:** Familiarizarte con la API de `torchvision.models`: cargar una red preentrenada, pasarla a modo evaluación y usarla para predecir la clase de una imagen nueva.

**Enunciado:**

1. Cargá una **ResNet-18 preentrenada** sobre ImageNet usando `models.resnet18(weights='DEFAULT')` y guardala en la variable `modelo`. Movela al `device`.
2. Ponela en modo evaluación con `.eval()`.
3. Preprocesá `img_prueba` con la función `preprocess` (tamaño 224) y enviá el tensor al `device`.
4. Hacé el forward pass dentro de un bloque `with torch.no_grad()` para obtener el vector de logits de forma `(1, 1000)`.
5. Convertí los logits en probabilidades con `torch.softmax`, y mostrá las **top-5 clases** con su probabilidad usando `imagenet_classes`.

> **Pista:** `torch.topk(probs, k=5)` te devuelve dos tensores: las top-k probabilidades y sus índices correspondientes. Iterá sobre los índices para imprimir el nombre de clase.
::::


::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Paso 1: cargar ResNet-18 preentrenada ───────────────────────────────────
# 'DEFAULT' toma los pesos más recientes recomendados por torchvision.
modelo = models.resnet18(weights='DEFAULT').to(device)

# ─── Paso 2: modo evaluación ─────────────────────────────────────────────────
# .eval() desactiva dropout y fija los running stats de BatchNorm.
modelo.eval()

# ─── Paso 3: preprocesar la imagen ───────────────────────────────────────────
x = preprocess(img_prueba, size=224).to(device)
print(f"Tensor de entrada: {tuple(x.shape)}")

# ─── Paso 4: forward pass sin construir el grafo de gradientes ───────────────
with torch.no_grad():
    logits = modelo(x)
print(f"Logits:            {tuple(logits.shape)}")

# ─── Paso 5: top-5 clases ────────────────────────────────────────────────────
probs = torch.softmax(logits, dim=1)[0]
valores, indices = torch.topk(probs, k=5)
print("\nTop-5 predicciones:")
for p, i in zip(valores, indices):
    print(f"  {imagenet_classes[i]:<30s}  p = {p.item():.4f}")
```
::::


::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Qué cambia al llamar `modelo.eval()` antes de la inferencia? ¿Por qué es importante hacerlo en una red que tiene capas `BatchNorm` y `Dropout`?
::::


::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

`.eval()` cambia el modo de ejecución de todas las capas del modelo. En particular:

- **`BatchNorm`**: durante el entrenamiento, normaliza usando la media y el desvío del lote actual; en `eval` usa las estadísticas **acumuladas** durante el entrenamiento (*running mean/var*). Sin `.eval()`, una inferencia con `batch_size=1` produciría estadísticas degeneradas y predicciones erráticas.
- **`Dropout`**: durante el entrenamiento apaga una fracción aleatoria de neuronas; en `eval` las deja todas activas y escala coherentemente la salida. Sin `.eval()`, cada llamada a la misma imagen daría una predicción distinta.

Llamar `.eval()` es condición necesaria para que la red se comporte de forma determinista y con las estadísticas que aprendió durante el entrenamiento.
```
::::


::::cell{#setup-feature-hook type=code role=setup}
```python
# ─── Setup: utilidades para inspeccionar activaciones ──────────────────────
# Esta celda define `get_feature_maps`, la función que usás en los Ej 2-5
# para leer activaciones de la red: dada una entrada `x` y el nombre de una
# capa, devuelve las activaciones de esa capa (vía forward hook). Si omitís
# `layer_name`, devuelve la salida final del modelo (los logits).
#
# Detalle importante para la Sección B: cuando `x` es una imagen que estamos
# OPTIMIZANDO (tiene requires_grad=True), la función aplica dos regularizadores
# DE FORMA TRANSPARENTE antes del forward:
#
#   1. JITTER — desplaza la imagen aleatoriamente en X e Y. Obliga al
#      optimizador a producir patrones robustos a traslación, lo que filtra
#      el ruido adversarial de alta frecuencia.
#   2. BLUR GAUSSIANO estocástico — con baja probabilidad aplica un filtro
#      gaussiano suave. Suprime residuos de alta frecuencia que sobreviven al
#      jitter y deja emerger texturas reconocibles.
#
# Cuando `x` es una imagen normal del dataset (requires_grad=False, como en el
# Ej 2), la función NO regulariza nada — el análisis de un forward sobre una
# imagen real debe ser fiel a lo que la red vería.
#
# Si querés ver el efecto, en los Ejs 3-5 probá comentando la llamada a
# _regularizar y comparar: los patrones aparecerán mucho más ruidosos.

def _make_gaussian_kernel(size=5, sigma=1.0):
    """Devuelve un kernel gaussiano 2D normalizado, shape (1, 1, size, size)."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    return kernel_2d.view(1, 1, size, size)


# Kernel gaussiano 5x5 replicado para los 3 canales RGB (se usa con groups=3).
_GAUSSIAN_KERNEL = _make_gaussian_kernel(size=5, sigma=1.0).to(device)
_GAUSSIAN_KERNEL_3CH = _GAUSSIAN_KERNEL.expand(3, 1, 5, 5).contiguous()


def _regularizar(x, max_shift=16, blur_prob=0.1):
    """
    Aplica jitter (desplazamiento aleatorio) y, con probabilidad `blur_prob`,
    un blur gaussiano 5x5. Ambas operaciones preservan el grafo de gradientes:
    torch.roll es una permutación y F.conv2d con un kernel fijo es diferenciable.
    """
    dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
    if torch.rand(1).item() < blur_prob:
        x = F.conv2d(x, _GAUSSIAN_KERNEL_3CH, padding=2, groups=3)
    return x


def get_feature_maps(model, x, layer_name=None):
    """
    Ejecuta un forward pass y devuelve activaciones del modelo.

    Parámetros:
    model (nn.Module): modelo a inspeccionar.
    x (Tensor): entrada de forma (1, 3, H, W).
    layer_name (str o None): nombre de una capa intermedia (como aparece en
        `model.named_modules()`). Si es None, retorna la salida final del
        modelo (los logits).

    Retorna:
    feats (Tensor): activaciones de la capa pedida, o logits si layer_name=None.

    Nota: si x.requires_grad es True, x se pasa por _regularizar antes del
    forward (ver explicación arriba). Esto hace que los Ejercicios 3-5
    produzcan patrones interpretables sin que el alumno tenga que encargarse
    del detalle de la regularización.
    """
    if x.requires_grad:
        x = _regularizar(x)

    if layer_name is None:
        return model(x)

    modulo = dict(model.named_modules())[layer_name]
    salida = {}
    def hook(_mod, _inp, out):
        salida['feat'] = out
    handle = modulo.register_forward_hook(hook)
    try:
        model(x)
    finally:
        handle.remove()  # importante: desregistrar para no acumular hooks
    return salida['feat']
```
::::


::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Visualizar mapas de activación

**Objetivo:** Observar directamente qué patrones detecta cada canal de una capa intermedia al pasar una imagen real por la red.

**Contexto:**

Una capa convolucional produce un tensor de forma `(1, C, H', W')`: `C` mapas de activación (uno por filtro) de resolución espacial `H' × W'`. Cada mapa es, literalmente, una imagen que indica dónde se "enciende" el filtro en la entrada. El setup provee la función `get_feature_maps(model, x, layer_name)` que ejecuta un forward pass y extrae las activaciones de la capa indicada.

**Enunciado:**

1. Preprocesá `img_prueba` con `preprocess` (tamaño 224) y enviala al `device`.
2. Usando `get_feature_maps`, extraé las activaciones de la capa `'layer1'` de tu ResNet-18 del ejercicio anterior. Imprimí la forma del tensor.
3. Mostrá en una grilla de **2 × 4** los primeros 8 canales del tensor (`feats[0, i]`) usando `plt.imshow` con `cmap='viridis'` y título `f"Canal {i}"`. Apagá los ejes con `ax.axis('off')`.
4. Repetí el punto 3 para la capa `'layer3'`. Comparalos visualmente.

> **Pista:** Para acceder a cada canal individualmente podés hacer `feats[0, i].detach().cpu().numpy()`. Usá `fig, axes = plt.subplots(2, 4, figsize=(12, 6))` y `axes.flatten()` para iterar.
::::


::::cell{#ej2-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Preprocesar la imagen ───────────────────────────────────────────────────
x = preprocess(img_prueba, size=224).to(device)

# ─── Función auxiliar para graficar los primeros 8 canales ───────────────────
def plot_primeros_canales(feats, titulo):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        mapa = feats[0, i].detach().cpu().numpy()
        ax.imshow(mapa, cmap='viridis')
        ax.set_title(f"Canal {i}")
        ax.axis('off')
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()

# ─── Capa temprana: layer1 ───────────────────────────────────────────────────
feats_1 = get_feature_maps(modelo, x, 'layer1')
print(f"layer1: {tuple(feats_1.shape)}")
plot_primeros_canales(feats_1, "Mapas de activación — layer1 (temprana)")

# ─── Capa más profunda: layer3 ───────────────────────────────────────────────
feats_3 = get_feature_maps(modelo, x, 'layer3')
print(f"layer3: {tuple(feats_3.shape)}")
plot_primeros_canales(feats_3, "Mapas de activación — layer3 (profunda)")
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Comparando las dos grillas, ¿qué diferencias notás entre los mapas de `layer1` y los de `layer3`?

- ¿Cómo cambia la resolución espacial?
- ¿Qué tipo de patrones parecen "encenderse" en cada caso (bordes, texturas, partes del objeto)?
- ¿Por qué en `layer3` muchos canales aparecen casi negros, mientras en `layer1` todos tienen respuesta?
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- **Resolución espacial:** `layer1` produce mapas de `56×56` y `layer3` de `14×14`. Cada submuestreo (stride 2 y max-pooling) reduce a la mitad la resolución y dobla la cantidad de canales, a cambio de un **campo receptivo** más grande: cada píxel de `layer3` resume información de una región mucho mayor de la imagen original.
- **Qué detectan:** los canales de `layer1` responden a patrones muy locales — bordes en distintas orientaciones, transiciones de color, texturas simples. En `layer3` los canales responden a combinaciones de esos patrones (partes de objetos, texturas complejas), y solo algunos se activan ante una imagen concreta porque cada filtro está especializado en un concepto visual específico.
- **Canales "apagados" en `layer3`:** a medida que se sube en la jerarquía, los filtros se vuelven más selectivos. Un filtro entrenado para detectar, digamos, "ojo de perro" se activará ante fotos de perros y quedará casi en cero ante una foto de una catedral. En `layer1`, los bordes son ubicuos, así que prácticamente todos los canales se encienden.
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     SECCIÓN B: Visualización de features por optimización
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#secB type=markdown role=section}
---
## Sección B: Visualización de features por optimización
::::


::::cell{#intro-secB type=markdown role=intro}
### ¿Qué busca un filtro?

En la sección anterior miramos qué parte de una imagen real activa cada filtro. Eso contesta la pregunta "¿dónde se enciende?", pero no contesta "¿qué está buscando?". Para eso hay una técnica complementaria: **invertir el problema**. En vez de pasar una imagen por la red y mirar la activación, vamos a **mantener la red fija** y buscar, por descenso de gradiente, la imagen que maximiza la activación promedio de un canal específico.

El resultado es una imagen "ideal" para ese filtro: si el filtro detecta bordes diagonales, la imagen va a parecer un patrón de rayas. Si detecta ojos, va a parecer algo con ojos. La red no se entrena: la imagen es el parámetro que se optimiza.
::::


::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — Optimizar la imagen que activa un filtro

**Objetivo:** Implementar el bucle de optimización que, manteniendo la red congelada, actualiza los píxeles de una imagen ruidosa hasta maximizar la activación promedio de un canal elegido.

**Contexto:**

Vamos a usar el mismo `modelo` (ResNet-18 preentrenada) del ejercicio anterior. Los pesos de la red no se entrenan: el único parámetro que se actualiza es la imagen. Para calcular el gradiente respecto de los píxeles, la imagen debe tener `requires_grad=True`.

**Enunciado:**

Implementá la función `visualizar_canal(model, layer_name, channel, num_iters=30, lr=0.1, size=224)`:

1. Generá una imagen inicial llamada `img` con ruido centrado y pequeño: `torch.randn(1, 3, size, size, device=device) * 0.1`. Marcala con `.requires_grad_(True)`.
2. Creá un optimizador `torch.optim.Adam([img], lr=lr, weight_decay=1e-6)` — el único parámetro a optimizar es la imagen.
3. En cada iteración (está **permitido usar un bucle `for`** porque es un loop de optimización):
   - `optimizer.zero_grad()`.
   - Extraé las activaciones de `layer_name` con `get_feature_maps(model, img, layer_name)`.
   - La pérdida es **menos la media del canal elegido**: `loss = -feats[0, channel].mean()`. Queremos **maximizar** la activación, así que minimizamos el negativo.
   - `loss.backward()` y `optimizer.step()`.
4. Al final, retorná la imagen optimizada (sin gradientes, con `.detach()`).

Probala para `layer_name='layer1', channel=5` y mostrá el resultado con `plt.imshow(deprocess(img))`.

> **Pistas:**
> - Como la red está en `.eval()` y queremos congelar sus pesos, no hace falta pasar sus parámetros al optimizador: el único parámetro a optimizar es `img`.
> - `deprocess` está definida en el setup global y devuelve un array listo para `imshow`.
::::


::::cell{#ej3-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
def visualizar_canal(model, layer_name, channel, num_iters=30, lr=0.1, size=224):
    """
    Optimiza una imagen ruidosa para maximizar la activación promedio de un
    canal específico de una capa del modelo.

    Parámetros:
    model (nn.Module): red preentrenada (con .eval() aplicado).
    layer_name (str): nombre de la capa objetivo.
    channel (int): índice del canal a maximizar.
    num_iters (int): iteraciones de descenso de gradiente.
    lr (float): learning rate de Adam.
    size (int): lado de la imagen generada.

    Retorna:
    img (Tensor): imagen optimizada, forma (1, 3, size, size), sin gradientes.
    """
    # ─── Inicialización: ruido centrado, con requires_grad ───────────────────
    img = (torch.randn(1, 3, size, size, device=device) * 0.1).requires_grad_(True)

    # ─── Optimizador: solo la imagen ─────────────────────────────────────────
    # weight_decay actúa como regularización L2 y evita que los píxeles se
    # vayan a valores absurdos.
    optimizer = torch.optim.Adam([img], lr=lr, weight_decay=1e-6)

    # ─── Bucle de optimización ───────────────────────────────────────────────
    for _ in range(num_iters):
        optimizer.zero_grad()
        feats = get_feature_maps(model, img, layer_name)
        # Maximizar la activación promedio → minimizar su negativo.
        loss = -feats[0, channel].mean()
        loss.backward()
        optimizer.step()

    return img.detach()


# ─── Prueba: canal 5 de layer1 ───────────────────────────────────────────────
img_viz = visualizar_canal(modelo, 'layer1', channel=5, num_iters=30, lr=0.1)
plt.figure(figsize=(4, 4))
plt.imshow(deprocess(img_viz))
plt.title("layer1 · canal 5")
plt.axis('off')
plt.show()
```
::::


::::cell{#ej3-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Durante el bucle, los pesos de la red permanecen fijos y lo que se actualiza en cada paso son los **píxeles de la imagen**. Explicá, en términos de la regla de la cadena, cómo llega el gradiente desde la pérdida hasta los píxeles de entrada. ¿Qué rol cumple `requires_grad=True` sobre la imagen?
::::


::::cell{#ej3-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

El grafo computacional que construye PyTorch va desde los píxeles de la imagen hacia adelante, pasando por todas las capas del modelo, hasta la pérdida. Cuando llamamos `loss.backward()`, la regla de la cadena propaga derivadas en sentido inverso:

$$\frac{\partial L}{\partial \text{pixel}_{ij}} \;=\; \frac{\partial L}{\partial \text{feat}} \cdot \frac{\partial \text{feat}}{\partial \text{capa}_k} \cdots \frac{\partial \text{capa}_1}{\partial \text{pixel}_{ij}}$$

`requires_grad=True` le dice a PyTorch: "este tensor es una **hoja entrenable** del grafo, acumulá el gradiente de la pérdida respecto a él en `.grad`". Sin esa marca, PyTorch trataría los píxeles como datos fijos y no guardaría gradientes; el optimizador no tendría nada que actualizar.

Los parámetros del modelo **no** se mueven porque no están en la lista del optimizador (`Adam([img], ...)`): aunque sí se calculan sus gradientes durante el backward, nadie los aplica.
```
::::


::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — Jerarquía de features: capa temprana vs. capa profunda

**Objetivo:** Usar `visualizar_canal` para comparar qué tipo de patrones se forman cuando optimizamos un canal de una capa temprana y uno de una capa profunda.

**Enunciado:**

1. Optimizá dos canales distintos durante **60 iteraciones** con `lr=0.1`:
   - Un canal de una **capa temprana**: `layer1`, canal `3`.
   - Un canal de una **capa profunda**: `layer4`, canal `17`.
2. Mostrá ambas imágenes lado a lado en una figura de `1 × 2` con `plt.subplots`. Usá `deprocess` para la visualización y ponele título a cada subplot.

> **Pista:** Si la capa profunda tarda más en converger, aumentá `num_iters` o `lr`. Para algunos canales puede hacer falta correrlo un par de veces hasta obtener un patrón reconocible: el bucle parte de ruido distinto en cada ejecución.
::::


::::cell{#ej4-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Optimizar un canal de layer1 y otro de layer4 ───────────────────────────
img_temprana = visualizar_canal(modelo, 'layer1', channel=3,  num_iters=60, lr=0.1)
img_profunda = visualizar_canal(modelo, 'layer4', channel=17, num_iters=60, lr=0.1)

# ─── Mostrar ambas imágenes lado a lado ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(deprocess(img_temprana))
axes[0].set_title("layer1 · canal 3 (temprana)")
axes[0].axis('off')
axes[1].imshow(deprocess(img_profunda))
axes[1].set_title("layer4 · canal 17 (profunda)")
axes[1].axis('off')
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej4-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Comparando visualmente las dos imágenes:

- ¿Qué tipo de patrones dominan en la visualización de `layer1`? ¿Qué sugiere eso sobre lo que detecta un filtro temprano?
- ¿Qué tipo de patrones aparecen en `layer4`? ¿Son más locales o más globales? ¿Más simples o más estructurados?
- ¿Cómo se conecta esta observación con la idea de que "las capas tempranas aprenden features generales y las profundas son más específicas de la tarea"?
::::


::::cell{#ej4-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- **`layer1`** suele mostrar patrones muy locales y periódicos: rayas, bordes en una orientación, transiciones de color. Son los átomos visuales básicos. Tiene sentido que sean generales porque aparecen en cualquier imagen natural: una arista no es específica de "perro" ni de "auto".
- **`layer4`** muestra patrones mucho más complejos y estructurados: combinaciones de texturas, formas repetidas, a veces configuraciones que recuerdan partes de objetos (ojos, ruedas, plumas). Cada canal se especializa en un concepto visual completo.
- Esto ilustra por qué el *transfer learning* funciona tan bien: las primeras capas reutilizan los mismos "átomos visuales" para cualquier tarea de visión, mientras que las últimas son las que más hay que adaptar al nuevo dataset. En la Sección C vamos a aprovechar exactamente esta propiedad: congelaremos (o casi) las primeras capas y ajustaremos las últimas para una tarea nueva.
```
::::


::::cell{#ej5-enunciado type=markdown role=enunciado}
### Ejercicio 5 — ¿Qué imagen activa una clase completa?

**Objetivo:** Extender la técnica del Ejercicio 3 al último nivel de la red: en vez de maximizar un canal de una capa intermedia, maximizar **el logit de una clase de salida** de ResNet-18. El resultado es una imagen que la red interpreta con máxima confianza como esa clase.

**Contexto:**

`layer4` es lo más profundo que tiene ResNet-18 en términos convolucionales, pero después vienen `avgpool` y `fc` (la capa lineal 512→1000 que mapea features a las 1000 clases de ImageNet). Cada una de esas 1000 salidas es un logit asociado a una clase concreta: el índice 207 corresponde a *golden retriever*, el 281 a *tabby cat*, el 340 a *zebra*, el 999 a *toilet paper*.

Si aplicamos la misma técnica de descenso de gradiente sobre los píxeles pero como objetivo ponemos "maximizá el logit de la clase N", obtenemos una imagen que condensa lo que ResNet-18 "sabe" sobre esa clase. Este resultado es el corazón visual de DeepDream y de la interpretabilidad por optimización.

**Enunciado:**

Implementá la función `visualizar_clase(model, clase_idx, num_iters=200, lr=0.1, size=224)`:

1. Inicializá una imagen con ruido pequeño y `requires_grad=True`, igual que en el Ej 3.
2. Creá un optimizador Adam sobre la imagen, con `weight_decay=1e-6`.
3. En el bucle de optimización:
   - Obtené los logits con `get_feature_maps(model, img)` — sin pasar `layer_name`, la función devuelve la salida final del modelo.
   - La pérdida es **menos el logit de la clase elegida**: `loss = -logits[0, clase_idx]`.
   - `backward()` + `step()`.
4. Al final, retorná la imagen optimizada (sin gradientes).

Probala con **dos clases distintas** de ImageNet y mostrá ambas lado a lado con el nombre de la clase en el título.

> **Pistas:**
> - `get_feature_maps(model, img)` devuelve un tensor de forma `(1, 1000)` cuando se omite `layer_name`.
> - Buenos índices para probar: `207` (golden retriever), `281` (tabby cat), `340` (zebra), `947` (mushroom), `971` (bubble).
> - Una regularización (jitter + blur estocástico) se aplica **automáticamente** dentro de `get_feature_maps` cuando `img.requires_grad=True`. Revisá la explicación en el setup de la función si te interesa el detalle.
::::


::::cell{#ej5-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
def visualizar_clase(model, clase_idx, num_iters=200, lr=0.1, size=224):
    """
    Optimiza una imagen ruidosa para maximizar el logit de una clase de salida.
    La regularización de imagen (jitter + blur estocástico) se aplica dentro
    de get_feature_maps cuando img.requires_grad=True.

    Parámetros:
    model (nn.Module): red preentrenada.
    clase_idx (int): índice de la clase en el vector de 1000 salidas.
    num_iters (int): iteraciones de descenso de gradiente.
    lr (float): learning rate de Adam.
    size (int): lado de la imagen generada.

    Retorna:
    img (Tensor): imagen optimizada, forma (1, 3, size, size), sin gradientes.
    """
    img = (torch.randn(1, 3, size, size, device=device) * 0.1).requires_grad_(True)
    optimizer = torch.optim.Adam([img], lr=lr, weight_decay=1e-6)

    for _ in range(num_iters):
        optimizer.zero_grad()
        # Sin layer_name, get_feature_maps devuelve los logits finales de la red.
        logits = get_feature_maps(model, img)
        # Maximizar el logit de la clase -> minimizar su negativo.
        loss = -logits[0, clase_idx]
        loss.backward()
        optimizer.step()

    return img.detach()


# ─── Dos clases distintas ────────────────────────────────────────────────────
clase_a, clase_b = 207, 340   # golden retriever, zebra
img_a = visualizar_clase(modelo, clase_a, num_iters=200)
img_b = visualizar_clase(modelo, clase_b, num_iters=200)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(deprocess(img_a))
axes[0].set_title(f"Clase {clase_a}: {imagenet_classes[clase_a]}")
axes[0].axis('off')
axes[1].imshow(deprocess(img_b))
axes[1].set_title(f"Clase {clase_b}: {imagenet_classes[clase_b]}")
axes[1].axis('off')
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej5-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Por qué optimizar el **logit** de la clase (pre-softmax) funciona mejor que optimizar la **probabilidad** (post-softmax)?

Pensá qué pasa con el gradiente en cada caso:

- Con logit: derivada directa respecto al valor pre-softmax.
- Con probabilidad: $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$. Para aumentar $p_i$ se puede **subir $z_i$** o **bajar $z_j$ para $j \neq i$**. La segunda vía suele ser más fácil para el optimizador.
::::


::::cell{#ej5-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Maximizar el **logit** de la clase objetivo es pedirle a la red: "encontrá la imagen que produce la máxima *evidencia* en esa dirección". El gradiente respecto al píxel apunta directamente hacia el patrón que activa la clase.

Maximizar la **probabilidad** post-softmax, en cambio, abre un atajo perverso para el optimizador: en vez de subir el logit de la clase deseada (hacer una imagen *más parecida* a esa clase), puede bajar los logits de **las otras 999 clases** (hacer una imagen *menos parecida a todo lo demás*). El resultado tiende a ser ruido adversarial, no un retrato reconocible de la clase.

Por eso en visualización de features y DeepDream siempre se optimizan logits pre-softmax. La moraleja es general: la red preentrenada es un mapeo diferenciable muy rico, y **qué derivada elegimos** define qué solución emerge.
```
::::


<!-- ══════════════════════════════════════════════════════════════════════
     SECCIÓN C: Fine-tuning
     ══════════════════════════════════════════════════════════════════════ -->


::::cell{#secC type=markdown role=section}
---
## Sección C: Fine-tuning sobre un dataset nuevo
::::


::::cell{#intro-secC type=markdown role=intro}
### La idea en un párrafo

El *fine-tuning* aplica la intuición de la Sección B: los features aprendidos sobre ImageNet son lo bastante generales como para resolver otras tareas de visión con pocos datos. La receta estándar tiene cuatro pasos:

1. Partir de una red preentrenada sobre un dataset grande (ImageNet).
2. Reemplazar la capa de salida por una nueva, dimensionada para el problema objetivo.
3. Inicializar esa capa nueva al azar y dejar el resto con los pesos preentrenados.
4. Entrenar la red entera, pero con **learning rate bajo para el backbone** y **learning rate alto para la capa nueva** — así perturbamos poco lo que ya funciona y aprendemos rápido lo que falta.

Vamos a aplicarla sobre un dataset chiquito y lejano a ImageNet: **EuroSAT**, 3000 imágenes satelitales RGB (300 por clase) etiquetadas en **10 clases de uso del suelo** (bosque, cultivos, residencial, industrial, ríos, mar, etc.). Es un dominio visualmente muy distinto a las fotos naturales de ImageNet: textura cenital, sin objetos centrados, sin nociones de "arriba" y "abajo". Si el fine-tuning funciona acá, confirma que los features preentrenados son genuinamente generales.
::::


::::cell{#setup-eurosat type=code role=setup}
```python
# ─── Setup: descargar EuroSAT y pre-computar el split estratificado ─────────
# EuroSAT es un dataset de ~27.000 imágenes satelitales RGB (Sentinel-2) de
# 64×64 píxeles, etiquetadas en 10 clases de uso del suelo. `torchvision` lo
# descarga la primera vez (~90 MB) y lo cachea en `./data/`; en corridas
# posteriores no vuelve a bajarlo.
#
# Para mantener el laboratorio ágil usamos un SUBCONJUNTO estratificado:
# 300 imágenes de train + 100 de test por clase (4000 en total). Los índices
# del split los fijamos con semilla 42 para que todos trabajemos sobre la
# misma partición y los resultados sean comparables entre grupos.
from collections import defaultdict
import random

os.makedirs("data", exist_ok=True)
torchvision.datasets.EuroSAT(root='data', download=True)  # descarga si falta
eurosat_root = 'data'

# Instancia auxiliar sin transform: solo la usamos para enumerar archivos y
# armar el split. En el Ej 6 vas a crear dos instancias "reales" con los
# transforms de train y test.
_eurosat_idx = torchvision.datasets.EuroSAT(root=eurosat_root)
CLASES_EUROSAT = _eurosat_idx.classes
print(f"Clases ({len(CLASES_EUROSAT)}): {CLASES_EUROSAT}")

TRAIN_POR_CLASE = 300
TEST_POR_CLASE  = 100

_rng = random.Random(42)
_por_clase = defaultdict(list)
for idx, (_, label) in enumerate(_eurosat_idx.samples):
    _por_clase[label].append(idx)

train_idx, test_idx = [], []
for cls, idxs in _por_clase.items():
    _rng.shuffle(idxs)
    train_idx.extend(idxs[:TRAIN_POR_CLASE])
    test_idx.extend(idxs[TRAIN_POR_CLASE:TRAIN_POR_CLASE + TEST_POR_CLASE])

print(f"Índices preparados — Train: {len(train_idx)}  |  Test: {len(test_idx)}")
```
::::


::::cell{#ej6-enunciado type=markdown role=enunciado}
### Ejercicio 6 — Preparar los DataLoaders

**Objetivo:** Construir las pipelines de preprocesamiento para entrenamiento y prueba, y cargar el subset estratificado de EuroSAT en `DataLoader`s listos para el bucle de entrenamiento.

**Enunciado:**

1. Definí un pipeline de **train** con `transforms.Compose([...])` que aplique:
   - `Resize((224, 224))` — los tiles de EuroSAT son de 64×64; los llevamos directo al tamaño de entrada de ResNet-18.
   - `RandomHorizontalFlip()`.
   - `RandomVerticalFlip()` — las imágenes satelitales no tienen un "arriba" canónico, así que el flip vertical es un augmentation válido (en fotografía natural no lo sería).
   - `ToTensor()`.
   - `Normalize(IMAGENET_MEAN, IMAGENET_STD)`.
2. Definí un pipeline de **test** equivalente pero sin augmentation: `Resize((224, 224))`, `ToTensor`, `Normalize`.
3. Creá dos datasets con `torchvision.datasets.EuroSAT`:
   - `train_full`: apuntando a `eurosat_root`, con `transform=train_tf`.
   - `test_full`: apuntando a `eurosat_root`, con `transform=test_tf`.
4. Envolvelos en los subconjuntos estratificados que ya tenés precomputados:
   - `train_ds = Subset(train_full, train_idx)`.
   - `test_ds  = Subset(test_full,  test_idx)`.
5. Creá los `DataLoader`s correspondientes con `batch_size=64`, `shuffle=True` para train, `shuffle=False` para test.
6. Imprimí el tamaño de cada subset y las clases detectadas (`train_full.classes`).
7. Mostrá **4 imágenes del lote de train** en una grilla `1 × 4` para verificar que el pipeline funciona. Recordá desnormalizar antes de mostrar y poné el nombre de la clase como título.

> **Pista:** `next(iter(train_loader))` te devuelve el primer lote `(X, y)`. Para desnormalizar un tensor de forma `(3, H, W)` normalizado con ImageNet, multiplicá por `IMAGENET_STD.view(3, 1, 1)` y sumá `IMAGENET_MEAN.view(3, 1, 1)`.
::::


::::cell{#ej6-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Pipelines de transformaciones ───────────────────────────────────────────
# Resize directo a 224 (sin RandomResizedCrop: recortar un tile de 64×64
# pierde la textura que caracteriza la clase). Flips H y V como augmentation:
# son válidos porque las imágenes cenitales no tienen orientación canónica.
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
])
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
])

# ─── Datasets y DataLoaders ──────────────────────────────────────────────────
# Dos instancias apuntando al mismo disco, una con cada transform, y los
# Subset aplicados con los índices estratificados ya precomputados.
train_full = torchvision.datasets.EuroSAT(root=eurosat_root, transform=train_tf)
test_full  = torchvision.datasets.EuroSAT(root=eurosat_root, transform=test_tf)
train_ds = Subset(train_full, train_idx)
test_ds  = Subset(test_full,  test_idx)

# num_workers=0 mantiene la carga de datos en el proceso principal. Es la
# opción más portable: en Colab T4 el costo es mínimo porque los tiles son
# chicos y la carga no es cuello de botella; en M3/MPS evita el overhead de
# multiprocessing (que ahí puede triplicar el tiempo por época).
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} imágenes  |  Test: {len(test_ds)} imágenes")
print(f"Clases: {train_full.classes}")

# ─── Visualizar un lote ──────────────────────────────────────────────────────
X_batch, y_batch = next(iter(train_loader))
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    img = X_batch[i] * IMAGENET_STD.view(3, 1, 1) + IMAGENET_MEAN.view(3, 1, 1)
    ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
    ax.set_title(train_full.classes[y_batch[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej6-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Por qué normalizamos con las medias y desvíos **de ImageNet**, y no con los del dataset EuroSAT? Pensá que las imágenes satelitales tienen una distribución de color muy distinta a las fotos naturales (mucho verde en bosque/cultivo, mucho azul en agua). ¿Qué pasaría con los pesos preentrenados de ResNet-18 si entrenáramos con otra normalización?
::::


::::cell{#ej6-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La red ResNet-18 fue entrenada con imágenes normalizadas usando `IMAGENET_MEAN` e `IMAGENET_STD`. Eso significa que **todos sus pesos están calibrados para recibir entradas en esa escala**: los filtros del primer `Conv2d`, por ejemplo, "saben" que un píxel blanco corresponde a un valor cercano a `(1-mean)/std ≈ 2.25`, no a `1.0`.

Si normalizáramos con medias y desvíos del dataset EuroSAT (que son muy distintos — imágenes cenitales con predominio de verde y azul), las entradas caerían en un rango distinto al que vieron los pesos durante el preentrenamiento, y aprovecharíamos mucho peor ese conocimiento. En la práctica, la regla es: **si usás un modelo preentrenado, normalizás como se normalizó durante su entrenamiento** — aun cuando el dominio objetivo sea claramente distinto. Esa aparente "desalineación" es parte del juego: el resto de la red se encarga de acomodar las estadísticas del nuevo dominio.
```
::::


::::cell{#ej7-enunciado type=markdown role=enunciado}
### Ejercicio 7 — Construir el modelo de fine-tuning

**Objetivo:** Armar una red que reutilice los pesos preentrenados de ResNet-18 y tenga una capa de salida nueva, adaptada a las **10 clases** del dataset EuroSAT.

**Enunciado:**

1. Instanciá una `models.resnet18(weights='DEFAULT')` y asignala a `finetune_net`.
2. Mirá la arquitectura de su capa final imprimiendo `finetune_net.fc`. Vas a ver algo como `Linear(in_features=512, out_features=1000, bias=True)`.
3. Reemplazá `finetune_net.fc` por una nueva `nn.Linear(in_features, 10)`, tomando `in_features` desde el atributo correspondiente de la capa original.
4. Inicializá los pesos de esa nueva capa con **Xavier uniforme** (`nn.init.xavier_uniform_`) y el bias en cero.
5. Movela al `device` e imprimí nuevamente `finetune_net.fc` para verificar el cambio.

> **Pista:** `in_features` está disponible como `finetune_net.fc.in_features`. Usarlo en lugar de escribir `512` a mano hace el código robusto a cambios de arquitectura.
::::


::::cell{#ej7-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Cargar ResNet-18 preentrenada ───────────────────────────────────────────
finetune_net = models.resnet18(weights='DEFAULT')
print(f"Capa fc original: {finetune_net.fc}")

# ─── Reemplazar la capa de salida por una nueva con 10 unidades ──────────────
# in_features=512 en ResNet-18, pero lo leemos dinámicamente para no hardcodear.
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 10)

# ─── Inicialización Xavier sobre la nueva capa ───────────────────────────────
# Los pesos del resto de la red ya tienen valores efectivos — no los tocamos.
nn.init.xavier_uniform_(finetune_net.fc.weight)
nn.init.zeros_(finetune_net.fc.bias)

finetune_net = finetune_net.to(device)
print(f"Capa fc nueva:    {finetune_net.fc}")
```
::::


::::cell{#ej7-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Por qué no reutilizamos los pesos preentrenados de la capa `fc` original? ¿Qué representaban esos 1000 valores de salida en la red original, y por qué no tienen sentido para la tarea nueva?
::::


::::cell{#ej7-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La capa `fc` original de ResNet-18 es un `Linear(512, 1000)`: produce 1000 logits, uno por cada clase de ImageNet. Cada fila del peso es, en la práctica, un "prototipo" de esa clase en el espacio de features de la penúltima capa — por ejemplo, la fila 207 corresponde a *Golden Retriever*, la 999 a *toilet paper*.

Para clasificar uso del suelo en 10 categorías necesitamos solo 10 logits, y las "direcciones" en el espacio de features que separan estas clases (bosque vs. cultivo, residencial vs. industrial, río vs. mar) no son ninguna de las 1000 direcciones de ImageNet: son direcciones nuevas que la red tiene que aprender desde cero. Por eso reemplazamos la capa entera por un `Linear(512, 10)` inicializado al azar. En cambio, sí aprovechamos los pesos preentrenados del resto del modelo porque los feature maps que produce (bordes, texturas, patrones de alto nivel) son útiles para distinguir tipos de terreno igual que lo eran para distinguir razas de perros.
```
::::


::::cell{#setup-train-model type=code role=setup}
```python
# ─── Setup: función de entrenamiento reutilizable ───────────────────────────
# `train_model` corre el bucle estándar de train + eval por época y devuelve
# el historial de pérdida y exactitud en un diccionario; `plot_history` lo
# grafica. Los exponemos como funciones para no reescribir el mismo bucle
# dos veces: los usás en el Ej 8 (fine-tuning) y en el Ej 9 (ResNet-18 desde
# cero) y después comparás los historiales.

def train_model(model, train_loader, test_loader, optimizer, num_epochs):
    """
    Entrena un modelo de clasificación y retorna el historial de métricas.

    Parámetros:
    model (nn.Module): modelo ya movido al device correcto.
    train_loader, test_loader (DataLoader): dataloaders de train y test.
    optimizer (Optimizer): optimizador (puede tener varios grupos de parámetros).
    num_epochs (int): número de épocas.

    Retorna:
    hist (dict): claves 'train_loss', 'train_acc', 'test_loss', 'test_acc'.
    """
    criterion = nn.CrossEntropyLoss()
    hist = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        # ─── Entrenamiento ───────────────────────────────────────────────────
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)
        hist['train_loss'].append(loss_sum / total)
        hist['train_acc'].append(correct / total)

        # ─── Evaluación ──────────────────────────────────────────────────────
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                loss_sum += loss.item() * y.size(0)
                correct  += (logits.argmax(1) == y).sum().item()
                total    += y.size(0)
        hist['test_loss'].append(loss_sum / total)
        hist['test_acc'].append(correct / total)

        print(f"Época {epoch+1:2d}  "
              f"train_loss={hist['train_loss'][-1]:.4f}  train_acc={hist['train_acc'][-1]:.4f}  "
              f"test_loss={hist['test_loss'][-1]:.4f}  test_acc={hist['test_acc'][-1]:.4f}")
    return hist


def plot_history(hist, titulo=""):
    """Grafica las curvas de pérdida y exactitud de un historial."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epocas = range(1, len(hist['train_loss']) + 1)

    axes[0].plot(epocas, hist['train_loss'], label='Train')
    axes[0].plot(epocas, hist['test_loss'],  label='Test')
    axes[0].set_title(f"Pérdida — {titulo}")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].grid(True); axes[0].legend()

    axes[1].plot(epocas, hist['train_acc'], label='Train')
    axes[1].plot(epocas, hist['test_acc'],  label='Test')
    axes[1].set_title(f"Exactitud — {titulo}")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy")
    axes[1].grid(True); axes[1].legend()
    plt.tight_layout(); plt.show()
```
::::


::::cell{#ej8-enunciado type=markdown role=enunciado}
### Ejercicio 8 — Fine-tuning con learning rates diferenciales

**Objetivo:** Entrenar `finetune_net` aplicando un **learning rate bajo al backbone preentrenado** y un **learning rate 10× más alto a la capa `fc` nueva**.

**Enunciado:**

1. Definí `lr_base = 5e-3`.
2. Construí dos grupos de parámetros:
   - `params_backbone`: todos los parámetros de `finetune_net` **excepto** los de `fc`.
   - `params_fc`: los parámetros de `finetune_net.fc`.
3. Creá un optimizador `torch.optim.SGD` con una lista de dos dicts:
   - `{'params': params_backbone}` (usa `lr_base`).
   - `{'params': params_fc, 'lr': lr_base * 10}`.
   Pasale `lr=lr_base` y `weight_decay=1e-3` como hiperparámetros globales.
4. Entrená durante **5 épocas** con `train_model` y graficá la historia con `plot_history`.
5. Guardá el historial en `hist_finetune` para compararlo con el del próximo ejercicio.

> **Pistas:**
> - Usá `model.named_parameters()` para separar los parámetros por nombre: los de `fc` se llaman `fc.weight` y `fc.bias`.
> - En PyTorch, cada dict del argumento de parámetros puede sobreescribir `lr`, `weight_decay`, etc. Los valores del constructor del optimizador son **los defaults** si el dict no los define.
::::


::::cell{#ej8-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Paso 1: learning rate base ──────────────────────────────────────────────
lr_base = 5e-3

# ─── Paso 2: separar los parámetros en dos grupos ────────────────────────────
# Todo lo que no es 'fc' se considera backbone preentrenado.
params_backbone = [p for nombre, p in finetune_net.named_parameters()
                   if not nombre.startswith('fc.')]
params_fc = list(finetune_net.fc.parameters())

# ─── Paso 3: optimizador con dos grupos y lr distintos ───────────────────────
# El grupo fc usa lr * 10 porque arranca desde cero; al backbone solo lo afinamos.
optimizer_ft = torch.optim.SGD(
    [{'params': params_backbone},
     {'params': params_fc, 'lr': lr_base * 10}],
    lr=lr_base, weight_decay=1e-3)

# ─── Paso 4: entrenamiento ───────────────────────────────────────────────────
hist_finetune = train_model(finetune_net, train_loader, test_loader,
                            optimizer_ft, num_epochs=5)
plot_history(hist_finetune, titulo="Fine-tuning (ResNet-18 preentrenada)")
```
::::


::::cell{#ej8-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Por qué usamos un learning rate **10 veces mayor** para la capa `fc` y uno **chico** para el backbone? Pensalo en términos de *qué está tratando de aprender* cada grupo de parámetros.
::::


::::cell{#ej8-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- Los pesos **del backbone** ya representan algo útil: los feature maps que produce ResNet-18 sobre una imagen satelital todavía son informativos porque fueron aprendidos sobre millones de imágenes naturales — los bordes, texturas y patrones repetitivos de bajo nivel son universales. Lo único que queremos es **ajustarlos** al nuevo dominio — cambios pequeños. Un lr alto los movería bruscamente y destruiría ese conocimiento (*catastrophic forgetting*).
- La capa **`fc`** nueva, en cambio, arranca de valores aleatorios. No sabe nada. Necesita aprender desde cero a combinar los features en una decisión entre 10 categorías, así que le conviene un lr grande para avanzar rápido.

El factor 10 es heurístico. Lo importante es la **asimetría**: el backbone se ajusta suavemente, la cabeza aprende rápido. Precisamente porque EuroSAT está lejos de ImageNet (vista cenital, sin objetos centrados), el backbone tiene bastante trabajo de adaptación por hacer — por eso en este laboratorio elegimos `lr_base = 5e-3` y no un valor más chico como el de datasets muy similares a ImageNet (como hot-dog vs. not-hot-dog, donde `5e-4` alcanza).
```
::::


::::cell{#ej9-enunciado type=markdown role=enunciado}
### Ejercicio 9 — Comparación: entrenar ResNet-18 desde cero

**Objetivo:** Entrenar la misma arquitectura ResNet-18 pero **sin pesos preentrenados** y comparar con el resultado del Ejercicio 8.

**Enunciado:**

1. Instanciá `scratch_net = models.resnet18(weights=None)` (los pesos arrancan al azar).
2. Reemplazá `scratch_net.fc` por `nn.Linear(in_features, 10)` — igual que antes.
3. Movela al `device`.
4. Como ahora **todos** los parámetros tienen que aprender desde cero, usá un **único grupo de parámetros** con `lr=5e-2` (el mismo que le diste a la capa `fc` nueva en el Ej 8 — sin backbone preentrenado que proteger, todo el modelo merece el lr "de cabeza nueva") y `weight_decay=1e-3`. Usá SGD.
5. Entrená 5 épocas con `train_model` y graficá la historia. Guardala en `hist_scratch`.
6. Imprimí una comparación simple de la **exactitud final de test** entre los dos modelos.

> **Pista:** `weights=None` crea la misma arquitectura pero con pesos aleatorios según las inicializaciones por defecto de torchvision.
::::


::::cell{#ej9-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Paso 1-3: ResNet-18 sin pesos preentrenados ─────────────────────────────
scratch_net = models.resnet18(weights=None)
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 10)
scratch_net = scratch_net.to(device)

# ─── Paso 4: optimizador único con lr mayor ──────────────────────────────────
# Todos los parámetros arrancan al azar, así que usamos el mismo lr (alto)
# para la red entera — equivalente al que usaba la capa fc del Ej 8.
optimizer_sc = torch.optim.SGD(scratch_net.parameters(),
                               lr=5e-2, weight_decay=1e-3)

# ─── Paso 5: entrenamiento ───────────────────────────────────────────────────
hist_scratch = train_model(scratch_net, train_loader, test_loader,
                           optimizer_sc, num_epochs=5)
plot_history(hist_scratch, titulo="Desde cero (ResNet-18 sin preentrenar)")

# ─── Paso 6: comparación de exactitud final ──────────────────────────────────
print("\nComparación final (exactitud de test):")
print(f"  Fine-tuning (preentrenada): {hist_finetune['test_acc'][-1]:.4f}")
print(f"  Desde cero:                 {hist_scratch['test_acc'][-1]:.4f}")
```
::::


::::cell{#ej9-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Observando las dos historias:

- ¿Cuál modelo llega a mayor exactitud en 5 épocas? ¿Por qué?
- Mirando la curva de pérdida, ¿en qué época `finetune_net` supera la mejor exactitud que logra `scratch_net` tras 5 épocas?
- ¿Qué pasaría si tuviéramos 1 millón de imágenes satelitales en vez de 3000? ¿Seguiría siendo ventajoso el fine-tuning?
::::


::::cell{#ej9-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- **Fine-tuning gana por amplio margen** en 5 épocas. La red preentrenada parte de un extractor de features que ya funciona, así que en la primera época ya está cerca del óptimo; la red desde cero tiene que aprender desde bordes y texturas, lo cual demanda muchos más datos y épocas.
- Normalmente `finetune_net` supera en la **primera o segunda época** la mejor exactitud que la red desde cero logra tras 5. Esa es la ventaja en datos: gratis obtenemos el equivalente a haber visto millones de imágenes.
- Con 1 millón de imágenes satelitales propias, la ventaja del fine-tuning se reduce y eventualmente desaparece: con datos suficientes, una red inicializada al azar puede igualar o superar a la preentrenada, porque el dataset alcanza para aprender buenas features desde cero — y encima específicas del dominio satelital (donde las features de ImageNet no son óptimas). La regla general es: **cuanto más chico el dataset objetivo, más importa el preentrenamiento**.
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
- [ ] Las curvas de entrenamiento del fine-tuning y from-scratch se ven y son razonables.
- [ ] No modifiqué ninguna celda fuera de las de actividad.
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Completaste el Laboratorio 3. Cargaste una red preentrenada sobre ImageNet, inspeccionaste sus mapas de activación, viste qué patrones buscan sus filtros, y adaptaste el modelo a un dataset nuevo con fine-tuning y learning rates diferenciales. Estos dos usos — **inspección** y **adaptación** — son dos de las formas más comunes de reutilizar una red preentrenada en visión por computadora.
::::
