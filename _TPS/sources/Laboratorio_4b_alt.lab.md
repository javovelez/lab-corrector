---
lab: "4b_alt"
title: "Laboratorio n° 4. Parte B (alt): Segmentación con U-Net + encoder ResNet"
subject: "Redes Neuronales Profundas"
block: "4 — Detección y segmentación"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 4b (alt) — U-Net con encoder ResNet34
     Fuente única. Compilar con:
         python tools/lab_build.py _TPS/sources/Laboratorio_4b_alt.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_4b_alt.ipynb
             _TPS/Soluciones/Laboratorio_4b_alt_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 4. Parte B (alt): Segmentación con U-Net + encoder ResNet

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 4 — Detección y segmentación

---

## Introducción

En el laboratorio anterior trabajamos con **detección de objetos**: el modelo predice una caja (bounding box) por cada objeto de la imagen. En este vamos un paso más allá: **segmentación semántica**, donde el modelo asigna una clase a **cada píxel** de la imagen. En lugar de "hay un perro en esta caja" la predicción pasa a ser "estos píxeles son perro, estos son fondo".

La diferencia es importante: una caja siempre incluye píxeles que no pertenecen al objeto (las esquinas del rectángulo, áreas vacías). La segmentación da el contorno exacto, lo que permite aplicaciones imposibles con detección: edición de fotos, conducción autónoma (saber dónde termina la calle), análisis médico (medir el área de una lesión), realidad aumentada, etc.

### El modelo que vamos a construir

Vamos a implementar una **U-Net moderna con encoder pre-entrenado**. La idea es la misma que la del paper original de U-Net (Ronneberger et al. 2015):

- Un **encoder** que reduce la resolución y aumenta los canales — captura *qué* hay en la imagen.
- Un **decoder** que recupera la resolución original — produce el mapa de clases píxel por píxel.
- **Conexiones de salto** (skip connections) entre los dos caminos para no perder información espacial fina al bajar y al volver a subir.

La forma de "U" del diagrama (de ahí el nombre) refleja exactamente eso: bajar, dar la vuelta abajo, y subir mientras se "consultan" las activaciones del lado descendente.

![](https://miro.medium.com/max/720/1*YaLdptIoloK184uJQTH1HA.png)

La diferencia clave con el paper original es **de dónde viene el encoder**. En el paper, el encoder era una pila de convoluciones inicializadas al azar y entrenadas desde cero junto con el decoder. Hoy ya nadie hace eso: lo estándar es **usar como encoder una red de clasificación pre-entrenada en ImageNet** (ResNet, EfficientNet, ViT, etc.) y construir el decoder a medida. Esa receta:

- Aprovecha los millones de imágenes con las que ya se entrenó el encoder.
- Convierte la segmentación en un problema de "afinar features que ya son útiles", no de "aprender features visuales desde cero".
- Achica drásticamente la cantidad de datos y de cómputo necesarios para tener un modelo razonable.

Vamos a usar **ResNet34** como encoder. Es un clásico de 2015 (He et al.), suficientemente chico para entrenar cómodo en Colab pero suficientemente grande para haber aprendido representaciones visuales generales. Lo importamos directamente desde `torchvision.models` con sus pesos de ImageNet, y lo enganchamos a un decoder que vamos a escribir nosotros.

### El dataset: Oxford-IIIT Pet

Vamos a usar **Oxford-IIIT Pet** (Parkhi et al. 2012), un dataset clásico de imágenes de mascotas. Cada imagen tiene asociada una **máscara de segmentación trimap** con tres valores: `1` para los píxeles de la mascota, `2` para el fondo y `3` para los píxeles de borde. En este lab tratamos los píxeles de borde como **fondo** (los anotadores los marcaron como "no estoy seguro" pero conceptualmente están afuera de la mascota), así que el problema queda como una segmentación **binaria**:

- Clase `0` = fondo (incluye los píxeles del borde).
- Clase `1` = mascota.

Pet es cómodo para un primer contacto con segmentación: tiene ~3.7k imágenes de trainval, las fotos están centradas en el sujeto, la ambigüedad de clase es baja y el dataset es lo suficientemente chico como para entrenar en una sesión de Colab.

### Lo que vas a hacer

El laboratorio se divide en siete bloques:

1. **Sección A — Dataset:** explorar Oxford-IIIT Pet, entender el formato de los trimaps y armar los `DataLoader` de entrenamiento y validación.
2. **Sección B — El encoder:** ver cómo está armada la ResNet34 que vamos a usar y empaquetarla en un módulo que devuelve los feature maps de los cinco niveles que el decoder va a consumir.
3. **Sección C — Bloques del decoder:** implementar las dos piezas que necesita el decoder — la doble convolución estándar y el bloque que sube un nivel (upsample + concat + doble conv).
4. **Sección D — Red completa:** ensamblar encoder + decoder en una sola clase `UNetResNet`.
5. **Sección E — Entrenamiento desde cero:** entrenar la red con el encoder **inicializado al azar** (`pretrained=False`). El resultado va a ser modesto a propósito — sirve como línea de base.
6. **Sección F — Predicción:** correr la red entrenada sobre imágenes nuevas y visualizar las máscaras predichas.
7. **Sección G — Fine-tuning con pesos de ImageNet:** repetir el entrenamiento con la **misma arquitectura** pero con el encoder cargado con pesos de ImageNet. Comparar cuantitativa y visualmente. Esa es la moraleja del lab: con la misma red y el mismo presupuesto de cómputo, el pre-entrenamiento del encoder marca una diferencia enorme.

> **Importante — GPU y tiempo:** este laboratorio entrena dos veces la misma red sobre imágenes de 256×256. **Activá la GPU en Colab** antes de empezar: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4)*. Cada entrenamiento toma **3-5 minutos** sobre T4. Total del lab: ~10 minutos de cómputo. Sin GPU es impracticable.

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para el material teórico (convolución, U-Net, transfer learning) consultá los notebooks de la clase.
- Las celdas de test al final de cada bloque te ayudan a verificar que tu implementación devuelve tensores con la forma correcta. **No las modifiques**: si fallan, el problema está en tu código.
::::


::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::


::::cell{#imports type=code role=setup}
```python
# ─── Setup: imports y detección de GPU ──────────────────────────────────────
# Imports que usa el laboratorio de punta a punta. torch y torchvision ya
# vienen en Colab; no instalamos nada nuevo.
import os
import gc
import random
import tarfile
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights

device = (
    "cuda" if torch.cuda.is_available()        else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Dispositivo:        {device}")
if device == "cpu":
    print("ADVERTENCIA: sin GPU el entrenamiento va a ser inviable. "
          "Activá la GPU en Colab (T4) antes de continuar.")
```
::::


::::cell{#setup-pet type=code role=setup}
```python
# ─── Setup: descarga del dataset Oxford-IIIT Pet ────────────────────────────
# Oxford-IIIT Pet (Parkhi et al. 2012) tiene ~7400 imágenes de mascotas (37
# razas de perros y gatos) con segmentación trimap: 1=mascota, 2=fondo, 3=borde.
# Bajamos los dos .tar.gz oficiales (~800 MB combinados, 1-2 minutos en Colab).
DATA_URL_IMG = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
DATA_URL_ANN = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
DATA_DIR    = "/content/data/pet"
pet_dir     = DATA_DIR
images_dir  = os.path.join(pet_dir, "images")
trimaps_dir = os.path.join(pet_dir, "annotations", "trimaps")

os.makedirs(DATA_DIR, exist_ok=True)

def _download_and_extract(url, tag):
    tar_path = os.path.join(DATA_DIR, f"{tag}.tar.gz")
    print(f"Descargando {tag} desde {url}...")
    urllib.request.urlretrieve(url, tar_path)
    print(f"Extrayendo {tag}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(DATA_DIR)
    os.remove(tar_path)

if not os.path.isdir(images_dir):
    _download_and_extract(DATA_URL_IMG, "images")
if not os.path.isdir(trimaps_dir):
    _download_and_extract(DATA_URL_ANN, "annotations")

n_imgs = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
n_msks = len([f for f in os.listdir(trimaps_dir) if f.endswith(".png")])
print(f"\npet_dir   = {pet_dir}")
print(f"Imágenes  = {n_imgs}")
print(f"Trimaps   = {n_msks}")
```
::::


::::cell{#setup-pet-meta type=code role=setup}
```python
# ─── Setup: clases, colormap y mapeo del trimap ─────────────────────────────
# El trimap original de Pet tiene tres valores: 1=mascota, 2=fondo, 3=borde.
# Para este lab tratamos los píxeles de borde como FONDO — los anotadores los
# marcaron como "no estoy seguro", pero conceptualmente están afuera del
# animal. El problema queda binario: clase 0 = fondo (incluye borde) y
# clase 1 = mascota.
#
# Esa elección simplifica todo el pipeline: una sola convención de etiquetas
# de punta a punta, sin píxeles "ignorados" ni casos especiales en la pérdida
# o en la métrica.
PET_CLASSES  = ['background', 'pet']
PET_COLORMAP = [[0, 0, 0], [255, 100, 0]]   # negro / naranja
NUM_CLASSES  = len(PET_CLASSES)             # 2

print(f"Clases ({NUM_CLASSES}): {PET_CLASSES}")


def trimap_to_index(trimap):
    """
    Mapea un trimap de Pet con valores {1, 2, 3} a índices binarios {0, 1}:
    - 1 (mascota) → 1
    - 2 (fondo)   → 0
    - 3 (borde)   → 0  (lo tratamos como fondo)
    Acepta tensores con forma (H, W) o (1, H, W).
    """
    return (trimap == 1).long()


# ─── Helper para visualización: leyenda con los colores de las clases ──────
# Pega una fila de cuadritos coloreados arriba de la figura asociando cada
# color de PET_COLORMAP a su nombre de clase. Lo usamos en las visualizaciones
# de las secciones F y G.
from matplotlib.patches import Patch

def add_seg_legend(fig, class_names=PET_CLASSES, colormap=PET_COLORMAP):
    """Agrega una leyenda global a fig con los colores de las clases."""
    handles = [
        Patch(facecolor=tuple(c / 255 for c in colormap[i]),
              edgecolor='black', label=class_names[i])
        for i in range(len(class_names))
    ]
    fig.legend(handles=handles, loc='upper center',
               ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
```
::::


::::cell{#setup-dataset type=code role=setup}
```python
# ─── Setup: PetSegDataset con random crop + flip horizontal ─────────────────
# Esta clase está preescrita: armarla a mano excede el alcance del lab. Lo
# importante es que entiendas qué hace. Pasos clave en __getitem__:
#   1. Lee la imagen y el trimap.
#   2. Si la imagen es más chica que crop_size en alguna dimensión, le hace
#      un resize manteniendo aspect ratio (raro en Pet, pero hay un puñado
#      de imágenes muy chicas).
#   3. Random crop CONSISTENTE entre imagen y máscara — si recortara cada una
#      por su lado, los píxeles dejarían de coincidir.
#   4. Horizontal flip aleatorio (50% de probabilidad), también consistente.
#      Es la única augmentation que sumamos: multiplica efectivamente el
#      dataset sin costo.
#   5. Normaliza la imagen con la media/std de ImageNet, que es justamente lo
#      que espera el encoder ResNet34 que vamos a usar después.
#   6. Mapea el trimap original {1, 2, 3} a índices binarios {0, 1} usando
#      trimap_to_index (los píxeles de borde quedan como fondo).
#
# El parámetro `subset_size` permite tomar solo una porción del dataset.
# Lo usamos para que el train no demore más de unos minutos.
class PetSegDataset(torch.utils.data.Dataset):
    """Oxford-IIIT Pet para segmentación binaria (pet / background)."""

    def __init__(self, split, crop_size, pet_dir, subset_size=None, augment=True):
        self.crop_size = crop_size
        self.pet_dir   = pet_dir
        self.augment   = augment
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Lista de nombres del split (formato: "<name> <class_id> ...").
        txt = os.path.join(pet_dir, "annotations", f"{split}.txt")
        with open(txt) as f:
            names = [line.split()[0] for line in f
                     if line.strip() and not line.startswith("#")]
        self.names = names
        if subset_size is not None and subset_size < len(self.names):
            rng = random.Random(42)  # subset reproducible.
            self.names = rng.sample(self.names, subset_size)
        print(f"{split}: total={len(names)} usables={len(self.names)}"
              f"{' (subset)' if subset_size else ''}")

    def _read(self, name):
        img = Image.open(os.path.join(self.pet_dir, "images", f"{name}.jpg")
                         ).convert("RGB")
        msk = Image.open(os.path.join(self.pet_dir, "annotations",
                                       "trimaps", f"{name}.png"))
        # Si la imagen es más chica que el crop, resize al mínimo + un margen
        # para que después el random_crop pueda recortar sin error.
        if img.size[0] < self.crop_size[1] or img.size[1] < self.crop_size[0]:
            f = max(self.crop_size[1] / img.size[0],
                    self.crop_size[0] / img.size[1]) * 1.1
            new_size = (int(img.size[0] * f), int(img.size[1] * f))
            img = img.resize(new_size, Image.BILINEAR)
            msk = msk.resize(new_size, Image.NEAREST)
        return img, msk

    def __getitem__(self, idx):
        img, msk = self._read(self.names[idx])

        # Tensores: imagen (3, H, W) uint8, máscara (H, W) int64.
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        msk_t = torch.from_numpy(np.array(msk, dtype=np.int64))

        # Random crop consistente entre imagen y máscara.
        i, j, h, w = transforms.RandomCrop.get_params(img_t, self.crop_size)
        img_t = transforms.functional.crop(img_t, i, j, h, w)
        msk_t = msk_t[i:i + h, j:j + w]

        # Horizontal flip (data augmentation simple, gratis y muy efectiva).
        if self.augment and random.random() < 0.5:
            img_t = transforms.functional.hflip(img_t)
            msk_t = torch.flip(msk_t, dims=[1])

        # Mapeo trimap → índices binarios (0 = fondo+borde, 1 = mascota).
        out = trimap_to_index(msk_t)

        # Normalización imagen (mean/std de ImageNet).
        img_norm = self.transform(img_t.float() / 255)
        return img_norm, out

    def __len__(self):
        return len(self.names)
```
::::


::::cell{#secA type=markdown role=section}
---
## Sección A: El dataset Oxford-IIIT Pet

Antes de meternos con la red, conviene pasar un rato mirando los datos. En segmentación las etiquetas son imágenes y eso obliga a pensar cosas que no aparecen en clasificación: **cómo se interpreta cada píxel del trimap** y **cómo se aumentan/recortan las imágenes manteniendo la coincidencia entre imagen y máscara**.
::::


::::cell{#secA-intro type=markdown role=scaffolding}
### Por qué no usamos `Resize` en segmentación

En clasificación uno hace `transforms.Resize((224, 224))` y listo: la imagen se reescala a un tamaño fijo y la red la procesa. En **segmentación no se puede hacer eso** sin pensarlo dos veces. Si reescalo la imagen, ¿qué hago con la máscara?

- Si la reescalo con interpolación bilineal (la default), los valores de los píxeles dejan de ser índices de clase enteros y pasan a ser promedios ponderados — no tiene sentido decir "este píxel es 17.3" cuando los índices son discretos.
- Si la reescalo con interpolación *nearest neighbor*, mantengo los índices pero pierdo precisión en los bordes de los objetos: el contorno se vuelve dentado y poco fiel.
- Si reescalo a un tamaño distinto al original, además tengo que ser cuidadoso de mantener la **misma transformación** entre imagen y máscara, o pierden la correspondencia píxel a píxel.

La solución estándar — la que usa nuestro `PetSegDataset` — es **recortar (crop)** un parche de tamaño fijo en lugar de reescalar. El recorte aleatorio durante el entrenamiento cumple además el rol de data augmentation: cada época ve un parche distinto de cada imagen.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 1 — Visualización del dataset
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Inspeccionar imágenes y máscaras

**Objetivo:** Cargar imágenes y trimaps directamente desde el disco (sin pasar por el `Dataset`) para entender qué forma tienen y qué información codifica cada cosa.

**Enunciado:**

1. Definí una función `read_pet_images` que reciba el directorio del dataset, una cantidad `n` y el nombre del split (`'trainval'` o `'test'`), y devuelva dos listas paralelas con las primeras `n` imágenes y sus respectivos trimaps como tensores. Los listados de nombres de archivo de cada split están en `annotations/trainval.txt` y `annotations/test.txt`. Cada línea de esos archivos tiene la forma `<nombre> <class_id> <species> <breed_id>` — solo nos interesa el primer campo. Las imágenes están en `images/` (extensión `.jpg`) y los trimaps en `annotations/trimaps/` (extensión `.png`).
2. Llamala con `n=4` sobre el split de trainval.
3. Visualizá las 4 imágenes y sus 4 máscaras binarias en una grilla de 2×4 — fila superior con las imágenes, fila inferior con las máscaras. Para mostrar las máscaras como las va a ver el modelo, convertí los trimaps `{1, 2, 3}` a índices binarios `{0, 1}` con `trimap_to_index` (definida en el setup) antes de graficar. Acordate de que matplotlib espera tensores con orden `(H, W, C)` mientras que los tensores de PyTorch vienen como `(C, H, W)`: hay que reordenar los ejes antes de mostrar la imagen.

> **Pista 1:** En `torchvision.io` hay una función para leer imágenes que devuelve directamente un tensor `uint8`. Para los trimaps **no** hace falta forzar RGB: vienen como paleta indexada de un solo canal con valores `{1, 2, 3}`, así que la lectura por default ya entrega lo que querés.
>
> **Pista 2:** La máscara binaria que produce `trimap_to_index` solo tiene dos valores (`0` y `1`). Si la mostrás directamente con un colormap continuo, el contraste es claro pero los colores pueden no coincidir con los que después usamos en el resto del lab. Una opción simple es pasarle a `imshow` el argumento `cmap='gray'` para que `0` salga negro y `1` blanco.
::::


::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Función para leer imágenes + trimaps de un split ──────────────────────
def read_pet_images(pet_dir, n, split='trainval'):
    """
    Lee las primeras n imágenes y trimaps del split indicado.

    Parámetros:
    pet_dir (str): ruta a la carpeta del dataset Pet.
    n (int): cantidad de imágenes a leer.
    split (str): 'trainval' o 'test'.

    Retorna:
    features (list[Tensor]): imágenes RGB uint8 (3, H, W).
    labels   (list[Tensor]): trimaps uint8 (1, H, W) con valores {1, 2, 3}.
    """
    txt = os.path.join(pet_dir, 'annotations', f'{split}.txt')
    with open(txt) as f:
        # Cada línea: <nombre> <class_id> <species> <breed_id>. Tomamos el primero.
        names = [line.split()[0] for line in f
                 if line.strip() and not line.startswith('#')]
    features, labels = [], []
    for name in names[:n]:
        features.append(torchvision.io.read_image(
            os.path.join(pet_dir, 'images', f'{name}.jpg')))
        # Los trimaps son grayscale (1 canal); la lectura default ya los
        # entrega como (1, H, W) con uint8 valores {1, 2, 3}.
        labels.append(torchvision.io.read_image(
            os.path.join(pet_dir, 'annotations', 'trimaps', f'{name}.png')))
    return features, labels


# ─── Visualización ──────────────────────────────────────────────────────────
n = 4
imgs, masks = read_pet_images(pet_dir, n, split='trainval')

fig, axs = plt.subplots(2, n, figsize=(4 * n, 8))
for i in range(n):
    axs[0, i].imshow(imgs[i].permute(1, 2, 0))
    axs[0, i].set_title(f'Imagen {i}')
    axs[0, i].axis('off')
    # masks[i] tiene shape (1, H, W) con valores {1, 2, 3}. La convertimos
    # a índices binarios (0 = fondo+borde, 1 = mascota) para mostrar lo que
    # el modelo va a recibir como ground truth.
    binary = trimap_to_index(masks[i][0])
    axs[1, i].imshow(binary, cmap='gray')
    axs[1, i].set_title(f'Máscara binaria {i}')
    axs[1, i].axis('off')
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirando las máscaras: ¿por qué en la mayoría de las imágenes la clase **fondo** ocupa más píxeles que la mascota, aún cuando la cámara está apuntando claramente al animal?
::::


::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Aunque las fotos de Pet están centradas en la mascota y rara vez la mascota es chica en relación con el cuadro, igual el **fondo** suele cubrir más píxeles. Un perro o gato típicos tienen una silueta que ocupa un 25-35% del área de la imagen; el otro 65-75% es piso, pasto, sofá, mesa, pared, lo que sea. Si sumás los píxeles a lo largo de todo el dataset, el fondo se lleva alrededor del 70-75% y la mascota el 25-30%. Hay desbalance, suave pero claro — y vamos a tener que tenerlo en cuenta cuando entrenemos la red en la Sección E.
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 2 — Construir DataLoaders
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Instanciar `PetSegDataset` y armar los `DataLoader`

**Objetivo:** Usar la clase `PetSegDataset` (preescrita) para crear los datasets de train y val, envolverlos en `DataLoader` e inspeccionar la forma de un batch.

**Enunciado:**

1. Definí el tamaño de crop como una tupla `(256, 256)`. Este tamaño no es arbitrario: la red que vamos a construir baja la resolución 5 veces dividiendo por 2, así que el input necesita ser **divisible por `2^5 = 32`** para que las cuentas cierren. 256 es la potencia de 2 más cómoda en este rango (la justificación detallada aparece en la Sección D).
2. Creá dos datasets:
   - **train:** instanciá `PetSegDataset` sobre el split `'trainval'` con un `subset_size` de 1500 imágenes y data augmentation activada. El subset es por una razón puramente práctica: Pet tiene ~3700 imágenes en trainval y entrenar con todas tomaría más de media hora; con 1500 alcanzamos `val_acc` razonable en pocos minutos.
   - **val:** instanciá `PetSegDataset` sobre el split `'test'` con todo el dataset (sin subset) y la augmentation desactivada — en validación no queremos que cada época vea una versión distinta de la misma imagen.
3. Envolvé cada dataset en un `DataLoader` con tamaño de batch 8, descartando el último batch si queda incompleto y con un par de workers para paralelizar la lectura del disco. Acordate de que el train se baraja entre épocas y val no.
4. Pedile al iterador de train su primer batch e imprimí la forma de las imágenes y las máscaras, junto con los valores únicos de las máscaras. Las imágenes deberían tener forma `(8, 3, 256, 256)` con valores normalizados (no en [0, 1]); las máscaras `(8, 256, 256)` con valores en `{0, 1}`.

> **Pista:** Descartar el último batch incompleto se logra con un argumento del `DataLoader` cuyo nombre habla por sí solo. En segmentación no es crítico, pero es la convención.
>
> **Nota:** Usar 2 workers paraleliza la carga del disco. En Colab, valores muy altos (≥4) a veces traen más overhead que beneficio.
::::


::::cell{#ej2-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Datasets y DataLoaders ─────────────────────────────────────────────────
# crop_size 256x256 está elegido para esta U-Net: el encoder ResNet34 baja
# la resolución 5 veces dividiendo por 2 (256→128→64→32→16→8), y todas las
# divisiones cierran en enteros porque 256 es divisible por 2^5 = 32.
crop_size = (256, 256)

# Train: 1500 imágenes (subset) con augmentation. Val: dataset entero, sin aug.
pet_train = PetSegDataset('trainval', crop_size=crop_size, pet_dir=pet_dir,
                          subset_size=1500, augment=True)
pet_val   = PetSegDataset('test',     crop_size=crop_size, pet_dir=pet_dir,
                          subset_size=None, augment=False)

batch_size = 8
train_iter = torch.utils.data.DataLoader(
    pet_train, batch_size=batch_size, shuffle=True,
    drop_last=True, num_workers=2)
val_iter = torch.utils.data.DataLoader(
    pet_val, batch_size=batch_size, shuffle=False,
    drop_last=True, num_workers=2)

# ─── Inspección del primer batch ────────────────────────────────────────────
X, Y = next(iter(train_iter))
print(f"X shape: {tuple(X.shape)}  dtype={X.dtype}")
print(f"Y shape: {tuple(Y.shape)}  dtype={Y.dtype}")
print(f"Valores únicos en Y (primer batch): "
      f"{torch.unique(Y).tolist()}")
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

El batch de imágenes tiene shape `(8, 3, 256, 256)` y el batch de máscaras tiene shape `(8, 256, 256)` — sin la dimensión de canales. ¿Por qué la máscara no tiene canales? ¿Qué representa cada valor del tensor `Y`?
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La máscara no tiene dimensión de canales porque **cada píxel guarda un único entero** que representa la clase a la que pertenece, no un vector de probabilidades ni un color. El tensor `Y[b, i, j]` contiene un `0` (fondo) o un `1` (mascota) — la clase semántica de ese píxel.

Esa convención —"target categórico como tensor de índices"— es la que espera `nn.CrossEntropyLoss` cuando se usa para clasificación multiclase, también a nivel de píxel: `inputs` tiene forma `(B, C, H, W)` con los logits por clase, y `targets` tiene forma `(B, H, W)` con el índice de la clase correcta. La pérdida calcula internamente el softmax sobre el eje de canales y compara con el índice del target. Por eso no necesitamos una codificación one-hot ni canales explícitos en la máscara: PyTorch los maneja por nosotros.
```
::::


::::cell{#secB type=markdown role=section}
---
## Sección B: El encoder ResNet34

La pieza más cara y más importante de la red es el **encoder**: la parte que extrae features visuales útiles de la imagen. En lugar de implementarlo desde cero, vamos a usar como encoder una red de clasificación pre-entrenada en ImageNet — esa es la receta estándar de la segmentación moderna.
::::


::::cell{#secB-resnet type=markdown role=scaffolding}
### Qué es un *backbone* y por qué ResNet34

En la jerga de visión por computadora, un **backbone** (o "espinazo") es una red de clasificación que se usa como **extractor de features** para tareas distintas a la clasificación: detección, segmentación, pose estimation, etc. La idea es siempre la misma:

1. Tomás una arquitectura de clasificación bien estudiada (ResNet, EfficientNet, ViT...) ya entrenada sobre un dataset grande (típicamente ImageNet, 1.2M imágenes, 1000 clases).
2. Le sacás la cabeza de clasificación (la capa lineal que produce los logits de las 1000 clases).
3. Lo que queda — convoluciones que producen feature maps a varias resoluciones — es tu encoder.
4. Le ponés una cabeza nueva específica para tu tarea (en segmentación, un decoder que reconstruye una máscara).

Eso te ahorra reentrenar las primeras capas, que son las más costosas y las más universales: detectores de bordes, texturas, partes de objetos. Esos features sirven para *cualquier* tarea de visión, no solo para clasificación.

#### Por qué ResNet34 específicamente

- Es **chica y rápida** (~21M de parámetros): cabe cómoda en una T4 con batch 8 sobre 256×256 sin OOM.
- Es la ResNet "canónica" — figura en cientos de papers, librerías y tutoriales. Si tu próximo problema usa una ResNet, vas a reconocer la estructura.
- Tiene **5 stages** que dejan feature maps a 5 resoluciones distintas (`/2`, `/4`, `/8`, `/16`, `/32`), perfecto para alimentar las skip connections de un decoder tipo U-Net.

#### Estructura interna (sin entrar en el detalle del BasicBlock)

ResNet34 está organizada en una capa de entrada (*stem*) seguida de cuatro *layers*. Cada layer es una pila de **BasicBlocks** (bloques residuales con dos convoluciones 3×3 y un atajo que suma la entrada a la salida). Los detalles del BasicBlock no nos hacen falta para construir el decoder — para nuestros fines podemos tratar cada layer como una caja negra que recibe un feature map y devuelve otro con más canales y la mitad de resolución espacial.

| Stage | Operaciones | Salida (con input 256×256) | Canales |
|---|---|---|---|
| stem | conv 7×7 stride 2 + bn + relu | 128×128 | 64 |
| (maxpool) | maxpool 3×3 stride 2 | 64×64 | 64 |
| layer1 | 3 BasicBlocks | 64×64 | 64 |
| layer2 | 4 BasicBlocks (primer block stride 2) | 32×32 | 128 |
| layer3 | 6 BasicBlocks (primer block stride 2) | 16×16 | 256 |
| layer4 | 3 BasicBlocks (primer block stride 2) | 8×8 | 512 |

Detalles a notar:

- **El stem baja la resolución de 256 a 64 en dos pasos** (el stride-2 del conv 7×7 más el maxpool). Eso explica por qué ResNet llega "muy abajo" tan rápido — y por qué es tan eficiente computacionalmente: casi todo el trabajo pasa en mapas chicos.
- **De los cuatro layers, solo el primero no baja resolución**: layer1 trabaja a 64×64; layer2/3/4 cada uno divide por 2 con un stride-2 en su primer block.
- **Vamos a engancharnos a cinco puntos** para alimentar las skip connections del decoder: la salida del stem (`s0`, antes del maxpool), y la salida de cada uno de los cuatro layers (`s1`, `s2`, `s3`, `s4`). El último (`s4`) es el "fondo" de la U.

#### Lo que viene a continuación

Te damos pre-armada una clase `ResNetEncoder` que envuelve `torchvision.models.resnet34` y devuelve los cinco feature maps que el decoder va a consumir. Empaquetar manualmente las stages no aporta nada didáctico (es despachar atributos), pero **vale la pena leer la implementación con detenimiento** para tener clara la correspondencia entre lo que devuelve el encoder y los `s0..s4` que aparecen en la tabla de arriba.
::::


::::cell{#setup-encoder type=code role=setup}
```python
# ─── Encoder ResNet34 (provisto) ────────────────────────────────────────────
# Wrapper que toma `torchvision.models.resnet34` y la expone como un módulo
# que produce cinco feature maps — uno por cada nivel de resolución que el
# decoder va a usar como skip connection.
#
# Decisión de diseño: en lugar de definir cada conv y BatchNorm a mano, le
# pedimos a torchvision el modelo armado y nos quedamos con sus submódulos.
# Eso garantiza que los pesos de ImageNet (cuando los carguemos en la
# Sección G) se mapean a la estructura correcta sin trabajo adicional.
class ResNetEncoder(nn.Module):
    """
    Envuelve un ResNet34 de torchvision y expone los feature maps de los
    cinco niveles que el decoder consume.

    Parámetros:
    pretrained (bool): si True, carga pesos pre-entrenados en ImageNet;
                      si False, inicializa todos los pesos al azar (default
                      de PyTorch para Conv2d y BatchNorm2d).

    Forward(x), x: (B, 3, H, W) con H y W múltiplos de 32, devuelve la tupla:
    (s0, s1, s2, s3, s4)
    con shapes:
      s0: (B,  64, H/2,  W/2)   — después del stem, antes del maxpool
      s1: (B,  64, H/4,  W/4)   — salida de layer1
      s2: (B, 128, H/8,  W/8)   — salida de layer2
      s3: (B, 256, H/16, W/16)  — salida de layer3
      s4: (B, 512, H/32, W/32)  — salida de layer4 (fondo de la U)
    """
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        bb = resnet34(weights=weights)
        # Stem: conv 7×7 stride 2 + bn + relu. Nos quedamos con la salida
        # ANTES del maxpool — esa es nuestra primera skip (s0, /2).
        self.stem    = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.maxpool = bb.maxpool
        # Las cuatro layers de ResNet34, sin tocar.
        self.layer1  = bb.layer1   # /4   (no baja resolución)
        self.layer2  = bb.layer2   # /8   (primer block stride 2)
        self.layer3  = bb.layer3   # /16  (primer block stride 2)
        self.layer4  = bb.layer4   # /32  (primer block stride 2)

    def forward(self, x):
        s0 = self.stem(x)           # (B,  64, H/2,  W/2)   — skip 0
        x  = self.maxpool(s0)       # (B,  64, H/4,  W/4)
        s1 = self.layer1(x)         # (B,  64, H/4,  W/4)   — skip 1
        s2 = self.layer2(s1)        # (B, 128, H/8,  W/8)   — skip 2
        s3 = self.layer3(s2)        # (B, 256, H/16, W/16)  — skip 3
        s4 = self.layer4(s3)        # (B, 512, H/32, W/32)  — fondo
        return s0, s1, s2, s3, s4
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 3 — Sanity check del encoder
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — "Encender" el encoder y mirar las shapes

**Objetivo:** Verificar de primera mano que el encoder produce los cinco feature maps con las shapes que la tabla de la Sección B nos prometió. No estás aprendiendo a escribir código nuevo en este ejercicio — estás creándote un mapa mental concreto del encoder que vas a usar en las próximas secciones.

**Enunciado:**

1. Instanciá `ResNetEncoder(pretrained=False)`. Pasale `pretrained=False` por ahora — la inicialización al azar es más rápida (no descarga pesos) y para un sanity check de shapes da lo mismo.
2. Generá un tensor de entrada al azar con la forma de un batch real: `(1, 3, 256, 256)`.
3. Pasalo por el encoder dentro de un `torch.no_grad()` (no necesitamos gradientes para esto).
4. Imprimí, para cada uno de los cinco feature maps, su shape. Compará con la tabla de la sección anterior.

> **Pista:** El forward del encoder devuelve una tupla, así que podés desempacarla con `s0, s1, s2, s3, s4 = encoder(x)`. Para imprimir la shape de un tensor cómoda, `tuple(t.shape)` te lo da como tupla de enteros legible.
::::


::::cell{#ej3-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Sanity check: shapes del encoder ──────────────────────────────────────
# Con un input (1, 3, 256, 256), las cinco salidas tienen que coincidir con
# la tabla de la Sección B.
encoder = ResNetEncoder(pretrained=False)
x = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    s0, s1, s2, s3, s4 = encoder(x)

print(f"s0 shape: {tuple(s0.shape)}   (esperado: (1,  64, 128, 128))")
print(f"s1 shape: {tuple(s1.shape)}   (esperado: (1,  64,  64,  64))")
print(f"s2 shape: {tuple(s2.shape)}   (esperado: (1, 128,  32,  32))")
print(f"s3 shape: {tuple(s3.shape)}   (esperado: (1, 256,  16,  16))")
print(f"s4 shape: {tuple(s4.shape)}   (esperado: (1, 512,   8,   8))")
del encoder, x, s0, s1, s2, s3, s4
```
::::


::::cell{#secC type=markdown role=section}
---
## Sección C: Bloques del decoder

Con el encoder listo, ahora hay que armar el decoder: la parte que sube la resolución desde el fondo de la U (8×8, 512 canales) hasta el tamaño original (256×256, `num_classes` canales). Vamos a construirlo a partir de dos piezas chiquitas:

1. **`DoubleConv`** — el bloque de doble convolución que se repite en cada nivel del decoder.
2. **`DecoderBlock`** — el bloque que sube un nivel: upsample × 2 + concat con la skip + `DoubleConv`.

Ensamblar el decoder completo va a ser entonces apilar cinco `DecoderBlock`s y cerrar con una conv 1×1 a `num_classes`.
::::


::::cell{#secC-intro type=markdown role=scaffolding}
### Por qué upsample bilineal en lugar de `ConvTranspose2d`

En la U-Net del paper original, la subida de resolución se hace con una **convolución transpuesta** (a veces llamada "deconvolución"): una capa con parámetros aprendibles que produce un parche 2×2 a partir de cada píxel de entrada. Acá vamos a usar otra opción más simple: **interpolación bilineal**, que **no tiene parámetros**. Cada píxel de la salida se calcula como un promedio ponderado de los cuatro píxeles más cercanos del input.

Las dos alternativas funcionan razonablemente bien y la diferencia práctica suele ser chica. Bilinear tiene tres ventajas operativas:

- **Es más liviana** (no tiene parámetros, así que no agrega cómputo en el backward).
- **No introduce los "checkerboard artifacts"** típicos de la convolución transpuesta cuando el stride no divide al kernel size.
- **Es más estable de entrenar** desde cero, sobre todo combinada con BatchNorm.

A continuación de cada upsample bilineal viene un `DoubleConv`, así que la red sigue teniendo capacidad de aprender los detalles finos del decoder a través de esas convoluciones.

### Por qué se concatenan las skips (en vez de sumarlas)

El skip connection del lado descendente se va a integrar al lado ascendente con una **concatenación** sobre el eje de canales: si el decoder trae 256 canales y la skip viene con 256, se concatenan para tener 512 canales que después un `DoubleConv` mezcla y baja a la cantidad de canales del nivel siguiente.

¿Por qué concatenar y no sumar (como hace ResNet en sus residuos)?

- **Concatenar preserva la información** de los dos lados separadamente y deja que el `DoubleConv` siguiente aprenda cómo mezclarlos. Las features del encoder son "qué hay" (semántica), las del decoder son "dónde está" (localización) — combinarlas con una conv aprendible da más flexibilidad que forzarlas a sumarse.
- **Sumar** implícitamente asume que las dos features son comparables y aditivas. En ResNet eso es válido porque las dos ramas pasan por convoluciones similares dentro del mismo bloque; en una U-Net las features del lado ascendente y descendente son cualitativamente distintas (vienen de profundidades muy diferentes).

Esa es la diferencia operativa entre los dos tipos de skip que vas a ver en la red: el skip "interno" del BasicBlock de ResNet (suma) y el skip "externo" entre encoder y decoder (concatenación).
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 4 — DoubleConv
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — `DoubleConv`: bloque de doble convolución

**Objetivo:** Implementar el bloque que aparece dentro de cada `DecoderBlock`: dos convoluciones 3×3 con BatchNorm y ReLU intercaladas.

**Enunciado:**

Implementá una clase `DoubleConv` (subclase de `nn.Module`) cuyo constructor reciba la cantidad de canales de entrada y la cantidad de canales de salida. El bloque tiene que aplicar, en orden:

1. Una primera convolución 3×3 **con padding=1** que mapee de los canales de entrada a los de salida, seguida de una `BatchNorm2d` sobre los canales de salida y una `ReLU`.
2. Una segunda convolución 3×3 también con padding=1 que mantenga la cantidad de canales (entrada y salida iguales a los canales de salida del paso 1), seguida de otra `BatchNorm2d` y otra `ReLU`.

Para una entrada de forma `(B, c_in, H, W)`, la salida tiene que ser `(B, c_out, H, W)` — el padding=1 hace que cada conv 3×3 preserve exactamente la resolución espacial, y `BatchNorm` + `ReLU` no cambian la forma.

> **Pista 1:** Podés guardar la pila de capas en un `nn.Sequential` dentro de `__init__` y que `forward` sea casi trivial.
>
> **Pista 2:** Es buena costumbre poner `bias=False` en cada `Conv2d` cuando inmediatamente después viene una `BatchNorm`: la BN absorbe el efecto del bias, así que el parámetro de la conv es redundante. No es estrictamente necesario para que la red entrene, pero ahorra parámetros y es la convención estándar.
::::


::::cell{#ej4-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class DoubleConv(nn.Module):
    """
    (Conv 3x3 + BN + ReLU) dos veces seguidas.
    Es la pieza de cómputo que aparece dentro de cada DecoderBlock.

    Padding=1 en cada conv 3x3: la resolución espacial se preserva.
    bias=False en las convs porque la BN siguiente absorbe el bias.

    Entrada: (B, in_ch, H, W)
    Salida:  (B, out_ch, H, W)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
```
::::


::::cell{#test-ej4 type=code role=test}
```python
# ─── Test DoubleConv ───────────────────────────────────────────────────────
# Input chico (1x3x32x32) para que el test apenas use memoria.
# Con padding=1 las convs 3x3 preservan la resolución espacial: 32x32 → 32x32.
block = DoubleConv(3, 64)
inp = torch.rand(1, 3, 32, 32)
out = block(inp)
assert out.shape == (1, 64, 32, 32), f"Forma incorrecta: {tuple(out.shape)}"
print("Test DoubleConv OK.")
del block, inp, out
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 5 — DecoderBlock
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej5-enunciado type=markdown role=enunciado}
### Ejercicio 5 — `DecoderBlock`: subir un nivel

**Objetivo:** Implementar el bloque que sube un nivel del decoder. Cada bloque hace tres cosas en orden:

1. **Upsample** del input a 2× su resolución espacial usando interpolación bilineal.
2. (Si hay skip) **concatena** la skip al input upsampleado, sobre el eje de canales.
3. **DoubleConv** que mezcla el resultado y produce la cantidad de canales de salida.

El detalle clave es que **el último nivel del decoder no tiene skip**: el encoder solo nos da features hasta `/2` de resolución, así que el último upsample (de `/2` a `/1`) sale "pelado". Por eso el bloque tiene que aceptar opcionalmente que la skip sea `None`.

**Enunciado:**

Implementá una clase `DecoderBlock` (subclase de `nn.Module`) cuyo constructor reciba **tres** parámetros:

- `in_channels`: canales del tensor que viene "desde abajo" (el output del bloque previo del decoder, o el feature `s4` si es el primer bloque).
- `skip_channels`: canales de la skip connection que se va a concatenar (`0` si no hay skip).
- `out_channels`: canales de salida del bloque.

En `__init__` instanciá un único submódulo: una `DoubleConv` que reciba `in_channels + skip_channels` canales y produzca `out_channels`. (No hace falta nada más: el upsample bilineal se hace con `F.interpolate` en el forward, no necesita parámetros.)

En `forward(self, x, skip=None)`:

1. Hacé upsample bilineal de `x` con `scale_factor=2`. Pasale `align_corners=False` (el default recomendado para feature maps).
2. Si `skip is not None`, concatená `x` y `skip` sobre el eje de canales (`dim=1`).
3. Pasalo por la `DoubleConv`.

Para una entrada de forma `(B, in_channels, H, W)`:
- Si la `skip` tiene forma `(B, skip_channels, 2H, 2W)`, la salida es `(B, out_channels, 2H, 2W)`.
- Si `skip is None`, la salida también es `(B, out_channels, 2H, 2W)`.

> **Pista — `F.interpolate`:** la firma es `F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)`. Si tu PyTorch tira un warning sobre `recompute_scale_factor`, lo podés ignorar — es informativo, no afecta el resultado.
::::


::::cell{#ej5-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class DecoderBlock(nn.Module):
    """
    Un nivel del decoder de la U-Net: upsample x2 + (opcional) concat de
    skip + DoubleConv.

    Parámetros:
    in_channels   : canales del feature map que viene desde abajo.
    skip_channels : canales de la skip connection (0 si no hay skip).
    out_channels  : canales de salida del bloque.

    El upsample es bilineal (no tiene parámetros). La concatenación se hace
    sobre el eje de canales (dim=1).
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # La DoubleConv recibe lo que sale del upsample + lo que aporta la skip.
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        # Upsample bilineal x2: duplica H y W. align_corners=False es la
        # convención recomendada para feature maps (vs imágenes naturales).
        x = F.interpolate(x, scale_factor=2,
                          mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)
```
::::


::::cell{#test-ej5 type=code role=test}
```python
# ─── Test DecoderBlock ─────────────────────────────────────────────────────
# Probamos los dos casos: con skip y sin skip.

# Caso 1: con skip. Input chico, skip al doble de resolución.
block = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
x    = torch.rand(1, 128, 16, 16)
skip = torch.rand(1,  64, 32, 32)
out  = block(x, skip)
assert out.shape == (1, 64, 32, 32), f"Forma incorrecta (con skip): {tuple(out.shape)}"

# Caso 2: sin skip. Input cualquiera, salida al doble de resolución.
block_noskip = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)
x = torch.rand(1, 32, 128, 128)
out = block_noskip(x, None)
assert out.shape == (1, 16, 256, 256), f"Forma incorrecta (sin skip): {tuple(out.shape)}"

print("Test DecoderBlock OK.")
del block, block_noskip, x, skip, out
```
::::


::::cell{#secD type=markdown role=section}
---
## Sección D: Ensamblar la `UNetResNet`

Con el encoder y los dos bloques del decoder listos, armar la red completa es seguir el diagrama. La red completa es: encoder → cinco `DecoderBlock`s → conv 1×1 a `num_classes`.
::::


::::cell{#secD-shapes type=markdown role=scaffolding}
### Recorrido de formas con `crop_size = (256, 256)` y `num_classes = 2`

Antes de implementar conviene ver cómo varían las formas a lo largo de la red. Con un input de `(B, 3, 256, 256)`:

| Paso | Operación | Forma de salida | Notas |
|---|---|---|---|
| 0 | input | `(3, 256, 256)` | imagen normalizada |
| 1 | encoder.stem | `(64, 128, 128)` | **s0** |
| 2 | encoder.maxpool | `(64, 64, 64)` | (no es skip) |
| 3 | encoder.layer1 | `(64, 64, 64)` | **s1** |
| 4 | encoder.layer2 | `(128, 32, 32)` | **s2** |
| 5 | encoder.layer3 | `(256, 16, 16)` | **s3** |
| 6 | encoder.layer4 | `(512, 8, 8)` | **s4** (fondo de la U) |
| 7 | `decoder5(s4, skip=s3)` | `(256, 16, 16)` | upsample → concat 256+256 → DoubleConv→256 |
| 8 | `decoder4(_, skip=s2)` | `(128, 32, 32)` | upsample → concat 128+128 → DoubleConv→128 |
| 9 | `decoder3(_, skip=s1)` | `(64, 64, 64)` | upsample → concat 64+64 → DoubleConv→64 |
| 10 | `decoder2(_, skip=s0)` | `(32, 128, 128)` | upsample → concat 32+64 → DoubleConv→32 |
| 11 | `decoder1(_, skip=None)` | `(16, 256, 256)` | upsample → DoubleConv→16 (sin skip) |
| 12 | `head` (conv 1×1) | `(2, 256, 256)` | output: logits por clase, por píxel |

**El output tiene exactamente la misma resolución que el input** — esa es la propiedad clave de cualquier red de segmentación. La cabeza final (`nn.Conv2d` de `kernel_size=1`) es una pequeña capa lineal por píxel que mapea los 16 canales del último decoder block a los `NUM_CLASSES` logits.

> **Por qué el último decoder block no tiene skip:** el encoder ResNet34 solo produce features hasta resolución `/2` (la salida del stem, `s0`). El último upsample del decoder lleva de `/2` a `/1` (resolución completa), pero a `/1` ya no hay nada más que la imagen de entrada con la que concatenar. Tampoco se gana mucho concatenándola: la imagen cruda no es un feature útil. Así que el último bloque queda sin skip.

> **Por qué los canales del decoder son `256, 128, 64, 32, 16` y no algo simétrico al encoder:** los canales del encoder los hereda fijos de ResNet34. Los del decoder los elegimos nosotros, y la receta `[256, 128, 64, 32, 16]` es un default sensato (el mismo que usan librerías como `segmentation_models_pytorch`): empezar con la mitad de los canales del fondo de la U y dividir por 2 en cada paso. Otros valores también funcionan; este balance entre capacidad y costo está bien probado.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 6 — UNetResNet
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej6-enunciado type=markdown role=enunciado}
### Ejercicio 6 — Clase `UNetResNet`

**Objetivo:** Ensamblar encoder y decoder en la red completa.

**Enunciado:**

Implementá una clase `UNetResNet` (subclase de `nn.Module`) cuyo constructor reciba dos parámetros:

- `num_classes` (int): número de clases de salida (en nuestro caso, 2).
- `pretrained` (bool, default `False`): si el encoder se inicializa con pesos de ImageNet o al azar.

**En el `__init__`** declará cinco submódulos:

1. **El encoder.** Instanciá `ResNetEncoder(pretrained=pretrained)`.
2. **Cinco `DecoderBlock`s.** Mirá la tabla de la sección anterior para deducir los `(in_channels, skip_channels, out_channels)` de cada uno. Te ayudamos con el primero: el bloque más profundo recibe `s4` (512 canales) "desde abajo" y `s3` (256 canales) como skip, y produce 256 canales de salida → `DecoderBlock(512, 256, 256)`. Los siguientes salen del mismo razonamiento.
3. **La cabeza final.** Una `nn.Conv2d` con `kernel_size=1` que mapee de los 16 canales del último decoder block a `num_classes` canales. La conv 1×1 actúa como una capa lineal por píxel sobre la dimensión de canales.

**En el `forward(self, x)`** seguí el diagrama:

1. Llamá al encoder y desempacá las cinco skips (`s0, s1, s2, s3, s4`).
2. Aplicá los cinco `DecoderBlock`s en orden, pasándole a cada uno la skip que corresponde — el último va con `skip=None`.
3. Aplicá la cabeza final y devolvé el resultado.

> **Pista:** Es buena costumbre nombrar los decoder blocks de forma que el orden quede explícito al leer. Por ejemplo, `decoder5` (el más profundo, que usa `s4` y `s3`), `decoder4`, ..., hasta `decoder1` (el más superficial, sin skip). El forward queda casi auto-documentado.
::::


::::cell{#ej6-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class UNetResNet(nn.Module):
    """
    U-Net moderna con encoder ResNet34 (opcionalmente pre-entrenado en
    ImageNet) y decoder construido a mano con upsample bilineal + concat +
    DoubleConv.

    Parámetros:
    num_classes (int): número de clases de salida.
    pretrained (bool): si True, el encoder arranca con pesos de ImageNet.

    Entrada: (B, 3, H, W) con H y W múltiplos de 32.
    Salida:  (B, num_classes, H, W) — un logit por clase y por píxel.
    """
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        # ─── Encoder ────────────────────────────────────────────────────────
        self.encoder = ResNetEncoder(pretrained=pretrained)

        # ─── Decoder: 5 bloques que llevan de /32 a /1 ─────────────────────
        # Cada DecoderBlock recibe (in_ch, skip_ch, out_ch) según la tabla
        # de shapes de la Sección D. Notar la asimetría con el encoder: los
        # canales de salida del decoder los elegimos nosotros, los del
        # encoder los hereda fijos de ResNet34.
        self.decoder5 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.decoder4 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder3 = DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)
        self.decoder2 = DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32)
        self.decoder1 = DecoderBlock(in_channels=32,  skip_channels=0,   out_channels=16)

        # ─── Cabeza: conv 1x1 a num_classes ────────────────────────────────
        # Capa lineal por píxel: mapea 16 canales a num_classes logits.
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # ─── Encoder: cinco skips ──────────────────────────────────────────
        s0, s1, s2, s3, s4 = self.encoder(x)
        # s0: (B,  64, /2)   s1: (B,  64, /4)   s2: (B, 128, /8)
        # s3: (B, 256, /16)  s4: (B, 512, /32)

        # ─── Decoder: subir nivel por nivel concatenando la skip ───────────
        x = self.decoder5(s4, skip=s3)    # → (B, 256, /16)
        x = self.decoder4(x,  skip=s2)    # → (B, 128, /8)
        x = self.decoder3(x,  skip=s1)    # → (B,  64, /4)
        x = self.decoder2(x,  skip=s0)    # → (B,  32, /2)
        x = self.decoder1(x,  skip=None)  # → (B,  16, /1)   sin skip arriba

        # ─── Cabeza: logits por clase y por píxel ──────────────────────────
        return self.head(x)               # → (B, num_classes, /1)
```
::::


::::cell{#test-ej6 type=code role=test}
```python
# ─── Test UNetResNet ───────────────────────────────────────────────────────
# Test con num_classes=2 e input de tamaño realista. Verificamos que la forma
# de salida es la misma que la de entrada (en H y W) y que el eje de canales
# es num_classes — esa es la propiedad clave de la red.
unet_test = UNetResNet(num_classes=2, pretrained=False)
inp = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    out = unet_test(inp)
print(f"input  : {tuple(inp.shape)}")
print(f"output : {tuple(out.shape)}  (esperado: (1, 2, 256, 256))")
assert out.shape == (1, 2, 256, 256), f"Forma incorrecta: {tuple(out.shape)}"
print(f"Parámetros del modelo: "
      f"{sum(p.numel() for p in unet_test.parameters()):,}")
print("Test UNetResNet OK.")
del unet_test, inp, out
```
::::


::::cell{#secE type=markdown role=section}
---
## Sección E: Entrenamiento desde cero (encoder al azar)

Antes de pasar al fine-tuning con pesos de ImageNet, vamos a entrenar la `UNetResNet` con `pretrained=False` — es decir, con todos los pesos (encoder + decoder) inicializados al azar. Este experimento sirve como **línea de base**: nos dice qué tan lejos llega la red entrenándola desde cero con nuestro presupuesto de cómputo y nuestro dataset chico. Después, en la Sección G, vamos a repetir el mismo entrenamiento con `pretrained=True` y ver cuánto cambia.
::::


::::cell{#setup-cleanup type=code role=setup}
```python
# ─── Limpieza de memoria antes de entrenar ──────────────────────────────────
# Las celdas de test crearon objetos que pueden quedar referenciados. Forzamos
# garbage collection y vaciamos la cache de CUDA. Si igual ves errores de OOM
# durante el train, reiniciá el entorno de Colab y volvé a ejecutar todo.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"VRAM libre tras limpiar: "
          f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB de "
          f"{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB")
```
::::


::::cell{#secE-pesos type=markdown role=scaffolding}
### Pesos por clase para la cross-entropy

Como vimos en el Ejercicio 1, el fondo cubre alrededor del 70-75% de los píxeles. Si entrenamos con `nn.CrossEntropyLoss` "plana" (las dos clases con el mismo peso), la red descubre rápido que **predecir "fondo" en cada píxel** ya da una accuracy del ~70% — un mínimo local cómodo del que no se sale. Para evitar esto, ponderamos la pérdida con pesos por clase: el fondo cuenta menos, la mascota más, y el gradiente fuerza a la red a discriminar la silueta.

La receta:

1. Recorrer todo el split de train contando cuántos píxeles hay de cada clase.
2. Calcular la frecuencia relativa `freq[c] = pixels_clase_c / pixels_totales`.
3. Definir `weights[c] = 1 / sqrt(freq[c] + ε)` con un `ε` pequeño para evitar dividir por cero.
4. Normalizar los pesos para que sumen `num_classes`. Eso mantiene el orden de magnitud de la pérdida estable.

> **Por qué `1/√freq` y no `1/freq` directamente:** `1/freq` parece la opción "natural" pero produce pesos demasiado desbalanceados. Si el fondo es 10× más frecuente que la mascota, con `1/freq` los pesos quedan en ratio 10:1 — la red descubre rápido que conviene **nunca** predecir fondo (un error en un píxel minoritario cuesta diez veces más) y termina con accuracy de píxel peor que la baseline trivial. La raíz cuadrada comprime ese rango: el mismo ratio 10× de frecuencias da pesos ~3.16:1, agresivos pero no tóxicos. En nuestro caso (Pet, fondo ~0.72 / mascota ~0.28) el ratio max/min entre pesos pasa de ~2.5× con `1/freq` a ~1.6× con `1/√freq`. Es la receta que usan, por ejemplo, ENet (Paszke et al. 2016) y SegNet (Badrinarayanan et al. 2017) en sus papers de segmentación.

Lo usamos pasándole el vector de pesos a `nn.CrossEntropyLoss(weight=weights)`.
::::


::::cell{#setup-pesos type=code role=setup}
```python
# ─── Cálculo de pesos por clase (provisto) ─────────────────────────────────
# Recorremos pet_train acumulando cuántos píxeles tiene cada clase. El bucle
# tarda 1-2 minutos porque toca abrir cada imagen del dataset una vez.
print("Contando píxeles por clase (puede tardar 1-2 min)...")
freqs = Counter()
for _, mask in pet_train:
    unique, counts = torch.unique(mask, return_counts=True)
    for c, n in zip(unique.tolist(), counts.tolist()):
        freqs[c] += n
total_pixels = sum(freqs.values())
print(f"Píxeles totales: {total_pixels:,}")

# Tabla de frecuencias para verificar que el desbalance es real.
df = pd.DataFrame({
    "clase":    PET_CLASSES,
    "píxeles":  [freqs.get(i, 0) for i in range(NUM_CLASSES)],
    "freq":     [round(freqs.get(i, 0) / total_pixels, 4)
                 for i in range(NUM_CLASSES)],
})
print(df.to_string(index=False))

# Pesos: 1 / sqrt(freq) normalizados a sumar NUM_CLASSES.
class_freq = np.array([freqs.get(i, 0) / total_pixels
                       for i in range(NUM_CLASSES)])
raw_w   = 1.0 / np.sqrt(class_freq + 1e-6)
weights = raw_w * (NUM_CLASSES / raw_w.sum())
weights = torch.tensor(weights, dtype=torch.float32)
print(f"\nPesos por clase (normalizados a sumar NUM_CLASSES={NUM_CLASSES}):")
for i, n in enumerate(PET_CLASSES):
    print(f"  {n:14s}  freq={class_freq[i]:.4f}  weight={weights[i].item():.3f}")
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 7 — Entrenamiento desde cero
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej7-enunciado type=markdown role=enunciado}
### Ejercicio 7 — Entrenar la `UNetResNet` desde cero

**Objetivo:** Entrenar la red completa con el encoder inicializado al azar (`pretrained=False`), por unas pocas épocas, y guardar el `val_acc` por época para comparar después contra la versión pre-entrenada.

> ⚠️ **Atención al tiempo de entrenamiento:** este ejercicio entrena por **4 épocas** sobre las 1500 imágenes del subset. En **GPU T4 de Colab tarda aproximadamente 3-5 minutos**.

**Enunciado:**

El código está estructurado por bloques numerados; completá los huecos donde dice `# Tu código aquí`.

1. **Modelo, loss y optimizador:**
   - Instanciá `UNetResNet` con `num_classes=NUM_CLASSES` y `pretrained=False`. Mandalo al `device`.
   - Definí la pérdida como `nn.CrossEntropyLoss` ponderada con los `weights` que se calcularon arriba (no te olvides de mandar los pesos al `device` también).
   - Como optimizador usá Adam con learning rate `1e-3`.

2. **Loop de entrenamiento:** entrená por 4 épocas. En cada época:
   - **Modo train.** Para cada batch del iterador de train:
     - Mandá imagen y máscara al device. Convertí la máscara a `long` (la cross-entropy exige índices enteros como target).
     - Forward por la red. La salida tiene shape `(B, NC, 256, 256)` — exactamente la misma resolución espacial que la máscara.
     - Calculá la pérdida, backpropagá, hacé el step del optimizador y limpiá los gradientes.
     - Acumulá lo necesario para reportar pérdida promedio y accuracy de pixel al final del epoch.
   - **Modo eval.** Recorré el iterador de validación midiendo accuracy de pixel sobre val. No olvides envolver el bloque en un contexto `no_grad` para no acumular gradientes inútilmente.
   - Imprimí una línea por época con `epoch | train_loss | train_acc | val_acc`.
   - **Guardá el `val_acc` de cada época en una lista llamada `val_acc_history`** — la vamos a comparar con los resultados del Ej. 9 (fine-tuning).

> **Pista — accuracy de píxel:** comparar `prediccion == ground_truth` (después del `argmax` sobre canales) te da un tensor booleano. Sumá los `True` y dividí por la cantidad total de píxeles.
>
> **Nota — qué esperar:** con la receta del lab (subset de 1500 imágenes, 4 épocas, encoder al azar, pesos `1/√freq`, augmentation por flip horizontal) `val_acc` debería arrancar cerca del 0.72 baseline en la primera época y llegar a **~0.78-0.85** en la cuarta. Cualquier valor claramente por encima de 0.72 indica que la red discrimina mascota de fondo, aunque sea de forma rudimentaria. Los resultados verdaderamente buenos van a aparecer en el Ej. 9 con el encoder pre-entrenado.
::::


::::cell{#ej7-code type=code role=student-code}
```python
# ─── 1) Modelo, loss y optimizador ──────────────────────────────────────────

# Tu código aquí

# ─── 2) Loop de entrenamiento (4 épocas, guardando val_acc_history) ────────

# Tu código aquí
```

```python solution
# ─── 1) Modelo, loss y optimizador ──────────────────────────────────────────
# pretrained=False: encoder inicializado al azar (sin pesos de ImageNet).
# Esta es la línea de base "todo desde cero" que después comparamos con el
# fine-tuning.
model = UNetResNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Parámetros del modelo: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# ─── 2) Loop de entrenamiento ───────────────────────────────────────────────
# Guardamos val_acc por época para comparar después con el fine-tuning del Ej. 9.
num_epochs = 4
val_acc_history = []
for epoch in range(num_epochs):
    # ── Train ───────────────────────────────────────────────────────────────
    model.train()
    L_sum, n_correct, n_pixels = 0.0, 0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device).long()
        # Forward: la red produce un logit por clase por píxel.
        # y_hat: (B, NC, 256, 256), y: (B, 256, 256).
        y_hat = model(X)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métricas: accuracy de pixel sobre el batch.
        with torch.no_grad():
            preds = y_hat.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_pixels  += y.numel()
            L_sum     += loss.item() * X.size(0)

    train_loss = L_sum / len(pet_train)
    train_acc  = n_correct / n_pixels

    # ── Val ─────────────────────────────────────────────────────────────────
    model.eval()
    n_correct_v, n_pixels_v = 0, 0
    with torch.no_grad():
        for X, y in val_iter:
            X, y = X.to(device), y.to(device).long()
            y_hat = model(X)
            preds = y_hat.argmax(dim=1)
            n_correct_v += (preds == y).sum().item()
            n_pixels_v  += y.numel()
    val_acc = n_correct_v / n_pixels_v
    val_acc_history.append(val_acc)

    print(f"epoch {epoch + 1:2d}/{num_epochs}  "
          f"train_loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}")
```
::::


::::cell{#ej7-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Si en lugar de usar pesos por clase entrenaras la misma red con `nn.CrossEntropyLoss()` "plana" (sin `weight`), ¿qué esperarías que pase con la **accuracy de pixel global** y con la **accuracy de pixel sobre la mascota**? Justificá.
::::


::::cell{#ej7-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Sin pesos por clase, la red optimiza una pérdida en la que cada píxel cuenta igual. Como ~72% de los píxeles son fondo, **predecir "fondo" en todos los píxeles** ya da una accuracy global cercana a 0.72. Es un mínimo local muy cómodo: la pérdida es relativamente baja y el gradiente para mejorarlo (aprender a discriminar la mascota) es chico porque cualquier desvío hacia "mascota" le suma error de fondo. La red termina convergiendo a algo así:

- **Accuracy global:** ~0.72 — engañosamente alta, porque está tirada hacia arriba por el fondo.
- **Accuracy sobre la mascota:** muy mala (~0.0-0.2) — la red prácticamente no segmenta al animal. La métrica de pixel global oculta esto.

Con pesos por clase (`1/√freq`), el fondo aporta menos al gradiente y la mascota aporta más — la red se ve obligada a aprender a discriminar la silueta. Eso típicamente:

- **Baja un poco la accuracy global** (porque se ganan errores en píxeles de fondo que antes acertaba "por gravedad").
- **Sube fuertemente la accuracy sobre la mascota y la mIoU** (la métrica estándar en segmentación, que promedia el IoU por clase y por lo tanto es robusta al desbalance).

Conclusión: con desbalance, la accuracy global es una métrica engañosa. Conviene mirar **accuracy por clase** o **mIoU**, y usar pesos por clase (o losses como Focal, Dice o Tversky) durante el entrenamiento.
```
::::


::::cell{#secF type=markdown role=section}
---
## Sección F: Predicción y visualización

Ya tenemos un modelo entrenado desde cero. Vamos a usarlo para producir máscaras predichas sobre imágenes del split de test y compararlas con el ground truth.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 8 — Predicción
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej8-enunciado type=markdown role=enunciado}
### Ejercicio 8 — Visualizar predicciones

**Objetivo:** Tomar imágenes del split de test, predecir sus máscaras con la red entrenada y mostrar la triple "imagen original / predicción / ground truth" para inspeccionar visualmente qué tan bien funciona el modelo.

**Enunciado:**

1. Implementá una función `label2image` que reciba un tensor 2D `(H, W)` con índices de clase (`0` o `1`) y devuelva un tensor 3D `(H, W, 3)` con los colores RGB correspondientes según `PET_COLORMAP` (negro para fondo, naranja para mascota).
2. Tomá 4 imágenes del split de test con la función `read_pet_images` que implementaste en el Ej. 1.
3. Para cada imagen:
   - Recortá un parche 256×256 **desde el centro** de la imagen, tanto en la imagen como en el trimap correspondiente. Centrar el recorte (en lugar de tomarlo desde una esquina) maximiza la chance de que la mascota quede dentro del cuadro.
   - Normalizá la imagen con la misma media/std de ImageNet que usaba el dataset durante el entrenamiento.
   - Pasala por la red en modo eval, dentro de un bloque `no_grad`, y quedate con la clase de mayor logit por píxel (`argmax` sobre la dimensión de canales).
   - Convertí la predicción a imagen RGB con `label2image`. Hacé lo mismo con el ground truth (acordate de aplicar `trimap_to_index` para mapear los valores `{1, 2, 3}` del trimap original a los índices binarios `{0, 1}` que usa la red).
4. Armá la visualización **separada en dos figuras** de 2 filas cada una. Cada figura tiene tres columnas: imagen (256×256), predicción (256×256) y ground truth (256×256). Llamá a `add_seg_legend(fig)` (helper preescrito en el setup) en **cada** figura para que ambas tengan arriba una leyenda con el color de cada clase.

> **Pista:** En `torchvision.transforms.functional` hay una función `crop` que toma una imagen y devuelve un crop a partir de coordenadas `top`, `left`, `height` y `width`. Para centrar el crop usá `top = (H - 256) // 2` y `left = (W - 256) // 2`.
::::


::::cell{#ej8-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── label2image: índices de clase → imagen RGB ─────────────────────────────
def label2image(pred):
    """
    Mapea un tensor 2D (H, W) de índices de clase a una imagen RGB (H, W, 3)
    usando PET_COLORMAP.
    """
    colormap = torch.tensor(PET_COLORMAP, device=pred.device, dtype=torch.uint8)
    img = colormap[pred.long()]
    return img.cpu()


# ─── Predicción sobre 4 imágenes del split de test ─────────────────────────
n = 4
test_imgs, test_masks = read_pet_images(pet_dir, n, split='test')
model.eval()

# La transformación de imagen tiene que coincidir con la del dataset.
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])


def predict_and_render(idx, ax_row):
    """Procesa la imagen idx-ésima y la dibuja en la fila de axes dada."""
    img_t = test_imgs[idx]
    msk_t = test_masks[idx].squeeze(0)  # (1, H, W) → (H, W)
    H, W = img_t.shape[1], img_t.shape[2]
    # Si la imagen es más chica que 256 en alguna dimensión, hacemos un
    # resize previo (mismo trato que el dataset) para evitar errores.
    if H < 256 or W < 256:
        f = max(256 / H, 256 / W) * 1.1
        img_t = transforms.functional.resize(
            img_t, [int(H * f), int(W * f)], antialias=True)
        msk_t = transforms.functional.resize(
            msk_t.unsqueeze(0), [int(H * f), int(W * f)],
            interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
        H, W = img_t.shape[1], img_t.shape[2]

    # Crop CENTRADO 256x256: maximiza la chance de que la mascota quede
    # dentro del cuadro.
    top  = (H - 256) // 2
    left = (W - 256) // 2
    img_crop  = transforms.functional.crop(img_t, top, left, 256, 256)
    mask_crop = transforms.functional.crop(msk_t, top, left, 256, 256).long()

    # Normalizo la imagen como en el train y paso por la red.
    X = norm(img_crop.float() / 255).unsqueeze(0).to(device)
    y_hat = model(X)                          # (1, NC, 256, 256)
    pred  = y_hat.argmax(dim=1).squeeze(0)    # (256, 256)

    # Ground truth: trimap → índices binarios.
    gt_idx = trimap_to_index(mask_crop)       # (256, 256)

    ax_row[0].imshow(img_crop.permute(1, 2, 0))
    ax_row[0].set_title("Imagen (256x256)")
    ax_row[0].axis('off')
    ax_row[1].imshow(label2image(pred))
    ax_row[1].set_title("Predicción (256x256)")
    ax_row[1].axis('off')
    ax_row[2].imshow(label2image(gt_idx))
    ax_row[2].set_title("Ground truth (256x256)")
    ax_row[2].axis('off')


# ─── Dibujamos en dos figuras de 2 filas cada una ──────────────────────────
ROWS_PER_FIG = 2
with torch.no_grad():
    for fig_idx in range(0, n, ROWS_PER_FIG):
        fig, axs = plt.subplots(ROWS_PER_FIG, 3, figsize=(12, 4 * ROWS_PER_FIG))
        for r in range(ROWS_PER_FIG):
            predict_and_render(fig_idx + r, axs[r])
        add_seg_legend(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()
```
::::


::::cell{#ej8-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Comparando las predicciones con el ground truth: ¿en qué regiones de la imagen la red funciona mejor y en cuáles peor? Proponé al menos dos mejoras concretas del pipeline (datos, arquitectura o entrenamiento) que apunten a los puntos débiles que detectes.
::::


::::cell{#ej8-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Patrón típico observado tras 4 épocas de entrenamiento desde cero sobre Pet (subset 1500):

**Mejor:**
- **Cuerpo central de la mascota y fondos uniformes amplios** — la red identifica bien grandes "blobs" coherentes: el torso del perro o del gato, una pared, un piso. Son patrones de baja frecuencia espacial que el receptive field del decoder resuelve sin esfuerzo.
- **Mascotas que ocupan buena parte del cuadro** — un perro en plano medio con fondo limpio sale bastante prolijo.

**Peor:**
- **Bordes finos y siluetas detalladas** — el contorno entre mascota y fondo aparece dentado o algo desplazado. Las orejas, las patas y las colas a menudo se "comen" o quedan recortadas. La causa principal: la pérdida ponderada da igual peso a todos los píxeles de la mascota — los del centro pesan lo mismo que los del borde, así que no hay incentivo extra para afinar el contorno.
  - *Mejora posible:* agregar `Dice loss` o `boundary loss` que penalicen específicamente los errores en el contorno.
- **Animales chicos en el cuadro** (escenas con la mascota en una esquina o lejos) — la red las detecta peor.
  - *Mejora posible:* multi-scale training (forzar al modelo a ver la misma imagen a varios crops de distintos tamaños), o más épocas combinadas con augmentation más fuerte (color jitter, rotaciones leves).

**Mejora general:** lo más efectivo —y lo que vamos a probar en la próxima sección— es inicializar el encoder con pesos pre-entrenados de ImageNet en vez de al azar. La red sigue siendo la misma; solo cambia de dónde arrancan los parámetros del encoder. Vas a ver el salto cualitativo en el próximo ejercicio.
```
::::


::::cell{#secG type=markdown role=section}
---
## Sección G: Fine-tuning con encoder pre-entrenado en ImageNet

Hasta acá entrenamos la `UNetResNet` con el encoder **inicializado al azar**. La prueba inicial del Ej. 7 muestra que la red arranca y supera el baseline trivial de "todo fondo", pero los resultados son modestos. La razón es bien conocida: una red profunda con ~24M de parámetros no puede aprender features visuales útiles a partir de 1500 imágenes. Necesitaría órdenes de magnitud más datos, mucho más cómputo, augmentation pesada, o todo lo anterior.
::::


::::cell{#secG-intro type=markdown role=scaffolding}
### Lo que cambia (y lo que no) respecto al Ejercicio 7

La receta práctica de la última década resuelve esto con **transfer learning**: en lugar de inicializar los pesos del encoder al azar, se reemplazan por los pesos de una red pre-entrenada sobre ImageNet (1.2 millones de imágenes naturales etiquetadas con 1000 clases). El encoder ya viene "sabiendo" extraer bordes, texturas y partes de objetos — y solo hay que afinar esos features para la tarea específica de segmentación. Las primeras épocas del fine-tuning ya producen resultados que el train desde cero no alcanza ni con muchísimas más épocas.

En nuestro caso, el cambio en código es mínimo: nuestra `UNetResNet` ya acepta el flag `pretrained` en su constructor. Para hacer el fine-tuning solo cambiamos esa única línea — `pretrained=True`. Eso es lo bueno de haber armado la red de esta manera: el experimento que comparamos es **idéntico en arquitectura y en presupuesto de cómputo**, la única variable es la inicialización del encoder.

> **La normalización ya está bien:** `PetSegDataset` aplica `transforms.Normalize` con la media/std de ImageNet, que es exactamente lo que esperan los pesos pre-entrenados. **No hace falta tocar nada del dataset ni de los DataLoaders** — entran tal cual al modelo nuevo. Si hubiéramos normalizado con otras estadísticas, el transfer learning andaría peor porque las activaciones de las primeras capas verían distribuciones para las que el encoder no fue pre-entrenado.

> **Aclaración sobre el decoder:** los pesos del decoder se inicializan al azar incluso cuando `pretrained=True`. Eso es esperado: el decoder es nuevo y no existía en la red original de clasificación. Lo que se pre-entrena es el encoder, que es la pieza más cara de aprender.
::::


::::cell{#setup-cleanup-ft type=code role=setup}
```python
# ─── Limpieza antes del fine-tuning ─────────────────────────────────────────
# Vamos a tener dos modelos en GPU al mismo tiempo (el desde cero del Ej. 7
# y el fine-tuning del Ej. 9) porque después los comparamos en visualización.
# Forzamos GC y vaciamos la cache de CUDA antes de instanciar el segundo.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"VRAM libre antes del fine-tuning: "
          f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB de "
          f"{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB")
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 9 — Fine-tuning con pesos de ImageNet
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej9-enunciado type=markdown role=enunciado}
### Ejercicio 9 — Fine-tuning de la `UNetResNet` con pesos de ImageNet

**Objetivo:** Repetir el entrenamiento de la Sección E pero con `pretrained=True`, comparar cuantitativa y visualmente con la versión desde cero, y observar el salto.

**Enunciado:**

1. **Modelo, loss y optimizador:**
   - Instanciá `UNetResNet` con `num_classes=NUM_CLASSES` y `pretrained=True`. Mandalo al `device`. Guardalo en una variable distinta a la del Ej. 7 (por ejemplo `model_ft`) — vamos a comparar los dos modelos al final.
   - Reutilizá los pesos por clase (no hace falta recalcularlos: el dataset es el mismo). Loss: cross-entropy ponderada. Optimizer: Adam con `lr=1e-3`.

2. **Loop de entrenamiento por 5 épocas.** Mismo patrón que el Ej. 7. Imprimí una línea por época y guardá el `val_acc` de cada una en una lista llamada `val_acc_ft_history`.

3. **Comparación cuantitativa.** Mostrá una tabla con `epoch | val_acc desde cero | val_acc fine-tuning`. La columna desde cero termina en epoch 4 (después queda en blanco) y la fine-tuning va hasta epoch 5. Usá un `pd.DataFrame` para que salga prolijo.

4. **Comparación visual.** Tomá las mismas 4 imágenes del split de test que usaste en el Ej. 8 y armá una visualización en dos figuras de 2 filas cada una. Cada fila tiene cuatro columnas: imagen / predicción desde cero / predicción fine-tuning / ground truth. Reusá `label2image`, `add_seg_legend` y la lógica de centro-crop + normalización del Ej. 8.

> **Nota — qué esperar:** con 5 épocas de fine-tuning sobre Pet, `val_acc` debería superar **0.90 ya en la primera época** y converger en torno a **0.93-0.96** en la última. Esa es la magnitud del salto: un encoder ImageNet le ahorra a la red años de entrenamiento sobre features visuales generales y le permite enfocarse exclusivamente en la tarea de segmentación.
::::


::::cell{#ej9-code type=code role=student-code}
```python
# ─── 1) Modelo, loss y optimizador ──────────────────────────────────────────

# Tu código aquí

# ─── 2) Loop de entrenamiento (5 épocas, guardando val_acc_ft_history) ─────

# Tu código aquí

# ─── 3) Comparación cuantitativa: tabla val_acc desde cero vs fine-tuning ──

# Tu código aquí

# ─── 4) Comparación visual: imagen / desde cero / fine-tuning / GT ─────────

# Tu código aquí
```

```python solution
# ─── 1) Modelo, loss y optimizador ──────────────────────────────────────────
# Misma arquitectura que el Ej. 7 — la única diferencia es pretrained=True.
# Eso reemplaza la inicialización al azar del encoder por los pesos de
# ImageNet; el decoder sigue arrancando con pesos al azar (es código nuevo,
# no existía en la red original de clasificación).
model_ft = UNetResNet(num_classes=NUM_CLASSES, pretrained=True).to(device)
criterion_ft = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-3)

print(f"Parámetros del modelo fine-tuning: "
      f"{sum(p.numel() for p in model_ft.parameters()):,}")

# ─── 2) Loop de entrenamiento (5 épocas) ───────────────────────────────────
num_epochs_ft = 5
val_acc_ft_history = []
for epoch in range(num_epochs_ft):
    # ── Train ───────────────────────────────────────────────────────────────
    model_ft.train()
    L_sum, n_correct, n_pixels = 0.0, 0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device).long()
        y_hat = model_ft(X)
        loss = criterion_ft(y_hat, y)
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()

        with torch.no_grad():
            preds = y_hat.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_pixels  += y.numel()
            L_sum     += loss.item() * X.size(0)

    train_loss = L_sum / len(pet_train)
    train_acc  = n_correct / n_pixels

    # ── Val ─────────────────────────────────────────────────────────────────
    model_ft.eval()
    n_correct_v, n_pixels_v = 0, 0
    with torch.no_grad():
        for X, y in val_iter:
            X, y = X.to(device), y.to(device).long()
            y_hat = model_ft(X)
            preds = y_hat.argmax(dim=1)
            n_correct_v += (preds == y).sum().item()
            n_pixels_v  += y.numel()
    val_acc = n_correct_v / n_pixels_v
    val_acc_ft_history.append(val_acc)

    print(f"epoch {epoch + 1}/{num_epochs_ft}  "
          f"train_loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}")

# ─── 3) Comparación cuantitativa ────────────────────────────────────────────
# val_acc_history viene del Ej. 7 (4 valores). val_acc_ft_history acaba de
# generarse (5 valores). Mostramos las 5 épocas lado a lado, dejando "—"
# cuando el desde-cero ya terminó.
rows = []
for e in range(num_epochs_ft):
    if e < len(val_acc_history):
        a_zc = f"{val_acc_history[e]:.4f}"
    else:
        a_zc = "—"
    a_ft = f"{val_acc_ft_history[e]:.4f}"
    rows.append({"epoch": e + 1,
                 "val_acc desde cero":  a_zc,
                 "val_acc fine-tuning": a_ft})
print("\nComparación val_acc por epoch:")
print(pd.DataFrame(rows).to_string(index=False))

# ─── 4) Comparación visual: imagen / desde cero / fine-tuning / GT ─────────
# Reusamos test_imgs/test_masks que ya leímos en el Ej. 8. Si por alguna
# razón el alumno no los tiene cargados, los volvemos a leer acá.
n = 4
test_imgs, test_masks = read_pet_images(pet_dir, n, split='test')

model.eval()
model_ft.eval()


def predict_compare(idx, ax_row):
    """
    Para la imagen idx-ésima del split de test, dibuja en ax_row (4 axes):
    imagen, predicción desde cero, predicción fine-tuning, ground truth.
    """
    img_t = test_imgs[idx]
    msk_t = test_masks[idx].squeeze(0)
    H, W = img_t.shape[1], img_t.shape[2]
    if H < 256 or W < 256:
        f = max(256 / H, 256 / W) * 1.1
        img_t = transforms.functional.resize(
            img_t, [int(H * f), int(W * f)], antialias=True)
        msk_t = transforms.functional.resize(
            msk_t.unsqueeze(0), [int(H * f), int(W * f)],
            interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
        H, W = img_t.shape[1], img_t.shape[2]

    top  = (H - 256) // 2
    left = (W - 256) // 2
    img_crop  = transforms.functional.crop(img_t, top, left, 256, 256)
    mask_crop = transforms.functional.crop(msk_t, top, left, 256, 256).long()
    X = norm(img_crop.float() / 255).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_zc = model(X).argmax(dim=1).squeeze(0)        # desde cero
        pred_ft = model_ft(X).argmax(dim=1).squeeze(0)     # fine-tuning
    gt_idx = trimap_to_index(mask_crop)

    ax_row[0].imshow(img_crop.permute(1, 2, 0))
    ax_row[0].set_title("Imagen")
    ax_row[0].axis('off')
    ax_row[1].imshow(label2image(pred_zc))
    ax_row[1].set_title("Pred. desde cero")
    ax_row[1].axis('off')
    ax_row[2].imshow(label2image(pred_ft))
    ax_row[2].set_title("Pred. fine-tuning")
    ax_row[2].axis('off')
    ax_row[3].imshow(label2image(gt_idx))
    ax_row[3].set_title("Ground truth")
    ax_row[3].axis('off')


ROWS_PER_FIG = 2
with torch.no_grad():
    for fig_idx in range(0, n, ROWS_PER_FIG):
        fig, axs = plt.subplots(ROWS_PER_FIG, 4,
                                figsize=(16, 4 * ROWS_PER_FIG))
        for r in range(ROWS_PER_FIG):
            predict_compare(fig_idx + r, axs[r])
        add_seg_legend(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()
```
::::


::::cell{#ej9-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Por qué el fine-tuning con un encoder pre-entrenado en ImageNet (un dataset de **clasificación** de 1000 clases como "perro labrador", "gato siamés", "auto", "edificio") transfiere tan bien a una tarea aparentemente distinta como **segmentación binaria de mascotas**? Pensá específicamente en qué está aprendiendo el encoder durante el pre-entrenamiento y por qué esos features son útiles también para decidir, píxel a píxel, si pertenece a una mascota o al fondo.
::::


::::cell{#ej9-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La pregunta tiene una respuesta corta y otra más profunda. La corta: el encoder pre-entrenado **ya sabe extraer features visuales generales** — bordes, texturas, formas, partes de objetos — y esos features son útiles para *cualquier* tarea de visión, no solo para clasificación.

La más profunda mira la jerarquía de features que aprende una CNN profunda durante el pre-entrenamiento:

- **Capas tempranas** (las más cercanas a la entrada): aprenden filtros de bordes orientados, gradientes de color, texturas locales. Son features universales — sirven para clasificar gatos, segmentar tumores, detectar autos o reconocer caras.
- **Capas intermedias:** aprenden combinaciones de bordes y texturas que forman *partes* de objetos (un ojo, una pata, una rueda, una textura de pelo).
- **Capas profundas:** aprenden combinaciones de partes que forman *objetos completos* o *escenas*.

Cuando entrenamos desde cero sobre 1500 imágenes de Pet, la red tiene que descubrir esa jerarquía completa con muy pocas señales de entrenamiento. Termina aprendiendo features mediocres en todos los niveles y por eso los resultados son modestos.

Cuando hacemos fine-tuning, las capas tempranas e intermedias del encoder ya están "bien afinadas" porque ImageNet tiene 1.2 millones de imágenes con miles de objetos distintos — ya vio bordes, texturas y partes en una variedad inmensa de contextos. Solo nos queda **adaptar las capas más profundas y el decoder a la tarea específica de segmentación binaria**, y eso requiere muchísimo menos data y cómputo.

Hay un detalle adicional que vale la pena: la diferencia entre las tareas (clasificación de 1000 clases vs segmentación binaria) **no importa tanto como la similitud del dominio**. Las imágenes de ImageNet y las de Pet son ambas fotos naturales de objetos comunes, con luz natural, en escenas cotidianas. Esa similitud de dominio es lo que hace que la transferencia sea tan exitosa. Si el dominio fuera radicalmente distinto (por ejemplo imágenes médicas, satelitales o microscopía electrónica), la ganancia del pre-entrenamiento ImageNet sería menor — y por eso en esos dominios se usan encoders pre-entrenados específicos del campo (RadImageNet para imagen médica, SatMAE para satelital).
```
::::


::::cell{#checklist type=markdown role=checklist}
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] Los tests de `DoubleConv`, `DecoderBlock` y `UNetResNet` pasan sin errores.
- [ ] Los entrenamientos del Ej. 7 (desde cero, 4 épocas) y del Ej. 9 (fine-tuning, 5 épocas) corrieron sin OOM. Si tuve OOM, reinicié el entorno y volví a ejecutar.
- [ ] El entrenamiento del Ej. 7 supera claramente el ~72% de "predecir todo fondo" (esperable: ~0.78-0.85). El fine-tuning del Ej. 9 lo supera con holgura (esperable: >0.90).
- [ ] La grilla de visualización del Ej. 8 muestra al menos una imagen donde la silueta de la mascota es reconocible.
- [ ] La comparación visual del Ej. 9 muestra una mejora clara de la columna fine-tuning respecto a la columna desde cero.
- [ ] Respondí las preguntas de análisis (Ej. 1, 2, 7, 8, 9).
- [ ] No modifiqué ninguna celda fuera de las de actividad (ni las de test ni las de setup).
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Implementaste tu primera red de segmentación semántica de punta a punta. Practicaste:

- **Manejo del dataset Oxford-IIIT Pet** — formato trimap, mapeo a índices binarios, random crop consistente entre imagen y máscara, horizontal flip como augmentation.
- **Uso de un backbone pre-entrenado** — empaquetar `torchvision.models.resnet34` como un módulo que devuelve los feature maps de cinco niveles, listos para alimentar las skip connections de un decoder.
- **Decoder a medida** — `DoubleConv` + `DecoderBlock` (upsample bilineal + concat + doble conv), apilados en cinco niveles más una cabeza 1×1 a `num_classes`.
- **Entrenamiento balanceado** — pesos por clase calculados como `1/√freq` para evitar el colapso a "todo fondo".
- **Predicción y visualización** — `label2image` para mapear índices a colores, comparación lado a lado con el ground truth.
- **El experimento limpio** — entrenar la **misma arquitectura** dos veces, una con encoder al azar y otra con encoder pre-entrenado en ImageNet, y comparar. La única variable es la inicialización del encoder.

La moraleja del lab es la diferencia que viste en la tabla del Ej. 9: con la misma red, el mismo dataset y el mismo presupuesto de cómputo, el pre-entrenamiento del encoder mueve la `val_acc` de ~0.80 a ~0.94. Esa es la receta real de producción para casi cualquier tarea de visión moderna: **arquitectura standard + backbone pre-entrenado + fine-tuning sobre tu dataset**. No es magia, es transferencia de features visuales aprendidas sobre millones de imágenes.

Con esto cierra el bloque de **detección y segmentación** de la materia. Lo siguiente vamos a verlo en redes recurrentes y arquitecturas para datos secuenciales.
::::
