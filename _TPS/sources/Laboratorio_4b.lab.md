---
lab: "4b"
title: "Laboratorio n° 4. Parte B: Segmentación semántica con U-Net"
subject: "Redes Neuronales Profundas"
block: "4 — Detección y segmentación"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 4b — Segmentación semántica con U-Net
     Fuente única. Compilar con:
         python tools/lab_build.py _TPS/sources/Laboratorio_4b.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_4b.ipynb
             _TPS/Soluciones/Laboratorio_4b_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 4. Parte B: Segmentación semántica con U-Net

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 4 — Detección y segmentación

---

## Introducción

En el laboratorio anterior trabajamos con **detección de objetos**: el modelo predice una caja (bounding box) por cada objeto de la imagen. En este vamos un paso más allá: **segmentación semántica**, donde el modelo asigna una clase a **cada píxel** de la imagen. En lugar de "hay un perro en esta caja" la predicción pasa a ser "estos píxeles son perro, estos son fondo, estos son sofá".

La diferencia es importante: una caja siempre incluye píxeles que no pertenecen al objeto (las esquinas del rectángulo, áreas vacías). La segmentación da el contorno exacto, lo que permite aplicaciones imposibles con detección: edición de fotos, conducción autónoma (saber dónde termina la calle), análisis médico (medir el área de una lesión), realidad aumentada, etc.

### El modelo: U-Net

Vamos a implementar **U-Net** (Ronneberger et al. 2015), una arquitectura que se diseñó originalmente para segmentación de imágenes biomédicas y que se convirtió en el caballito de batalla de la segmentación moderna. La idea central es muy simple:

- Un **camino contractivo** (encoder) que reduce la resolución y aumenta los canales — captura *qué* hay en la imagen.
- Un **camino expansivo** (decoder) que recupera la resolución original — produce el mapa de clases píxel por píxel.
- **Conexiones de salto** (skip connections) entre los dos caminos para no perder información espacial fina al bajar y al volver a subir.

La forma de "U" del diagrama (de ahí el nombre) refleja exactamente eso: bajar, dar la vuelta abajo, y subir mientras se "consultan" las activaciones del lado descendente.

![](https://miro.medium.com/max/720/1*YaLdptIoloK184uJQTH1HA.png)

> **Variante que vamos a implementar:** el paper original (2015) usa convoluciones 3×3 **sin padding**, lo que hace que cada doble convolución muerda 4 píxeles de los bordes. Resultado: el output sale con menor resolución que el input (388×388 desde 572×572), las skip connections necesitan recortes explícitos para alinearse, y la red se entrena solo sobre el centro de cada imagen. Esa variante "fiel al paper" tiene valor histórico pero es difícil de optimizar desde cero. Acá implementamos la **variante moderna** que usa padding=1 en cada conv 3×3 — la salida tiene exactamente el mismo tamaño que la entrada, las skips concatenan directamente y la red entrena sobre todos los píxeles. Es la receta que usan hoy nnU-Net, MONAI y `segmentation_models_pytorch`. Agregamos también **BatchNorm** después de cada convolución (tampoco estaba en el paper de 2015 — la técnica se popularizó después). Los demás elementos sí son los del paper: encoder de 5 niveles, decoder simétrico, skip connections en cada nivel.

### El dataset: Oxford-IIIT Pet

Vamos a usar **Oxford-IIIT Pet** (Parkhi et al. 2012), un dataset clásico de imágenes de mascotas. Cada imagen tiene asociada una máscara de segmentación **trimap** con tres valores: `1` para los píxeles de la mascota, `2` para el fondo y `3` para los píxeles de borde (esos los vamos a tratar como "ignorar" durante el entrenamiento, igual que el borde blanco de VOC). Para nuestro lab queda como un problema de segmentación **binaria**: clase 0 = fondo, clase 1 = mascota.

> **Por qué Pet y no Pascal VOC:** VOC2012 (21 clases, ~1.5k imágenes de train) es un problema demasiado difícil para una U-Net entrenada **desde cero** sin pesos preentrenados ni augmentation pesada — el paper original de U-Net se entrenaba sobre datasets biomédicos con deformaciones elásticas como augmentation. Pet (2 clases efectivas, ~3.7k imágenes de trainval) es mucho más manejable: las imágenes están centradas en el sujeto, hay poca ambigüedad de clase y la red aprende a segmentar reconocido en ~15-20 minutos de Colab T4.

### Lo que vas a hacer

El laboratorio se divide en seis bloques:

1. **Sección A — Dataset:** explorar Oxford-IIIT Pet, entender el formato de los trimaps y armar los `DataLoader` de entrenamiento y validación.
2. **Sección B — Bloques de U-Net:** implementar las cinco piezas que componen la red — doble convolución, downsampling, upsampling, capa final y la función auxiliar de recorte para alinear skip connections.
3. **Sección C — Red completa:** ensamblar las piezas en la clase `UNet`.
4. **Sección D — Entrenamiento desde cero (prueba inicial):** entrenar la red sobre Pet por unas pocas epochs para confirmar que la arquitectura arranca y aprende algo. Usamos **pesos por clase** en la función de pérdida para no terminar con una red que predice mayormente "fondo". El resultado va a ser modesto a propósito — entrenar una U-Net desde cero sobre un dataset chico no es la receta de producción.
5. **Sección E — Predicción:** correr la red entrenada desde cero sobre imágenes nuevas y visualizar las máscaras predichas.
6. **Sección F — Fine-tuning desde una U-Net pre-entrenada:** repetir el entrenamiento partiendo de una U-Net cuyo encoder es un ResNet34 con pesos de ImageNet (vía `segmentation_models_pytorch`). En pocas epochs vamos a alcanzar resultados claramente mejores que entrenando desde cero. Esa es la receta real de producción y ésa es la moraleja del lab.

> **Importante — GPU y tiempo:** este laboratorio entrena dos redes profundas (la U-Net desde cero ~31M de parámetros + la U-Net con encoder ResNet34 ~24M) sobre imágenes de 256×256. **Activá la GPU en Colab** antes de empezar: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4)*. Cada entrenamiento toma **5-8 minutos** sobre T4. Total del lab: ~15 minutos de cómputo. Sin GPU es impracticable.
>
> Atención también con la memoria: las celdas de test al final de la Sección B crean modelos de juguete que pueden quedar referenciados después si no los liberás. Más adelante hay una celda de limpieza explícita justo antes del entrenamiento. Si igual ves errores de OOM (out of memory) durante el train, usá *Entorno de ejecución > Reiniciar* y ejecutá de nuevo todo de arriba a abajo.

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para el material teórico (convolución transpuesta, FCN, segmentación semántica) consultá el notebook `Segmentacion.ipynb` de la clase.
- Las celdas de test al final de cada bloque te ayudan a verificar que tu implementación devuelve tensores con la forma correcta. **No las modifiques**: si fallan, el problema está en tu código.
- Los entrenamientos del Ej. 9 (prueba inicial desde cero) y del Ej. 11 (fine-tuning) toman **5-8 minutos cada uno** sobre Colab T4. Podés ir avanzando con otra cosa mientras corren.
::::


::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::


::::cell{#imports type=code role=setup}
```python
# ─── Setup: imports y detección de GPU ──────────────────────────────────────
# Imports que usa el laboratorio de punta a punta. Acá no instalamos nada nuevo
# porque torch y torchvision ya vienen en Colab.
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
# ─── Setup: clases, colormap y mapeo de trimap ──────────────────────────────
# Trimap original de Pet: 1=mascota, 2=fondo, 3=borde. Lo remapeamos a:
#   0 = background, 1 = pet, 255 = ignore (borde).
# Por qué borde como ignore: los bordes del trimap son una franja "no estoy
# seguro" del anotador y suelen ser ruidosos. Si los incluyéramos como una
# tercera clase, la red gastaría capacidad aprendiendo ruido.
PET_CLASSES  = ['background', 'pet']
PET_COLORMAP = [[0, 0, 0], [255, 100, 0]]   # negro / naranja

NUM_CLASSES  = len(PET_CLASSES)   # 2
IGNORE_INDEX = 255                # los píxeles de borde no se evalúan

print(f"Clases ({NUM_CLASSES}): {PET_CLASSES}")


# ─── Helper para visualización: leyenda con colores de las clases ──────────
# Se usa en el Ej. 10. Pega una fila de cuadritos coloreados arriba de la
# figura asociando cada color de PET_COLORMAP a su nombre de clase, más un
# cuadrito gris para los píxeles `ignore`.
from matplotlib.patches import Patch

def add_seg_legend(fig, class_names=PET_CLASSES, colormap=PET_COLORMAP):
    """Agrega una leyenda global a fig con los colores de las clases."""
    handles = [
        Patch(facecolor=tuple(c / 255 for c in colormap[i]),
              edgecolor='black', label=class_names[i])
        for i in range(len(class_names))
    ]
    handles.append(Patch(facecolor=(0.5, 0.5, 0.5),
                         edgecolor='black', label='ignore (borde)'))
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
#   5. Normaliza la imagen con la media/std de ImageNet.
#   6. Mapea el trimap {1=pet, 2=bg, 3=border} a {1, 0, 255} para que la
#      cross-entropy del Ej. 9 ignore los píxeles de borde con ignore_index.
#
# El parámetro `subset_size` permite tomar solo una porción del dataset.
# Lo usamos para que el train no demore más de ~20 minutos.
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

        # Mapeo trimap {1=pet, 2=bg, 3=border} → {1=pet, 0=bg, 255=ignore}.
        out = torch.full_like(msk_t, IGNORE_INDEX)
        out[msk_t == 2] = 0
        out[msk_t == 1] = 1

        # Normalización imagen.
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
### Ejercicio 1 — Inspeccionar imágenes y trimaps

**Objetivo:** Cargar imágenes y trimaps directamente desde el disco (sin pasar por el `Dataset`) para entender qué forma tienen y qué información codifica cada cosa.

**Enunciado:**

1. Definí una función `read_pet_images` que reciba el directorio del dataset, una cantidad `n` y el nombre del split (`'trainval'` o `'test'`), y devuelva dos listas paralelas con las primeras `n` imágenes y sus respectivos trimaps como tensores. Los listados de nombres de archivo de cada split están en `annotations/trainval.txt` y `annotations/test.txt`. Cada línea de esos archivos tiene la forma `<nombre> <class_id> <species> <breed_id>` — solo nos interesa el primer campo. Las imágenes están en `images/` (extensión `.jpg`) y los trimaps en `annotations/trimaps/` (extensión `.png`).
2. Llamala con `n=4` sobre el split de trainval.
3. Visualizá las 4 imágenes y sus 4 trimaps en una grilla de 2×4 — fila superior con las imágenes, fila inferior con los trimaps. Acordate de que matplotlib espera tensores con orden `(H, W, C)` mientras que los tensores de PyTorch vienen como `(C, H, W)`: hay que reordenar los ejes antes de mostrar.

> **Pista 1:** En `torchvision.io` hay una función para leer imágenes que devuelve directamente un tensor `uint8`. Para los trimaps **no** hace falta forzar RGB: vienen como paleta indexada de un solo canal con valores `{1, 2, 3}`, así que la lectura por default ya entrega lo que querés.
>
> **Pista 2:** Los trimaps tienen muy pocos valores únicos (`1`=mascota, `2`=fondo, `3`=borde). Si los mostrás directamente con un colormap continuo van a verse casi negros. Pasale a `imshow` un colormap discreto (por ejemplo `cmap='viridis'` o `cmap='tab10'`) para que los tres valores se distingan.
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
    # masks[i] tiene shape (1, H, W). Mostramos el único canal con un colormap
    # discreto que distinga los tres valores del trimap.
    axs[1, i].imshow(masks[i][0], cmap='viridis')
    axs[1, i].set_title(f'Trimap {i}  (vals: {torch.unique(masks[i]).tolist()})')
    axs[1, i].axis('off')
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirando los trimaps: ¿por qué creés que en la mayoría de las imágenes la clase **fondo** ocupa más píxeles que la mascota, aún cuando la cámara está claramente apuntando al animal? ¿Qué problema podría traer ese desbalance al entrenar una red de segmentación con la cross-entropy "estándar" (sin pesos por clase)?
::::


::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Aunque las fotos de Pet están centradas en la mascota y rara vez la mascota es chica en relación con el cuadro, igual el **fondo** suele cubrir más píxeles. Un perro o gato típicos tienen una silueta que ocupa quizá un 25-35% del área de la imagen; el otro 65-75% es piso, pasto, sofá, mesa, pared, lo que sea. Si sumás los píxeles a lo largo de todo el dataset, fondo se lleva alrededor del 70% y la mascota el 30%. Hay desbalance, suave pero claro.

El problema con cross-entropy "plana" (las dos clases con el mismo peso) es que la pérdida se domina por la clase mayoritaria. Una red que se limite a predecir "fondo" en cada píxel ya acierta ~70% de los píxeles y obtiene una pérdida baja, sin haber aprendido nada útil sobre la silueta del animal. Como el gradiente sigue mayormente la dirección que reduce la pérdida del fondo, la red se queda en ese mínimo "perezoso" y nunca aprende a discriminar al sujeto.

La solución estándar es **ponderar la pérdida** dándole más peso a los píxeles de la clase minoritaria (mascota), o equivalentemente menos peso al fondo. Eso es exactamente lo que vamos a hacer en la Sección D. En problemas multiclase con muchas clases minoritarias el efecto es aún más fuerte que en este caso binario.
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 2 — Construir DataLoaders
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Instanciar `PetSegDataset` y armar los `DataLoader`

**Objetivo:** Usar la clase `PetSegDataset` (preescrita) para crear los datasets de train y val, envolverlos en `DataLoader` e inspeccionar la forma de un batch.

**Enunciado:**

1. Definí el tamaño de crop como una tupla `(256, 256)`. Este tamaño no es arbitrario: la U-Net que vamos a implementar después baja la resolución 4 veces dividiendo por 2 cada vez, así que el input necesita ser **divisible por `2^4 = 16`** para que las cuentas cierren. 256 es la potencia de 2 más cómoda en este rango; la justificación detallada aparece en la sección C.
2. Creá dos datasets:
   - **train:** instanciá `PetSegDataset` sobre el split `'trainval'` con un `subset_size` de 1500 imágenes y data augmentation activada. El subset es por una razón puramente práctica: Pet tiene ~3700 imágenes en trainval y entrenar con todas tomaría más de media hora; con 1500 alcanzamos `val_acc` razonable en ~20 minutos.
   - **val:** instanciá `PetSegDataset` sobre el split `'test'` con todo el dataset (sin subset) y la augmentation desactivada — en validación no queremos que cada época vea una versión distinta de la misma imagen.
3. Envolvé cada dataset en un `DataLoader` con tamaño de batch 8, descartando el último batch si queda incompleto y con un par de workers para paralelizar la lectura del disco. Acordate de que el train se baraja entre épocas y val no.
4. Pedile al iterador de train su primer batch e imprimí la forma de las imágenes y las máscaras, junto con el rango de valores de las máscaras (mínimo y máximo). Las imágenes deberían tener forma `(8, 3, 256, 256)` con valores normalizados (no en [0, 1]); las máscaras `(8, 256, 256)` con valores en `{0, 1, 255}` (fondo, mascota, ignorar).

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
# crop_size 256x256 está elegido para U-Net: bajamos resolución 4 veces
# dividiendo por 2 (256→128→64→32→16 en el cuello de la U) y todas las
# divisiones cierran en enteros porque 256 es divisible por 2^4 = 16.
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
print(f"Y range: min={Y.min().item()}  max={Y.max().item()}")
print(f"Valores únicos en Y (primer batch): "
      f"{torch.unique(Y).tolist()}")
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

El batch de imágenes tiene shape `(8, 3, 256, 256)` y el batch de máscaras tiene shape `(8, 256, 256)` — sin la dimensión de canales. ¿Por qué la máscara no tiene canales? ¿Qué está representando cada valor del tensor `Y`, y qué significa específicamente el valor `255`?
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

La máscara no tiene dimensión de canales porque **cada píxel guarda un único entero** que representa la clase a la que pertenece, no un vector de probabilidades ni un color. El tensor `Y[b, i, j]` contiene un `0` (fondo) o un `1` (mascota) — la clase semántica de ese píxel — o un `255`, que es nuestro `IGNORE_INDEX` para los píxeles del borde del trimap original. Esos píxeles no se toman en cuenta para calcular la pérdida ni para la métrica.

Esa convención —"target categórico como tensor de índices"— es la que espera `nn.CrossEntropyLoss` cuando se usa para clasificación multiclase, también a nivel de píxel: `inputs` tiene forma `(B, C, H, W)` con los logits por clase, y `targets` tiene forma `(B, H, W)` con el índice de la clase correcta. La pérdida calcula internamente el softmax sobre el eje de canales y compara con el índice del target. Por eso no necesitamos una codificación one-hot ni canales explícitos en la máscara: PyTorch los maneja por nosotros. El parámetro `ignore_index=255` que vamos a pasarle a la loss es lo que descarta los píxeles del borde sin que tengamos que filtrarlos a mano.
```
::::


::::cell{#secB type=markdown role=section}
---
## Sección B: Bloques de la U-Net

Vamos a construir la U-Net armando primero las piezas de Lego que la componen. La idea es que cada bloque sea autocontenido y testeable: una vez que cada uno pasa su test, ensamblar la red completa es casi mecánico.

Las piezas son cinco:

1. **`SimpleConvolution`** — el bloque de **doble convolución** que se repite en cada nivel de la U. Cada convolución 3×3 va con `padding=1` para preservar la resolución espacial.
2. **`DownConvolution`** — el bloque del lado descendente: maxpool + doble conv. Reduce la resolución espacial a la mitad (vía maxpool) y aumenta los canales (vía doble conv).
3. **`UpConvolution`** — el bloque del lado ascendente: doble conv + convolución transpuesta. Procesa las features que llegan concatenadas y duplica la resolución para subir un nivel.
4. **`LastConvolution`** — el bloque final: doble conv + convolución 1×1 que mapea de los 64 canales que llegan a las `num_classes` de salida.
5. **`crop_img`** — función auxiliar que recorta un tensor para que coincida espacialmente con otro. En esta versión moderna de la U-Net **no la usamos dentro de la red** (las dimensiones cuadran solas gracias al padding), pero la implementamos como ejercicio: en la U-Net del paper original, donde las convs no tienen padding, esta operación es indispensable para alinear los skips antes de concatenar. Es un buen ejercicio sobre slicing de tensores 4D y entender por qué la versión moderna se simplifica.

> **Nota sobre el padding:** las convoluciones 3×3 con `padding=1` preservan la resolución espacial — el output de la doble convolución tiene exactamente la misma forma que el input. La consecuencia para las skip connections es importante: como cada nivel de la bajada y el correspondiente nivel de la subida tienen exactamente la misma forma espacial, los podemos concatenar **directamente**, sin recortar nada. Eso es una simplificación significativa respecto del paper original (Ronneberger 2015), donde cada doble convolución sin padding mordía 4 píxeles y los skips del lado descendente terminaban siendo más grandes que los del ascendente — había que recortarlos al centro antes de concatenar. La variante moderna con padding (la que vamos a implementar) es la que se usa hoy en producción.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 3 — SimpleConvolution
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — `SimpleConvolution`: bloque de doble convolución

**Objetivo:** Implementar el bloque que aparece en cada nivel de la U-Net: dos convoluciones 3×3 con ReLU intercaladas, más un dropout suave al final.

![](https://miro.medium.com/max/640/1*Uan1yYCi3ZO1xrtLohyWzg.png)

**Enunciado:**

Implementá una clase `SimpleConvolution` (subclase de `nn.Module`) cuyo constructor reciba la cantidad de canales de entrada y la cantidad de canales de salida. El bloque tiene que aplicar, en orden:

1. Una primera convolución 3×3 **con padding=1** que mapee de los canales de entrada a los de salida, seguida de una BatchNorm 2D sobre los canales de salida y una ReLU.
2. Una segunda convolución 3×3 también con padding=1 que mantenga la cantidad de canales (entrada y salida igual a los canales de salida del paso anterior), seguida de otra BatchNorm 2D y otra ReLU.
3. Un dropout con probabilidad 0.1 al final.

Para una entrada de forma `(B, c_in, H, W)`, la salida tiene que ser `(B, c_out, H, W)` — el padding=1 hace que cada convolución 3×3 preserve exactamente la resolución espacial. La BatchNorm, la ReLU y el Dropout tampoco cambian la forma del tensor.

> **Pista:** Podés guardar la pila de capas en un `nn.Sequential` dentro de `__init__` y que `forward` sea casi trivial.

> **Nota sobre BatchNorm:** la U-Net del paper original (2015) **no usa BatchNorm** — la técnica recién se popularizó después. Sin BN una U-Net entrenada **desde cero** sobre datasets chicos cuesta mucho convergir: los gradientes se atenúan en redes profundas y la red termina en mínimos triviales (predecir siempre la clase mayoritaria). Hoy todas las implementaciones modernas incluyen BatchNorm después de cada Conv2d. Lo agregamos por la misma razón.
::::


::::cell{#ej3-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class SimpleConvolution(nn.Module):
    """
    Doble (Conv 3x3 + BatchNorm + ReLU) + Dropout(0.1).
    Es el bloque que aparece en cada nivel de la U-Net.

    Padding=1 en cada conv 3x3: la resolución espacial se preserva.

    Entrada: (B, input_channel, H, W)
    Salida:  (B, output_channel, H, W)
    """
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # padding=1 en convs 3x3: el output mantiene H y W del input.
        # BatchNorm después de cada Conv2d: estabiliza los gradientes en redes
        # profundas. La U-Net original del paper (2015) no la usaba pero hoy
        # es estándar — sin BN la red entrenada desde cero converge a soluciones
        # triviales como "predecir siempre la clase mayoritaria".
        # Dropout suave (0.1): este bloque se usa 9 veces a lo largo de la
        # U-Net, así que un dropout chico se acumula. Subirlo a 0.2 ahoga la
        # señal y dificulta el aprendizaje desde scratch.
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        return self.block(x)
```
::::


::::cell{#test-ej3 type=code role=test}
```python
# ─── Test SimpleConvolution ────────────────────────────────────────────────
# Input chico (1x1x32x32) para que el test apenas use memoria.
# Con padding=1 las convs 3x3 preservan la resolución espacial: 32x32 → 32x32.
block = SimpleConvolution(1, 16)
inp = torch.rand(1, 1, 32, 32)
out = block(inp)
assert out.shape == (1, 16, 32, 32), f"Forma incorrecta: {tuple(out.shape)}"
print("Test SimpleConvolution OK.")
del block, inp, out
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 4 — DownConvolution
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — `DownConvolution`: bajar un nivel

**Objetivo:** Implementar el bloque del camino descendente: primero un maxpool 2×2 que reduce la resolución a la mitad, después una `SimpleConvolution` que actualiza los canales.

![](https://miro.medium.com/max/640/1*9zoULdYOeKQsLWQGExhVlQ.png)

**Enunciado:**

Implementá una clase `DownConvolution` (subclase de `nn.Module`) cuyo constructor reciba canales de entrada y de salida. El bloque tiene que aplicar, en orden:

1. Un max-pooling 2×2 con stride 2 (divide a la mitad alto y ancho).
2. El bloque de doble convolución del ejercicio anterior, mapeando de los canales de entrada a los de salida.

Para una entrada de forma `(B, c_in, H, W)` con `H` y `W` pares, la salida tiene que ser `(B, c_out, H/2, W/2)` — la única reducción espacial viene del maxpool, porque la doble convolución preserva resolución.

> **Pista:** Podés reutilizar la `SimpleConvolution` que acabás de definir guardándola como un atributo del módulo. No hace falta reimplementarla.
::::


::::cell{#ej4-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class DownConvolution(nn.Module):
    """
    MaxPool 2x2 + SimpleConvolution. Es el bloque del camino descendente.

    La única reducción espacial viene del maxpool (la doble conv preserva).

    Entrada: (B, input_channel, H, W)  con H, W pares.
    Salida:  (B, output_channel, H//2, W//2)
    """
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # MaxPool con kernel_size = stride = 2: divide resolución por 2.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Reutilizamos el bloque del ejercicio anterior.
        self.conv = SimpleConvolution(input_channel, output_channel)

    def forward(self, x):
        return self.conv(self.pool(x))
```
::::


::::cell{#test-ej4 type=code role=test}
```python
# ─── Test DownConvolution ──────────────────────────────────────────────────
block = DownConvolution(16, 32)
inp = torch.rand(1, 16, 32, 32)
out = block(inp)
# 32 → maxpool → 16; simpleconv preserva → 16. canales 16 → 32.
assert out.shape == (1, 32, 16, 16), f"Forma incorrecta: {tuple(out.shape)}"
print("Test DownConvolution OK.")
del block, inp, out
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 5 — UpConvolution
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej5-enunciado type=markdown role=enunciado}
### Ejercicio 5 — `UpConvolution`: subir un nivel

**Objetivo:** Implementar el bloque del camino ascendente: una `SimpleConvolution` que procesa las features que llegan ya concatenadas, seguida de una **convolución transpuesta** que duplica la resolución espacial.

![](https://miro.medium.com/max/640/1*nmfwdmaW5A7_zxI0BcPcGQ.png)

**Enunciado:**

Implementá una clase `UpConvolution` (subclase de `nn.Module`) cuyo constructor reciba canales de entrada y de salida. El bloque tiene que aplicar, en orden:

1. La doble convolución del Ej. 3, que recibe el tensor concatenado (skip + nivel inferior) con los canales de entrada y los baja a los canales de salida. Como la doble convolución preserva la resolución espacial, la salida tiene el mismo alto y ancho que la entrada.
2. Una convolución transpuesta con kernel 2×2 y stride 2 que duplique la resolución espacial y, además, **reduzca los canales a la mitad** respecto de la salida del paso anterior. La razón de bajar los canales acá: a la salida de este bloque concatenamos con una skip connection del nivel superior que ya tiene esa cantidad de canales — para que al concatenar quede una cantidad redonda (skip + ascendente, cada uno con la mitad), conviene que el ascendente venga ya con la mitad.

Para una entrada de forma `(B, c_in, H, W)`, la salida tiene que ser `(B, c_out // 2, 2H, 2W)` — la doble convolución preserva la resolución y la convolución transpuesta con stride 2 la duplica.

> **Pista:** La convolución transpuesta con kernel 2×2 y stride 2 es la inversa "de tamaño" de un MaxPool 2×2: por cada píxel de entrada produce un parche 2×2 en la salida. Repasá la sección sobre convolución transpuesta del notebook teórico si querés volver a ver cómo opera.
::::


::::cell{#ej5-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class UpConvolution(nn.Module):
    """
    SimpleConvolution + ConvTranspose2d. Es el bloque del camino ascendente.

    Procesa las features que vienen concatenadas (skip + nivel inferior) y
    sube un nivel duplicando la resolución y reduciendo los canales a la
    mitad — porque arriba va a concatenarse con una skip que ya tiene esa
    cantidad de canales.

    Entrada: (B, input_channel, H, W)
    Salida:  (B, output_channel // 2, 2H, 2W)
    """
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv = SimpleConvolution(input_channel, output_channel)
        # Conv transpuesta con kernel=stride=2: duplica el alto y el ancho,
        # y baja los canales a la mitad para que al concatenar con la skip
        # de arriba quede una cantidad de canales redonda.
        self.upconv = nn.ConvTranspose2d(
            output_channel, output_channel // 2,
            kernel_size=2, stride=2)

    def forward(self, x):
        return self.upconv(self.conv(x))
```
::::


::::cell{#test-ej5 type=code role=test}
```python
# ─── Test UpConvolution ────────────────────────────────────────────────────
block = UpConvolution(64, 32)
inp = torch.rand(1, 64, 16, 16)
# 16 → simpleconv preserva → 16 → trans conv x2 → 32
# canales: 64 → 32 (simpleconv) → 16 (trans conv halves)
out = block(inp)
assert out.shape == (1, 16, 32, 32), f"Forma incorrecta: {tuple(out.shape)}"
print("Test UpConvolution OK.")
del block, inp, out
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 6 — LastConvolution
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej6-enunciado type=markdown role=enunciado}
### Ejercicio 6 — `LastConvolution`: bloque final

**Objetivo:** Implementar el bloque final que cierra la U-Net: `SimpleConvolution` que termina de procesar las features y una **convolución 1×1** que mapea a `num_classes` canales (uno por clase semántica).

![](https://miro.medium.com/max/720/1*cqs5XJRsBXS0RAkdIl_wUQ.png)

**Enunciado:**

Implementá una clase `LastConvolution` (subclase de `nn.Module`) cuyo constructor reciba **tres** parámetros: canales de entrada, canales intermedios y número de clases. El bloque tiene que aplicar, en orden:

1. La doble convolución del Ej. 3, que baja los canales de entrada a los canales intermedios (típicamente 64) y preserva la resolución espacial.
2. Una convolución 1×1 que mezcle esos canales intermedios y produzca un canal por clase. Recordá que la convolución 1×1 tampoco cambia la resolución espacial: actúa como una capa lineal por píxel sobre la dimensión de canales.

Para una entrada de forma `(B, c_in, H, W)`, la salida tiene que ser `(B, num_classes, H, W)` — el alto y ancho se preservan en todo el bloque.

> **Pista:** La convolución 1×1 sobre features con `c_int` canales y salida `num_classes` equivale a aplicar la misma matriz `(num_classes, c_int)` a cada vector de features, posición por posición. Es un buen ejercicio mental verificar por qué eso es lo mismo que una capa lineal píxel a píxel.
::::


::::cell{#ej6-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class LastConvolution(nn.Module):
    """
    SimpleConvolution + Conv2d 1x1. Es el bloque que cierra la U-Net.

    La 1x1 mapea de `output_channel` canales (típicamente 64) a num_classes,
    produciendo los logits por clase para cada píxel. Preserva resolución
    en todo el bloque.

    Entrada: (B, input_channel, H, W)
    Salida:  (B, num_classes, H, W)
    """
    def __init__(self, input_channel, output_channel, num_classes):
        super().__init__()
        self.conv = SimpleConvolution(input_channel, output_channel)
        # Conv 1x1: una capa lineal por píxel sobre la dim de canales.
        self.final = nn.Conv2d(output_channel, num_classes, kernel_size=1)

    def forward(self, x):
        return self.final(self.conv(x))
```
::::


::::cell{#test-ej6 type=code role=test}
```python
# ─── Test LastConvolution ──────────────────────────────────────────────────
block = LastConvolution(32, 16, num_classes=3)
inp = torch.rand(1, 32, 32, 32)
out = block(inp)
# Resolución se preserva en todo el bloque: 32 → 32.
# canales: 32 → 16 (simpleconv) → 3 (1x1)
assert out.shape == (1, 3, 32, 32), f"Forma incorrecta: {tuple(out.shape)}"
print("Test LastConvolution OK.")
del block, inp, out
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 7 — crop_img
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej7-enunciado type=markdown role=enunciado}
### Ejercicio 7 — `crop_img`: alinear tensores espacialmente

**Objetivo:** Implementar una función auxiliar que recorta un tensor en el centro para que coincida espacialmente con otro.

![](https://miro.medium.com/max/720/1*2XyH7YGv7MuJWPycqx7hew.png)

**Contexto:**

En la U-Net **del paper original** (Ronneberger 2015), las convoluciones 3×3 son **sin padding** y van mordiendo píxeles del borde a medida que avanzan. Eso hace que los feature maps del lado descendente terminen siendo **más grandes** espacialmente que los del ascendente con los que se concatenan. Antes de concatenar hay que **recortar el centro** del tensor más grande para que las dos formas espaciales coincidan exactamente.

Nuestra U-Net moderna con padding=1 **no necesita esta operación dentro de la red** (las convs preservan resolución y los skips cuadran solos). Igual implementamos `crop_img` por su valor pedagógico:

- En la U-Net del paper original, **es indispensable**: cada concatenación entre el lado descendente y el ascendente requiere un center-crop del lado más grande. Saber cómo se hace es entender por qué la variante moderna con padding es una simplificación significativa.
- Es un buen ejercicio sobre slicing de tensores 4D y manejo simétrico de bordes — operaciones que aparecen seguido cuando se trabaja con redes de visión.

**Enunciado:**

Implementá una función `crop_img` que reciba dos tensores 4D — el "fuente" (más grande, el que se recorta) y el "objetivo" (el que dicta la forma espacial final). Asumí que ambos tienen forma `(B, C, H, W)` con `B` y los canales eventualmente distintos.

Pasos sugeridos:

1. Calculá la diferencia entre el alto del fuente y el del objetivo (idem para el ancho).
2. Repartí esa diferencia simétricamente entre el borde de arriba y el de abajo (idem izquierda/derecha). En el caso de diferencia impar, no es crítico cómo resolvas el píxel suelto siempre que el resultado tenga la forma esperada.
3. Devolvé el fuente con esos bordes recortados.

> **Pista:** Indexar tensores con slices te deja extraer un parche directo: `tensor[:, :, top:bottom, left:right]`. No hace falta ningún `for`.
::::


::::cell{#ej7-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
def crop_img(source_tensor, target_tensor):
    """
    Center-crop de source para que coincida espacialmente con target.

    Parámetros:
    source_tensor (Tensor): (B, C, sH, sW) — el más grande, lo recortamos.
    target_tensor (Tensor): (B, *, tH, tW) — el más chico, dicta la forma.

    Retorna:
    cropped (Tensor): (B, C, tH, tW) — source recortado en el centro.
    """
    _, _, sH, sW = source_tensor.shape
    _, _, tH, tW = target_tensor.shape
    # Cuánto tenemos que sacar en cada dirección.
    dh = sH - tH
    dw = sW - tW
    # Repartimos la diferencia simétricamente: arriba/izquierda dh//2,
    # abajo/derecha lo que sobra (importa cuando dh es impar).
    top  = dh // 2
    left = dw // 2
    return source_tensor[:, :, top:top + tH, left:left + tW]
```
::::


::::cell{#test-ej7 type=code role=test}
```python
# ─── Test crop_img ─────────────────────────────────────────────────────────
src = torch.rand(1, 16, 32, 32)
tgt = torch.rand(1, 8, 20, 20)
cropped = crop_img(src, tgt)
assert cropped.shape == (1, 16, 20, 20), f"Forma incorrecta: {tuple(cropped.shape)}"
# Chequeo de "centrado": el contenido del recorte debe coincidir con el
# parche central de src.
assert torch.allclose(cropped, src[:, :, 6:26, 6:26])
print("Test crop_img OK.")
del src, tgt, cropped
```
::::


::::cell{#secC type=markdown role=section}
---
## Sección C: Ensamblar la U-Net completa

Con los cinco bloques listos, armar la red es seguir el diagrama. La forma de "U" del modelo se ve directamente en el código: dos listas paralelas de bloques (uno de bajada y uno de subida) que se conectan por las skip connections.
::::


::::cell{#secC-shapes type=markdown role=scaffolding}
### Recorrido de formas con `crop_size = (256, 256)`

Antes de implementar conviene ver cómo varían las formas espaciales y de canales a lo largo de la red, para que el código tenga sentido. Con un input de `(B, 3, 256, 256)` (donde `NC = NUM_CLASSES`):

| Paso | Bloque | Forma de salida | Comentario |
|---|---|---|---|
| 0 | input | `(3, 256, 256)` | imagen normalizada |
| 1 | `SimpleConvolution(3, 64)` | `(64, 256, 256)` | **skip1** — preserva resolución |
| 2 | `DownConvolution(64, 128)` | `(128, 128, 128)` | **skip2** — pool ÷2 |
| 3 | `DownConvolution(128, 256)` | `(256, 64, 64)` | **skip3** |
| 4 | `DownConvolution(256, 512)` | `(512, 32, 32)` | **skip4** |
| 5 | `DownConvolution(512, 1024)` | `(1024, 16, 16)` | fondo de la U |
| 6 | `ConvTranspose2d(1024, 512)` | `(512, 32, 32)` | bridge: sube un nivel |
| 7 | concat con **skip4** | `(1024, 32, 32)` | 512 + 512 |
| 8 | `UpConvolution(1024, 512)` | `(256, 64, 64)` | conv preserva → trans conv ×2 |
| 9 | concat con **skip3** | `(512, 64, 64)` | 256 + 256 |
| 10 | `UpConvolution(512, 256)` | `(128, 128, 128)` | |
| 11 | concat con **skip2** | `(256, 128, 128)` | 128 + 128 |
| 12 | `UpConvolution(256, 128)` | `(64, 256, 256)` | |
| 13 | concat con **skip1** | `(128, 256, 256)` | 64 + 64 |
| 14 | `LastConvolution(128, 64, NC)` | `(NC, 256, 256)` | output |

La salida es `(NC, 256, 256)`: por cada píxel del input, `NC` logits que el `argmax` colapsa a la clase predicha. **El output tiene exactamente la misma resolución que el input**, así que la red supervisa cada píxel — no se pierde nada en los bordes. Esa es la ganancia operativa de usar `padding=1` en las convoluciones.

> **Por qué `crop_size = 256`:** elegimos un tamaño que sea divisible por 16 (=`2^4`) para que las cuatro divisiones por 2 del lado descendente cierren todas en enteros. 256 = 2^8 es la potencia de 2 más cómoda en este rango: al bajar 4 veces llega a 16 y al subir vuelve a 256. Si elegiéramos un número como 257 (impar), el primer pool daría 128 con un píxel de pérdida, después 64, después 32, después 16 (perdiendo medio píxel por nivel) y al subir no podríamos reconstruir 257 exactamente — habría desencuentros entre skips. Con `padding=1` la simetría se mantiene solo si el tamaño de entrada es divisible por la cantidad de niveles de pool.

> **Atención al primer paso ascendente:** después del fondo de la U (`DownConvolution(512, 1024)`) la red **necesita un upsampling adicional ANTES** del primer `UpConvolution`. Lo hacemos con una `ConvTranspose2d(1024, 512, kernel_size=2, stride=2)` independiente, que duplica la resolución (16 → 32) y baja los canales a la mitad para que al concatenar con `skip4` (que tiene 512 canales) dé `1024 = 512 + 512` canales para el primer `UpConvolution(1024, 512)`.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 8 — UNet
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej8-enunciado type=markdown role=enunciado}
### Ejercicio 8 — Clase `UNet`

**Objetivo:** Ensamblar todos los bloques en la red completa.

**Enunciado:**

Implementá una clase `UNet` (subclase de `nn.Module`) cuyo constructor reciba la cantidad de canales de entrada y la cantidad de clases. Tu trabajo es ensamblar los bloques que ya tenés siguiendo la tabla de shapes de la sección C: para cada transición de la tabla, el bloque que la cumple es siempre uno de los que ya implementaste.

**En el `__init__`** declará todos los submódulos que vas a necesitar:

- El bloque inicial que va antes del primer pooling. Es el único que no tiene maxpool delante y mapea de los canales de entrada de la imagen a 64 canales.
- Cuatro bloques descendentes, uno por nivel del encoder. La tabla te dice los pares de canales de entrada/salida en cada uno (la regla simple: en cada bajada se duplican los canales).
- El upsampling intermedio entre el fondo de la U y el primer bloque ascendente: la operación que duplica la resolución espacial y baja los canales de 1024 a 512. Se puede armar con una sola convolución transpuesta de kernel 2×2 y stride 2 (no hace falta envolverla en un módulo propio).
- Tres bloques ascendentes, uno por cada nivel del decoder. Mirá la tabla para deducir qué canales tiene cada uno a la entrada y a la salida — recordá que cada UpConv recibe un tensor concatenado (lado descendente + lado ascendente) y entrega un tensor con la mitad de los canales de salida intermedios, listo para concatenar de nuevo arriba.
- El bloque final, que mapea a `num_classes` canales.

**En el `forward`** seguí el diagrama de la U:

1. Aplicá el bloque inicial y los cuatro descendentes, guardando la salida de los **cuatro primeros niveles** como skip connections (las vas a usar más adelante para concatenar). El quinto nivel — el fondo de la U — no necesita skip.
2. Aplicá el upsampling intermedio.
3. Concatená con la skip del nivel correspondiente y aplicá el siguiente bloque ascendente. Repetí tres veces y cerrá con el bloque final. Como las convs preservan resolución (gracias al `padding=1`), las skips y los feature maps ascendentes ya cuadran espacialmente — concatenan directamente, sin recortar.

> **Pista:** En PyTorch hay una función para concatenar una lista de tensores sobre un eje. Para una pila `(B, C, H, W)` el eje de canales es el `dim=1`.
::::


::::cell{#ej8-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class UNet(nn.Module):
    """
    U-Net moderna (variante con padding=1 + BatchNorm), inspirada en
    Ronneberger et al. 2015. Encoder-decoder con skip connections.

    Parámetros:
    input_channel (int): canales de entrada (3 para RGB).
    num_classes (int): número de clases de salida (2 para Pet binario).

    Entrada: (B, input_channel, 256, 256)  → Salida: (B, num_classes, 256, 256)
    """
    def __init__(self, input_channel, num_classes):
        super().__init__()
        # ─── Encoder (camino descendente) ───────────────────────────────────
        self.start = SimpleConvolution(input_channel, 64)
        self.down1 = DownConvolution(64, 128)
        self.down2 = DownConvolution(128, 256)
        self.down3 = DownConvolution(256, 512)
        self.down4 = DownConvolution(512, 1024)
        # ─── Bridge: primera subida (sin doble-conv) ───────────────────────
        # Después del fondo necesitamos un upsampling antes de poder llamar
        # al primer UpConvolution (que ya espera el tensor concatenado con
        # la skip4). Lo hacemos con una ConvTranspose pelada.
        self.bridge = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # ─── Decoder (camino ascendente) ───────────────────────────────────
        self.up1 = UpConvolution(1024, 512)
        self.up2 = UpConvolution(512, 256)
        self.up3 = UpConvolution(256, 128)
        # ─── Bloque final: doble conv + 1x1 a num_classes ──────────────────
        self.last = LastConvolution(128, 64, num_classes)

    def forward(self, x):
        # ─── Bajada: guardamos los skips para concatenar después ───────────
        skip1 = self.start(x)        # (B,   64, 256, 256)
        skip2 = self.down1(skip1)    # (B,  128, 128, 128)
        skip3 = self.down2(skip2)    # (B,  256,  64,  64)
        skip4 = self.down3(skip3)    # (B,  512,  32,  32)
        x     = self.down4(skip4)    # (B, 1024,  16,  16)

        # ─── Subida: bridge + 3 UpConvolutions ─────────────────────────────
        # Como las convs preservan resolución, las skips y los feature maps
        # ascendentes cuadran espacialmente sin necesidad de crop_img.
        x = self.bridge(x)                                # (B, 512,  32,  32)
        x = self.up1(torch.cat([skip4, x], dim=1))        # (B, 256,  64,  64)
        x = self.up2(torch.cat([skip3, x], dim=1))        # (B, 128, 128, 128)
        x = self.up3(torch.cat([skip2, x], dim=1))        # (B,  64, 256, 256)

        # ─── Bloque final ──────────────────────────────────────────────────
        x = self.last(torch.cat([skip1, x], dim=1))       # (B, NC, 256, 256)
        return x
```
::::


::::cell{#test-ej8 type=code role=test}
```python
# ─── Test UNet ─────────────────────────────────────────────────────────────
# Test con num_classes=3 e input chico para no quemar memoria. Verificamos
# que la forma de salida es la misma que la de entrada — esa es la propiedad
# clave de la U-Net con padding=1.
unet_test = UNet(input_channel=3, num_classes=3)
inp = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    out = unet_test(inp)
print(f"input  : {tuple(inp.shape)}")
print(f"output : {tuple(out.shape)}  (esperado: (1, 3, 256, 256))")
assert out.shape == (1, 3, 256, 256), f"Forma incorrecta: {tuple(out.shape)}"
print("Test UNet OK.")
del unet_test, inp, out
```
::::


::::cell{#ej8-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La U-Net que implementaste tiene dos diferencias clave respecto de la del paper original (Ronneberger 2015): nuestras convoluciones 3×3 usan `padding=1` (las del paper no tenían padding), y agregamos `BatchNorm2d` después de cada `Conv2d` (el paper no la usaba). Para cada uno de esos cambios, indicá qué consecuencia tiene en la forma del output (input 572×572 vs nuestro 256×256) y/o en el comportamiento del entrenamiento.
::::


::::cell{#ej8-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

**1. `padding=1` en las convoluciones 3×3.**

Cada convolución 3×3 sin padding "muerde" un píxel en cada borde — para una doble convolución eso son 4 píxeles menos por dimensión espacial. Acumulado a lo largo de los cinco niveles de la bajada y los cuatro de la subida, en el paper original entraba `572×572` y salía `388×388` (una pérdida considerable). Además los feature maps del lado descendente terminaban siendo más grandes que los del ascendente con los que se concatenaban, así que las skip connections requerían un recorte explícito al centro antes del concat — esa era la razón de existir de `crop_img` en la implementación fiel al paper.

Con `padding=1` cada conv 3×3 preserva exactamente la forma espacial del input. Las consecuencias son inmediatas: **(a)** el output tiene la misma resolución que la entrada, así que la red predice una clase por cada píxel del input (no solo por un parche central); **(b)** las skips y los feature maps ascendentes ya cuadran espacialmente, así que se concatenan directamente sin recortar nada; **(c)** la red supervisa todos los píxeles de cada imagen durante el entrenamiento, no solo el centro — eso multiplica varias veces la cantidad de píxeles supervisados por imagen y mejora notablemente la calidad de las predicciones cerca de los bordes.

**2. `BatchNorm2d` después de cada Conv2d.**

La BatchNorm normaliza las activaciones por canal antes de pasarlas a la ReLU, manteniéndolas en un rango con media cercana a 0 y varianza cercana a 1. Tiene dos efectos importantes para el entrenamiento de redes profundas:

- **Estabilización del gradiente.** En una red de 9 capas profundas como la nuestra, las activaciones pueden saturar (todo cerca de cero, o todo muy grande) y los gradientes se atenúan o explotan. La BN mantiene esa distribución bajo control en cada paso, lo que permite entrenar con learning rates razonables sin que la red diverja.
- **Mejor convergencia desde scratch.** Sin BN, una U-Net entrenada desde cero sobre un dataset chico tiende a quedarse en mínimos triviales del estilo "predecir siempre la clase mayoritaria del crop", porque el optimizador no puede explorar bien el espacio de parámetros. Con BN la red logra discriminar texturas locales y producir predicciones útiles.

El paper de 2015 no usaba ninguna de las dos cosas porque trabajaba sobre datasets biomédicos con augmentation pesada (deformaciones elásticas) y problemas binarios — un setup que perdona más. Para datasets chicos, multiclase, con augmentation simple, la combinación `padding=1 + BatchNorm` es lo que vuelve la red entrenable en la práctica.
```
::::


::::cell{#secD type=markdown role=section}
---
## Sección D: Entrenamiento

Antes de empezar a entrenar, hay dos cosas que solemos pasar por alto y que importan particularmente en segmentación:

1. **Liberar memoria.** Las celdas de test crearon tensores y modelos pequeños que pueden quedar referenciados todavía. En GPU eso se nota porque cuando creamos la U-Net "real" puede aparecer un OOM si la suma del modelo nuevo + lo que quedó de los tests se pasa de la VRAM de la T4 (~15 GB).
2. **Pesos por clase.** Como vimos en el Ej. 1, el fondo domina el dataset. Sin pesos, la red converge a "todo es fondo". Vamos a calcular los pesos como `1 / frecuencia_de_clase` y pasarlos a la cross-entropy.
::::


::::cell{#setup-cleanup type=code role=setup}
```python
# ─── Limpieza de memoria antes de entrenar ──────────────────────────────────
# Las celdas de test crearon objetos que pueden quedar referenciados. Forzamos
# garbage collection y vaciamos la cache de CUDA. Si no querés tener que pensar
# en esto, lo equivalente es Entorno de ejecución > Reiniciar y ejecutar todo:
# pero en general conviene saber que existen estas dos llamadas.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"VRAM libre tras limpiar: "
          f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB de "
          f"{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB")
```
::::


::::cell{#secD-pesos type=markdown role=scaffolding}
### Cálculo de pesos por clase

La receta es:

1. Recorrer todo el split de train contando cuántos píxeles hay de cada clase (los píxeles `IGNORE_INDEX=255` no se cuentan).
2. Calcular la frecuencia relativa `freq[c] = pixels_clase_c / pixels_totales`.
3. Definir `weights[c] = 1 / sqrt(freq[c] + ε)` con un `ε` pequeño para evitar dividir por 0 si una clase no aparece.
4. **Normalizar** los pesos para que sumen `num_classes`. Eso mantiene el orden de magnitud de la pérdida estable: si los pesos son enormes, la pérdida también lo es y el learning rate adecuado cambia.

Lo usamos pasándole el vector de pesos a `nn.CrossEntropyLoss(weight=weights, ignore_index=255)`.

> **Por qué `1/√freq` y no `1/freq` directamente:** `1/freq` parece la opción "natural" pero produce pesos demasiado desbalanceados cuando hay diferencias grandes de frecuencia entre clases. En datasets multiclase desbalanceados (como VOC2012, donde el fondo ocupa ~66% y las 20 clases minoritarias se reparten el resto) el ratio entre el peso máximo y mínimo llega a ~170×. Bajo esa pérdida la red descubre rápido que **nunca conviene predecir la clase mayoritaria**: equivocarse en un píxel minoritario cuesta cientísimo más, así que la política óptima es repartirse las minoritarias y nunca predecir la dominante. El resultado es una red que tiene accuracy de pixel mucho peor que predecir todo "fondo".
>
> Tomar la **raíz cuadrada** de la frecuencia comprime el rango: en nuestro caso (Pet, fondo ~0.70 / mascota ~0.30) el ratio entre pesos pasa de ~2.3× con `1/freq` a ~1.5× con `1/√freq`. Más importante todavía, en problemas multiclase fuertemente desbalanceados la diferencia es enorme — `1/√freq` mantiene los pesos en un rango razonable. Es la receta que usan, por ejemplo, ENet (Paszke et al. 2016) y SegNet (Badrinarayanan et al. 2017) en sus respectivos papers de segmentación.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 9 — Entrenamiento con pesos por clase
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej9-enunciado type=markdown role=enunciado}
### Ejercicio 9 — Entrenamiento de la U-Net (prueba inicial desde cero)

**Objetivo:** Calcular los pesos por clase, definir la función de pérdida ponderada y entrenar la U-Net sobre Pet **desde cero** (con pesos inicializados al azar) por unas pocas epochs. La idea es confirmar que la arquitectura arranca y produce predicciones útiles — no obtener el mejor modelo posible. En el Ej. 11 vamos a repetir el experimento partiendo de un encoder pre-entrenado y comparar resultados.

> ⚠️ **Atención al tiempo de entrenamiento:** este ejercicio entrena la U-Net por **4 epochs** sobre las 1500 imágenes del subset. En **GPU T4 de Colab tarda aproximadamente 5-7 minutos**.
>
> ¿Por qué solo 4 epochs y no 12 o 20? Porque el cuello de botella **no es la cantidad de iteraciones**: una U-Net de 31M de parámetros entrenada desde cero sobre 1500 imágenes sigue convergiendo lento aunque le des una hora. Con 4 epochs ya vemos la tendencia (sale del baseline trivial de "todo fondo") y nos sobra tiempo para el fine-tuning del Ej. 11, que es donde realmente se ve qué saca al modelo del estancamiento.

**Enunciado:**

Esta es la celda principal de entrenamiento desde cero. Tiene varias partes; el código tiene la estructura armada con bloques numerados, completá los huecos donde dice `# Tu código aquí`.

1. **Conteo de píxeles por clase:** recorré el dataset de train acumulando, para cada clase, cuántos píxeles aparecen en total. Excluí los píxeles marcados como ignorables. Esto puede tardar 1-2 minutos porque toca abrir cada imagen del dataset.
2. **Pesos:** calculá la frecuencia relativa de cada clase y armá un vector de pesos inversamente proporcional a la **raíz cuadrada** de esa frecuencia (cuidado con dividir por cero — sumá un epsilon chico adentro de la raíz). La raíz suaviza el desbalance respecto a `1/freq` puro, que produce pesos demasiado agresivos — la justificación está arriba en el bloque de scaffolding. Normalizá los pesos para que su suma sea igual al número de clases (eso mantiene la pérdida en un orden de magnitud razonable y evita tener que retunear el learning rate). Convertilo a tensor de PyTorch.
3. **Modelo, loss y optimizador:**
   - Instanciá la U-Net con 3 canales de entrada y `NUM_CLASSES` de salida, y mandala al `device`.
   - Definí la pérdida como cross-entropy multiclase **ponderada** con los pesos del paso anterior. Asegurate de pasarle también la opción para que ignore los píxeles marcados con `IGNORE_INDEX` — sin eso, los píxeles del borde contribuyen al gradiente como si fueran clase 255 (que ni siquiera existe) y rompen el entrenamiento.
   - Como optimizador usá Adam con learning rate 1e-3.
4. **Loop de entrenamiento:** entrená por 4 epochs. En cada epoch:
   - Modo train. Para cada batch del iterador de train:
     - Mandá imagen y máscara al device. Convertí la máscara a `long` (la cross-entropy exige índices enteros como target).
     - Forward por la red. La salida tiene shape `(B, NC, 256, 256)` — exactamente la misma resolución espacial que la máscara, así que se pueden comparar directamente sin recortar nada.
     - Calculá la pérdida, backpropagá, hacé el step del optimizador y limpiá los gradientes.
     - Acumulá lo necesario para reportar pérdida promedio y accuracy de pixel al final del epoch.
   - Modo eval. Recorré el iterador de validación midiendo accuracy de pixel sobre val. No olvides envolver el bloque en un contexto `no_grad` para no acumular gradientes inútilmente.
   - Imprimí una línea por epoch con `epoch | train_loss | train_acc | val_acc`.
   - **Guardá el `val_acc` de cada epoch en una lista llamada `val_acc_history`** — la vamos a comparar con los resultados del Ej. 11 (fine-tuning).

> **Pista — accuracy de pixel:** comparar `prediccion == ground_truth` (después del `argmax` sobre canales) te da un tensor booleano. Sumá los `True` y dividí por la cantidad de píxeles "no-ignorables". Para excluir del conteo los píxeles `IGNORE_INDEX`, hacé un AND con una máscara booleana que marque los píxeles válidos del ground truth.
>
> **Nota — qué esperar:** con la receta del lab (subset de 1500 imágenes, 4 epochs, U-Net con padding=1 + BatchNorm, pesos `1/√freq`, augmentation por flip horizontal) `val_acc` debería arrancar en ~0.72-0.78 en epoch 1 y llegar a **~0.78-0.85** en epoch 4. La línea de base "predecir todo fondo" sería ~0.70, así que cualquier valor claramente por encima indica que la red discrimina mascota de fondo, aunque sea de forma rudimentaria. Los resultados verdaderamente buenos van a aparecer en el Ej. 11.
::::


::::cell{#ej9-code type=code role=student-code}
```python
# ─── 1) Conteo de píxeles por clase ─────────────────────────────────────────
# Recorremos pet_train y vamos sumando los píxeles de cada clase, ignorando
# los píxeles marcados como IGNORE_INDEX.

# Tu código aquí

# ─── 2) Pesos: 1 / freq normalizado a sumar NUM_CLASSES ─────────────────────

# Tu código aquí

# ─── 3) Modelo, loss y optimizador ──────────────────────────────────────────

# Tu código aquí

# ─── 4) Loop de entrenamiento (4 epochs, guardando val_acc_history) ────────

# Tu código aquí
```

```python solution
# ─── 1) Conteo de píxeles por clase ─────────────────────────────────────────
# Recorremos pet_train acumulando cuántos píxeles tiene cada clase. No usamos
# DataLoader (no necesitamos batch ni paralelismo): un for sobre el dataset
# alcanza y tiene la ventaja de que no fija el shuffle de DataLoader.
freqs = Counter()
for _, mask in pet_train:
    # mask es un tensor (H, W) con índices. Excluimos IGNORE_INDEX.
    valid = mask[mask != IGNORE_INDEX]
    unique, counts = torch.unique(valid, return_counts=True)
    for c, n in zip(unique.tolist(), counts.tolist()):
        freqs[c] += n
total_pixels = sum(freqs.values())
print(f"Píxeles totales (sin IGNORE_INDEX): {total_pixels:,}")

# Pequeña tabla de frecuencias para verificar que el desbalance es real:
df = pd.DataFrame({
    "clase":    PET_CLASSES,
    "píxeles":  [freqs.get(i, 0) for i in range(NUM_CLASSES)],
    "freq":     [round(freqs.get(i, 0) / total_pixels, 4)
                 for i in range(NUM_CLASSES)],
})
print(df.to_string(index=False))

# ─── 2) Pesos: 1 / sqrt(freq) normalizado a sumar NUM_CLASSES ──────────────
# Usamos 1/sqrt(freq) (no 1/freq) para suavizar el desbalance: con 1/freq el
# fondo queda con peso ~0.02 y las minoritarias con pesos > 1 — la red
# descubre que nunca conviene predecir fondo y colapsa al lado opuesto del
# problema. Con sqrt el ratio max/min entre pesos baja de ~170x a ~13x: el
# fondo todavía se penaliza menos, pero no de forma extrema. El +1e-6 evita
# dividir por cero si una clase no aparece; la normalización a sum=NUM_CLASSES
# mantiene el orden de magnitud de la loss estable (con pesos sin normalizar
# la loss inicial puede ser de varios cientos y el lr habría que retocarlo).
class_freq = np.array([freqs.get(i, 0) / total_pixels
                       for i in range(NUM_CLASSES)])
raw_w   = 1.0 / np.sqrt(class_freq + 1e-6)
weights = raw_w * (NUM_CLASSES / raw_w.sum())
weights = torch.tensor(weights, dtype=torch.float32)
print(f"\nPesos por clase (normalizados a sumar NUM_CLASSES={NUM_CLASSES}):")
for i, n in enumerate(PET_CLASSES):
    print(f"  {n:14s}  freq={class_freq[i]:.4f}  weight={weights[i].item():.3f}")

# ─── 3) Modelo, loss y optimizador ──────────────────────────────────────────
model = UNet(input_channel=3, num_classes=NUM_CLASSES).to(device)
# CrossEntropyLoss ya hace el log_softmax y compara con índices enteros —
# ignore_index=255 es lo que nos asegura que los píxeles del borde no
# contribuyen al gradiente.
criterion = nn.CrossEntropyLoss(
    weight=weights.to(device), ignore_index=IGNORE_INDEX)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\nParámetros del modelo: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# ─── 4) Loop de entrenamiento ───────────────────────────────────────────────
# Guardamos val_acc por epoch para comparar después con el fine-tuning del Ej. 11.
num_epochs = 4
val_acc_history = []
for epoch in range(num_epochs):
    # ── Train ───────────────────────────────────────────────────────────────
    model.train()
    L_sum, n_correct, n_valid = 0.0, 0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device).long()
        # Como la U-Net con padding preserva resolución, y_hat e y tienen
        # la misma forma espacial — se comparan directamente.
        y_hat = model(X)                                  # (B, NC, 256, 256)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métricas: sumamos solo sobre píxeles "válidos" (no-IGNORE).
        with torch.no_grad():
            preds = y_hat.argmax(dim=1)
            mask = (y != IGNORE_INDEX)
            n_correct += ((preds == y) & mask).sum().item()
            n_valid   += mask.sum().item()
            L_sum     += loss.item() * X.size(0)

    train_loss = L_sum / len(pet_train)
    train_acc  = n_correct / n_valid

    # ── Val ─────────────────────────────────────────────────────────────────
    model.eval()
    n_correct_v, n_valid_v = 0, 0
    with torch.no_grad():
        for X, y in val_iter:
            X, y = X.to(device), y.to(device).long()
            y_hat = model(X)
            preds = y_hat.argmax(dim=1)
            mask = (y != IGNORE_INDEX)
            n_correct_v += ((preds == y) & mask).sum().item()
            n_valid_v   += mask.sum().item()
    val_acc = n_correct_v / n_valid_v
    val_acc_history.append(val_acc)

    print(f"epoch {epoch + 1:2d}/{num_epochs}  "
          f"train_loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}")
```
::::


::::cell{#ej9-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Si en lugar de usar pesos por clase entrenaras la misma red con `nn.CrossEntropyLoss()` "plana" (sin `weight`), ¿qué esperarías que pase con la **accuracy de pixel global** y con la **accuracy de pixel sobre la mascota**? Justificá.
::::


::::cell{#ej9-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Sin pesos por clase, la red optimiza una pérdida en la que cada píxel cuenta igual. Como ~70% de los píxeles son fondo, **predecir "fondo" en todos los píxeles** ya da una accuracy global del orden de 0.70. Es un mínimo local muy cómodo: la pérdida es relativamente baja y el gradiente para mejorarlo (aprender a discriminar la mascota) es chiquito porque cualquier desvío hacia "mascota" le suma error de fondo. La red termina convergiendo a algo así:

- **Accuracy global:** ~0.70 — engañosamente alta, porque está tirada hacia arriba por el fondo.
- **Accuracy sobre la mascota:** muy mala (~0.0-0.2) — la red prácticamente no segmenta al animal. La métrica de pixel global oculta esto.

Con pesos por clase (`1 / √freq`), el fondo aporta menos al gradiente y la mascota aporta más — la red se ve obligada a aprender a discriminar la silueta. Eso típicamente:

- **Baja un poco la accuracy global** (porque se ganan errores en píxeles de fondo que antes acertaba "por gravedad").
- **Sube fuertemente la accuracy sobre la mascota y la mIoU** (la métrica estándar en segmentación, que promedia el IoU por clase y por lo tanto es robusta al desbalance).

Conclusión: con desbalance la accuracy global es una métrica engañosa. Conviene mirar **accuracy por clase** o **mIoU**, y usar pesos por clase (o losses como Focal, Dice o Tversky) durante el entrenamiento. En este lab nos quedamos con la accuracy global por simplicidad, pero teniendo presente la limitación.
```
::::


::::cell{#secE type=markdown role=section}
---
## Sección E: Predicción y visualización

Ya tenemos un modelo entrenado. Vamos a usarlo para producir máscaras predichas sobre imágenes del split de validación y compararlas con el ground truth.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 10 — Predicción
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej10-enunciado type=markdown role=enunciado}
### Ejercicio 10 — Visualizar predicciones

**Objetivo:** Tomar imágenes del split de validación, predecir sus máscaras con la red entrenada y mostrar la triple "imagen original / predicción / ground truth" para inspeccionar visualmente qué tan bien funciona el modelo.

**Enunciado:**

1. Implementá una función `label2image` que reciba un tensor 2D `(H, W)` con índices de clase y devuelva un tensor 3D `(H, W, 3)` con los colores RGB correspondientes según `PET_COLORMAP`. Para los píxeles marcados como `IGNORE_INDEX`, podés pintar un color "neutro" (gris) — no es crítico porque son pocos.
2. Tomá 4 imágenes del split de test con la función que implementaste en el Ej. 1.
3. Para cada imagen:
   - Recortá un parche 256×256 **desde el centro** de la imagen, tanto en la imagen como en el trimap correspondiente. Centrar el recorte (en lugar de tomarlo desde una esquina) maximiza la chance de que la mascota quede dentro del cuadro — en Pet los animales suelen estar centrados en la foto.
   - Normalizá la imagen con la misma media/std de ImageNet que usaba el dataset durante el entrenamiento.
   - Pasala por la red en modo eval, dentro de un bloque `no_grad`, y quedate con la clase de mayor logit por píxel (`argmax` sobre la dimensión de canales).
   - Convertí la predicción a imagen RGB con `label2image`. Hacé lo mismo con el ground truth: ojo que el trimap viene con valores `{1, 2, 3}` y nuestro modelo predice `{0, 1}` — vas a tener que aplicar el mismo mapeo que hace el `PetSegDataset` (`2→0`, `1→1`, `3→IGNORE_INDEX`) antes de visualizar.
4. Armá la visualización **separada en dos figuras** de 2 filas cada una (no una sola figura de 4 filas). Cada figura tiene tres columnas: imagen (256×256), predicción (256×256) y ground truth (256×256). Las tres tienen la misma resolución porque la U-Net con padding preserva la forma espacial. Separar en dos figuras hace que cada una sea más cómoda de leer en pantalla y mantiene la leyenda cerca de las imágenes a las que aplica.
5. Llamá a `add_seg_legend(fig)` (helper preescrito en el setup) en **cada** figura, para que ambas tengan arriba una leyenda con el color de cada clase. Sin la leyenda mirando la grilla no se sabe qué clase representa cada color.

> **Pista:** En `torchvision.transforms.functional` hay una función `crop` que toma una imagen y devuelve un crop a partir de coordenadas `top`, `left`, `height` y `width`. Para centrar el crop usá `top = (H - 256) // 2` y `left = (W - 256) // 2`.
::::


::::cell{#ej10-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── label2image: índices de clase → imagen RGB ─────────────────────────────
def label2image(pred):
    """
    Mapea un tensor 2D (H, W) de índices de clase a una imagen RGB (H, W, 3)
    usando PET_COLORMAP. Los píxeles IGNORE_INDEX se pintan de gris.
    """
    colormap = torch.tensor(PET_COLORMAP, device=pred.device, dtype=torch.uint8)
    # Reemplazamos IGNORE_INDEX por 0 (background) para poder indexar; después
    # parcheamos esos píxeles a gris para que se vean distintos.
    safe = pred.clone()
    safe[safe == IGNORE_INDEX] = 0
    img = colormap[safe.long()]
    img[pred == IGNORE_INDEX] = torch.tensor([128, 128, 128],
                                              device=pred.device,
                                              dtype=torch.uint8)
    return img.cpu()


def trimap_to_index(trimap):
    """
    Mapea un trimap original {1, 2, 3} a índices {1, 0, 255} = {pet, bg, ignore}.
    Es el mismo mapeo que aplica PetSegDataset.__getitem__.
    """
    out = torch.full_like(trimap, IGNORE_INDEX)
    out[trimap == 2] = 0
    out[trimap == 1] = 1
    return out


# ─── Predicción sobre un puñado de imágenes del split de test ──────────────
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
    # dentro del cuadro (las imágenes de Pet suelen tener al animal cerca
    # del centro, no en una esquina).
    top  = (H - 256) // 2
    left = (W - 256) // 2
    img_crop  = transforms.functional.crop(img_t, top, left, 256, 256)
    mask_crop = transforms.functional.crop(msk_t, top, left, 256, 256).long()

    # Normalizo la imagen como en el train y paso por la red.
    X = norm(img_crop.float() / 255).unsqueeze(0).to(device)
    y_hat = model(X)                          # (1, NC, 256, 256)
    pred  = y_hat.argmax(dim=1).squeeze(0)    # (256, 256)

    # Ground truth: trimap → índices del modelo. La forma espacial ya es
    # la misma que la pred (256x256), no hay que recortar.
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
# Separar en dos figuras (en lugar de una grilla 4x3 alta) hace que cada
# figura sea más cómoda de leer en una pantalla y deja la leyenda cerca de
# las imágenes a las que aplica.
ROWS_PER_FIG = 2
with torch.no_grad():
    for fig_idx in range(0, n, ROWS_PER_FIG):
        fig, axs = plt.subplots(ROWS_PER_FIG, 3, figsize=(12, 4 * ROWS_PER_FIG))
        for r in range(ROWS_PER_FIG):
            predict_and_render(fig_idx + r, axs[r])
        add_seg_legend(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # deja espacio arriba para la leyenda
        plt.show()
```
::::


::::cell{#ej10-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Comparando las predicciones con el ground truth: ¿en qué tipo de regiones la red anda mejor (zonas amplias del cuerpo, fondos uniformes, etc.) y en cuáles peor (bordes finos, patas y orejas, mascotas chicas en el cuadro)? Proponé al menos dos mejoras concretas del pipeline (datos, arquitectura o entrenamiento) que apunten a los puntos débiles que detectes.
::::


::::cell{#ej10-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Patrón típico observado tras la prueba inicial de 4 epochs sobre Pet (subset 1500):

**Mejor:**
- **Cuerpo central de la mascota y fondos uniformes amplios** — la red identifica bien grandes "blobs" coherentes: el torso del perro o del gato, una pared, un piso. Son patrones de baja frecuencia espacial que el receptive field de U-Net resuelve sin esfuerzo.
- **Mascotas que ocupan buena parte del cuadro** — un perro en plano medio con fondo limpio sale bastante prolijo. El modelo aprendió que "el sujeto suele estar en el centro" como sesgo del dataset.

**Peor:**
- **Bordes finos y siluetas detalladas** — el contorno entre mascota y fondo aparece dentado o algo desplazado. Las orejas, las patas y las colas a menudo se "comen" o quedan recortadas. La causa principal: la pérdida ponderada da igual peso a todos los píxeles de la mascota — los del centro pesan lo mismo que los del borde, así que no hay incentivo extra para afinar el contorno.
  - *Mejora posible:* agregar `Dice loss` o `boundary loss` que penalicen específicamente los errores en el contorno.
- **Animales chicos en el cuadro** (escenas con la mascota en una esquina, parcialmente ocluida o lejos) — la red las detecta peor. El receptive field grande del cuello de la U está pensado para sujetos grandes; los chicos se diluyen.
  - *Mejora posible:* multi-scale training (forzar al modelo a ver la misma imagen a varios crops de distintos tamaños), o más epochs combinadas con augmentation más fuerte (color jitter, rotaciones leves).
- **Diferencias de raza muy distintas a las del subset de train** — entrenamos con 1500 imágenes elegidas al azar; algunas razas están subrepresentadas. Si en val aparece una raza con apariencia atípica, la red la generaliza peor.
  - *Mejora posible:* entrenar con el dataset completo (3680 imágenes), aceptando los ~40 minutos de cómputo, o aplicar augmentation más diversa (color jitter es trivial).

**Mejora general:** los modelos de segmentación de producción rara vez son U-Net puras desde cero — usan backbones pre-entrenados (ResNet, EfficientNet, ViT) en el encoder y agregan multi-scale features (ASPP, FPN). En el Ej. 11 vamos a aplicar exactamente la primera de esas mejoras (encoder ResNet pre-entrenado en ImageNet) y vas a ver el salto cualitativo respecto a esta prueba inicial.
```
::::


::::cell{#secF type=markdown role=section}
---
## Sección F: Fine-tuning desde una U-Net pre-entrenada

Hasta acá entrenamos la U-Net **desde cero**. La prueba inicial del Ej. 9 muestra que la red arranca — supera el baseline trivial de "todo fondo" — pero los resultados son modestos y las predicciones del Ej. 10 lo confirman: blobs aproximados, contornos imprecisos, mascotas chicas que se diluyen. La razón es bien conocida: una red profunda con ~31M de parámetros no puede aprender features visuales útiles a partir de 1500 imágenes. Necesitaría órdenes de magnitud más datos, mucho más cómputo, augmentation pesada o todo lo anterior.

La receta práctica de la última década resuelve esto con **transfer learning**: en lugar de inicializar los pesos del encoder al azar, se reemplazan por los pesos de una red pre-entrenada sobre ImageNet (1.2 millones de imágenes naturales etiquetadas con 1000 clases). El encoder ya viene "sabiendo" extraer bordes, texturas y partes de objetos — y solo hay que afinar esos features para la tarea específica de segmentación. Las primeras epochs del fine-tuning ya producen resultados que el train desde cero no alcanza ni con muchísimas más epochs.

Para implementarlo vamos a usar la librería `segmentation_models_pytorch` (smp), que es el estándar de facto en PyTorch para arquitecturas de segmentación con encoders pre-entrenados. Su clase `smp.Unet` arma una U-Net cuyo encoder es una CNN clásica (ResNet, EfficientNet, etc.) cargada con pesos de ImageNet, y cuyo decoder replica la parte estándar de U-Net que ya implementaste a mano (con padding y BN).

> **Importante — la normalización ya está bien:** `PetSegDataset` aplica `transforms.Normalize` con la media/std de ImageNet, que es exactamente lo que esperan los encoders pre-entrenados de smp. **No hace falta tocar nada del dataset ni de los DataLoaders** — entran tal cual al modelo nuevo. Si hubiéramos normalizado con otras estadísticas, el transfer learning andaría peor porque las activaciones de las primeras capas verían distribuciones para las que el encoder no fue pre-entrenado.
::::


::::cell{#setup-cleanup-ft type=code role=setup}
```python
# ─── Limpieza antes del fine-tuning ─────────────────────────────────────────
# Vamos a tener dos modelos en GPU al mismo tiempo (el desde cero del Ej. 9
# y el fine-tuning del Ej. 11) porque después los comparamos en visualización.
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
     EJERCICIO 11 — Fine-tuning con smp.Unet
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej11-enunciado type=markdown role=enunciado}
### Ejercicio 11 — Fine-tuning de una U-Net con encoder pre-entrenado

**Objetivo:** Repetir el entrenamiento de la Sección D pero partiendo de una U-Net cuyo encoder está pre-entrenado sobre ImageNet, comparar cuantitativa y visualmente con la red entrenada desde cero.

**Enunciado:**

1. **Instalación de `segmentation_models_pytorch`.** En Colab no viene preinstalado: hay que correr `!pip install -q segmentation-models-pytorch`. La instalación toma unos segundos. Después importá la librería como `smp`.

2. **Modelo: `smp.Unet` con encoder ResNet34 pre-entrenado.** Pasale `encoder_name="resnet34"`, `encoder_weights="imagenet"`, `in_channels=3` y `classes=NUM_CLASSES`. Mandalo al `device`. La estructura interna es la de una U-Net (encoder-decoder con skip connections) similar a la que implementaste — la diferencia clave es que el encoder es un ResNet-34 con pesos pre-entrenados, en lugar de una secuencia de `DownConvolution` con pesos al azar.

3. **Loss y optimizador.** Reutilizá los pesos por clase del Ej. 9 (no hay que recalcularlos: el dataset es el mismo). Mismas hiperparámetros: cross-entropy ponderada con `ignore_index=IGNORE_INDEX`, Adam con `lr=1e-3`.

4. **Loop de entrenamiento por 5 epochs.** Mismo patrón que el Ej. 9 (train + eval por epoch, accuracy de pixel sobre píxeles válidos). Imprimí una línea por epoch y guardá el `val_acc` de cada una en `val_acc_ft_history`.

5. **Comparación cuantitativa.** Mostrá una tabla con `epoch | val_acc desde cero | val_acc fine-tuning`. La columna desde cero termina en epoch 4 (después queda en blanco) y la fine-tuning va hasta epoch 5. Usá un `pd.DataFrame` para que salga prolijo.

6. **Comparación visual.** Tomá las mismas 4 imágenes del split de test que usaste en el Ej. 10 (con `read_pet_images`) y armá una visualización en dos figuras de 2 filas cada una. Cada fila tiene cuatro columnas: imagen / predicción desde cero / predicción fine-tuning / ground truth. Reusá `label2image`, `add_seg_legend` y la lógica de centro-crop + normalización del Ej. 10.

> **Pista — el modelo de smp y el tuyo reciben/devuelven exactamente lo mismo:** un tensor `(B, 3, 256, 256)` normalizado entra, un tensor `(B, NUM_CLASSES, 256, 256)` de logits sale. Eso significa que el loop de entrenamiento del Ej. 9 funciona casi tal cual, solo cambiando la línea de instanciación del modelo.
>
> **Nota — qué esperar:** con 5 epochs de fine-tuning sobre Pet, `val_acc` debería superar **0.90 ya en la primera epoch** y converger en torno a **0.93-0.96** en la última. Esa es la magnitud del salto: un encoder ImageNet le ahorra a la red años de entrenamiento sobre features visuales generales y le permite enfocarse exclusivamente en la tarea de segmentación.
::::


::::cell{#ej11-code type=code role=student-code}
```python
# ─── 1) Instalación de segmentation_models_pytorch ──────────────────────────
# En Colab no viene preinstalado. Si ya lo tenés, este comando es no-op.

# Tu código aquí

# ─── 2) Modelo: smp.Unet con encoder ResNet34 pre-entrenado ────────────────

# Tu código aquí

# ─── 3) Loss y optimizador (reutilizamos los pesos por clase del Ej. 9) ────

# Tu código aquí

# ─── 4) Loop de entrenamiento (5 epochs, guardando val_acc_ft_history) ─────

# Tu código aquí

# ─── 5) Comparación cuantitativa: tabla val_acc desde cero vs fine-tuning ──

# Tu código aquí

# ─── 6) Comparación visual: imagen / desde cero / fine-tuning / GT ─────────

# Tu código aquí
```

```python solution
# ─── 1) Instalación de segmentation_models_pytorch ──────────────────────────
# La línea `!pip install ...` es magic de Jupyter: ejecuta el comando en el
# shell. La opción -q (quiet) reduce el output. Toma unos segundos en Colab.
!pip install -q segmentation-models-pytorch

import segmentation_models_pytorch as smp

# ─── 2) Modelo: smp.Unet con encoder ResNet34 pre-entrenado ────────────────
# encoder_name='resnet34': el encoder es un ResNet de 34 capas (He et al. 2015).
# encoder_weights='imagenet': pesos pre-entrenados sobre ImageNet (1.2M imágenes,
# 1000 clases). El decoder replica la arquitectura U-Net moderna (padding=1, BN)
# que vos implementaste a mano. Diferencia operativa: de dónde arrancan los
# pesos del encoder.
model_ft = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(device)

print(f"Parámetros del modelo fine-tuning: "
      f"{sum(p.numel() for p in model_ft.parameters()):,}")

# ─── 3) Loss y optimizador (reutilizamos los pesos por clase del Ej. 9) ────
# El dataset es el mismo, así que las frecuencias por clase no cambiaron y
# `weights` ya está calculado. Solo necesitamos un optimizer nuevo apuntando
# a los parámetros de model_ft.
criterion_ft = nn.CrossEntropyLoss(
    weight=weights.to(device), ignore_index=IGNORE_INDEX)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-3)

# ─── 4) Loop de entrenamiento (5 epochs) ───────────────────────────────────
num_epochs_ft = 5
val_acc_ft_history = []
for epoch in range(num_epochs_ft):
    # ── Train ───────────────────────────────────────────────────────────────
    model_ft.train()
    L_sum, n_correct, n_valid = 0.0, 0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device).long()
        y_hat = model_ft(X)
        loss = criterion_ft(y_hat, y)
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()

        with torch.no_grad():
            preds = y_hat.argmax(dim=1)
            mask = (y != IGNORE_INDEX)
            n_correct += ((preds == y) & mask).sum().item()
            n_valid   += mask.sum().item()
            L_sum     += loss.item() * X.size(0)

    train_loss = L_sum / len(pet_train)
    train_acc  = n_correct / n_valid

    # ── Val ─────────────────────────────────────────────────────────────────
    model_ft.eval()
    n_correct_v, n_valid_v = 0, 0
    with torch.no_grad():
        for X, y in val_iter:
            X, y = X.to(device), y.to(device).long()
            y_hat = model_ft(X)
            preds = y_hat.argmax(dim=1)
            mask = (y != IGNORE_INDEX)
            n_correct_v += ((preds == y) & mask).sum().item()
            n_valid_v   += mask.sum().item()
    val_acc = n_correct_v / n_valid_v
    val_acc_ft_history.append(val_acc)

    print(f"epoch {epoch + 1}/{num_epochs_ft}  "
          f"train_loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}")

# ─── 5) Comparación cuantitativa ────────────────────────────────────────────
# val_acc_history viene del Ej. 9 (4 valores). val_acc_ft_history acaba de
# generarse (5 valores). Mostramos las 5 epochs lado a lado, dejando "—"
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

# ─── 6) Comparación visual: imagen / desde cero / fine-tuning / GT ─────────
# Reusamos test_imgs/test_masks que ya leímos en el Ej. 10. Si por alguna
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
    msk_t = test_masks[idx].squeeze(0)  # (1, H, W) → (H, W)
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


::::cell{#ej11-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

¿Por qué el fine-tuning con un encoder pre-entrenado en ImageNet (un dataset de **clasificación** de 1000 clases como "perro labrador", "gato siamés", "auto", "edificio") transfiere tan bien a una tarea aparentemente distinta como **segmentación binaria de mascotas**? Pensá específicamente en qué está aprendiendo el encoder durante el pre-entrenamiento y por qué esos features son útiles también para decidir, píxel a píxel, si pertenece a una mascota o al fondo.
::::


::::cell{#ej11-respuesta type=markdown role=student-answer}
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
- [ ] Los tests de `SimpleConvolution`, `DownConvolution`, `UpConvolution`, `LastConvolution`, `crop_img` y `UNet` pasan sin errores.
- [ ] Los entrenamientos del Ej. 9 (prueba inicial, 4 epochs) y del Ej. 11 (fine-tuning, 5 epochs) corrieron sin OOM. Si tuve OOM, reinicié el entorno y volví a ejecutar.
- [ ] La prueba inicial del Ej. 9 supera claramente el ~70% de "predecir todo fondo" (esperable: ~0.78-0.85). El fine-tuning del Ej. 11 lo supera con holgura (esperable: >0.90).
- [ ] La grilla de visualización del Ej. 10 muestra al menos una imagen donde la silueta de la mascota es reconocible, aunque sea de forma rudimentaria.
- [ ] La comparación visual del Ej. 11 muestra una mejora clara de la columna fine-tuning respecto a la columna desde cero.
- [ ] Respondí las preguntas de análisis (Ej. 1, 2, 8, 9, 10, 11).
- [ ] No modifiqué ninguna celda fuera de las de actividad (ni las de test ni las de setup).
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Implementaste tu primera red de segmentación semántica de punta a punta. Practicaste:

- **Manejo del dataset Oxford-IIIT Pet** — formato trimap, mapeo a índices de clase, random crop consistente entre imagen y máscara, horizontal flip como augmentation, `ignore_index` para los bordes ruidosos.
- **Arquitectura U-Net moderna** — bloques de doble conv con `padding=1` y `BatchNorm`, downsampling con maxpool, upsampling con convolución transpuesta, skip connections que concatenan directamente (sin recortes, gracias al padding). Vimos también, a través del ejercicio de `crop_img`, qué hace la variante del paper original (sin padding) para alinear los skips.
- **Entrenamiento balanceado** — pesos por clase calculados como `1/√freq` (no `1/freq`, que produce pesos demasiado agresivos) para evitar el colapso a "todo fondo".
- **Predicción y visualización** — `label2image` para mapear índices a colores, comparación lado a lado con el ground truth.
- **Fine-tuning desde un encoder pre-entrenado** — usaste `segmentation_models_pytorch` con encoder ResNet34 inicializado con pesos de ImageNet. La diferencia respecto a entrenar desde cero quedó evidente: pocas epochs, mejor performance, contornos más finos.

Lo que hace que la U-Net siga vigente más de diez años después de su publicación es justamente lo que viste acá: una arquitectura **simple**, **simétrica** y **sin componentes exóticos**, que captura bien la receta general de las redes encoder-decoder con skip connections. Hoy se la sigue usando como punto de partida en imagen biomédica, satelital, microscopía y cualquier dominio donde la señal de entrenamiento sea escasa — casi siempre con un encoder pre-entrenado, como vimos en el ejercicio final.

Con esto cierra el bloque de **detección y segmentación** de la materia. Lo siguiente vamos a verlo en redes recurrentes y arquitecturas para datos secuenciales.
::::
