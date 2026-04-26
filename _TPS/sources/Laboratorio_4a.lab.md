---
lab: "4a"
title: "Laboratorio n° 4. Parte A: Detección de objetos"
subject: "Redes Neuronales Profundas"
block: "4 — Detección y segmentación"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 4a — Detección de objetos con YOLO
     Fuente única. Compilar con:
         python tools/lab_build.py _TPS/sources/Laboratorio_4a.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_4a.ipynb
             _TPS/Soluciones/Laboratorio_4a_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 4. Parte A: Detección de objetos

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 4 — Detección y segmentación

---

## Introducción

Hasta ahora trabajamos con redes que **clasifican** una imagen entera (¿es un perro o un gato?) o que **transforman** una imagen entera (denoising, transferencia de estilo). En este laboratorio abordamos una tarea distinta: **detección de objetos**, donde el modelo debe responder dos preguntas a la vez por cada objeto de la imagen:

1. **¿Qué es?** (clasificación)
2. **¿Dónde está?** (localización, mediante una caja delimitadora o *bounding box*)

### El gap entre teoría y práctica

La teoría de detección moderna es densa: anchors, *region proposals*, *non-maximum suppression*, métricas como mAP e IoU, regresión de cajas con pérdidas tipo CIoU, arquitecturas como Faster R-CNN, RetinaNet, FCOS, YOLO, DETR, RT-DETR... Cubrir todo esto en profundidad consumiría varias clases. **En clase vimos los fundamentos conceptuales**; en este laboratorio vamos a **usar herramientas actuales** sin reimplementar internamente cómo funcionan.

La idea es que se lleven una experiencia práctica con los modelos que se usan hoy en la industria, entendiendo **qué hacen** y **cuándo conviene cada uno**, aunque no derivemos la matemática interna de cada arquitectura. Para que esto no quede como una caja negra total, antes de cada arquitectura nueva hay una sección breve que explica cómo funciona en términos generales y qué es importante tener en cuenta.

### Vocabulario mínimo

Términos que vamos a usar repetidamente:

- **Bounding box (bbox):** caja rectangular que rodea un objeto, definida por sus coordenadas (típicamente `(x_centro, y_centro, ancho, alto)` o las dos esquinas opuestas).
- **IoU (Intersection over Union):** métrica entre dos cajas que mide cuánto se superponen. `IoU = área_intersección / área_unión`. Vale entre 0 (no se tocan) y 1 (idénticas).
- **NMS (Non-Maximum Suppression):** algoritmo de post-procesamiento que elimina cajas duplicadas. Si dos detecciones superpuestas refieren al mismo objeto (IoU alta), se queda con la de mayor confianza y descarta el resto.
- **Confianza (confidence threshold):** un detector predice cientos de cajas con un score asociado; típicamente se filtran las que están por debajo de un umbral (ej. 0.25).
- **mAP (mean Average Precision):** métrica estándar para comparar detectores. Combina precisión y recall a varios umbrales de IoU. Más alto es mejor; `mAP50` evalúa con IoU≥0.5, `mAP50-95` promedia varios IoU.

### Panorama de detectores en 2026

En el campo conviven cuatro familias principales. Es importante saber qué hay porque la elección depende del problema:

| Familia | Ejemplos | Fortaleza | Cuándo conviene |
|---|---|---|---|
| **YOLO** (one-stage) | YOLOv8, YOLOv11 (Ultralytics) | Velocidad, fine-tuning trivial, ecosistema maduro | Tiempo real, edge devices, proyectos donde "que ande rápido" importa más que el último 1% de mAP |
| **Transformer-based** | DETR, RT-DETR, DINO | Sin anchors, sin NMS, formulación más limpia | Cuando la flexibilidad del transformer aporta (relaciones entre objetos, escenas complejas) |
| **Two-stage clásicos** | Faster R-CNN, Mask R-CNN (torchvision) | Estabilidad, transparencia pedagógica | Pipelines legacy, segmentación de instancias |
| **Open-vocabulary / zero-shot** | Grounding DINO, **YOLO-World**, OWL-ViT | Detectar clases descritas en texto sin entrenar | Prototipado, datasets sin etiquetar, clases que cambian |

**En este laboratorio usamos YOLO (Ultralytics) como herramienta principal** — es el estándar de facto para detección práctica hoy — y al final tocamos zero-shot con YOLO-World para ver hacia dónde va el campo.

> **Nota sobre licencia:** la implementación de Ultralytics está bajo **AGPL-3.0**. Para uso educativo y prototipado no hay problema. Si más adelante usás YOLO en un producto comercial cerrado, vas a tener que evaluar licenciamiento (Ultralytics ofrece licencia comercial paga, o existen forks bajo licencias permisivas).

> **Importante — GPU:** este laboratorio fine-tunea un modelo sobre cientos de imágenes. **Activá la GPU en Colab** antes de empezar: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4)*. Sin GPU el entrenamiento es impracticable (horas en lugar de minutos).

---

## Instrucciones generales

- Completá el código en las celdas marcadas con `# Tu código aquí`.
- Respondé las preguntas de análisis en las celdas de texto (tipo Markdown).
- Para el vocabulario teórico (IoU, mAP, NMS, anchors) consultá el material de la clase.
- A diferencia de laboratorios previos, acá usamos un framework de alto nivel (`ultralytics`) que oculta muchos detalles. La pregunta no va a ser "¿cómo se implementa?" sino "¿qué efecto tiene cambiar este parámetro?".
::::


::::cell{#reglas type=markdown role=rules}
## IMPORTANTE: qué celdas podés modificar

Este laboratorio es un **entregable**. Solo debés completar las celdas de actividad, que son las que aparecen con el comentario `# Tu código aquí` o el texto `*(Escribí tu respuesta acá)*`. Todas las demás celdas (enunciados, explicaciones, ejemplos provistos y el encabezado) **no se tocan**: la corrección se hace celda por celda de manera automática y modificar lo que no corresponde puede invalidar tu entrega.

Si necesitás probar algo fuera de una celda de actividad, hacelo en una copia aparte y revertí los cambios antes de entregar.
::::


::::cell{#imports type=code role=setup}
```python
# ─── Setup: instalación e imports ───────────────────────────────────────────
# Ultralytics no viene preinstalado en Colab; lo instalamos primero.
# Trae YOLO (detección, segmentación, pose) y todas sus dependencias.
!pip install -q ultralytics

import os
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from ultralytics import YOLO

device = (
    "cuda" if torch.cuda.is_available()        else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Dispositivo:        {device}")
if device == "cpu":
    print("ADVERTENCIA: sin GPU el fine-tuning va a ser inviable. "
          "Activá la GPU en Colab (T4) antes de continuar.")
```
::::


::::cell{#setup-data type=code role=setup}
```python
# ─── Setup: descarga y preparación del dataset BCCD ─────────────────────────
# BCCD (Blood Cell Count and Detection) es un dataset clínico de microscopía
# con 3 clases de células: glóbulos rojos (RBC), glóbulos blancos (WBC) y
# plaquetas (Platelets). Es chico (~360 imágenes) y la aplicación natural
# -- contar glóbulos -- es exactamente lo que hace un hemograma en un
# laboratorio de análisis clínicos.
#
# El zip ya está armado en formato YOLO estándar:
#   BCCD/images/{train,val,test}/  con .jpg
#   BCCD/labels/{train,val,test}/  con .txt (una línea por caja, formato YOLO)
DATA_URL  = "https://github.com/javovelez/tps_RNP/raw/main/BCCD_yolo.zip"
DATA_ROOT = Path("/content/BCCD")

if not DATA_ROOT.exists():
    !wget -q --show-progress {DATA_URL} -O /content/BCCD.zip
    !unzip -q /content/BCCD.zip -d /content/
    !rm /content/BCCD.zip
else:
    print("Dataset ya descargado.")

# ─── Generamos data.yaml programáticamente ─────────────────────────────────
# Ultralytics necesita un YAML que describa rutas y clases del dataset. Lo
# escribimos acá en lugar de depender de uno embebido en el zip: si cambia
# la ruta de Colab, basta con re-ejecutar esta celda.
DATA_YAML = DATA_ROOT / "data.yaml"
DATA_YAML.write_text(f"""\
path: {DATA_ROOT}
train: images/train
val:   images/val
test:  images/test
nc: 3
names: ['Platelets', 'RBC', 'WBC']
""")
print(f"data.yaml escrito en: {DATA_YAML}")

# ─── Visualizar una imagen de muestra con sus cajas ────────────────────────
class_names  = ['Platelets', 'RBC', 'WBC']
class_colors = ['red', 'blue', 'green']

sample_img = sorted((DATA_ROOT / "images" / "train").iterdir())[0]
sample_lbl = (DATA_ROOT / "labels" / "train" / sample_img.stem).with_suffix(".txt")
img = np.array(Image.open(sample_img))
H, W = img.shape[:2]

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img)
with open(sample_lbl) as f:
    for line in f:
        cls, xc, yc, w, h = map(float, line.split())
        x = (xc - w/2) * W
        y = (yc - h/2) * H
        ax.add_patch(plt.Rectangle((x, y), w*W, h*H,
                                    fill=False,
                                    edgecolor=class_colors[int(cls)],
                                    linewidth=2))
ax.set_title(f"Muestra del dataset BCCD: {sample_img.name}")
ax.axis('off')
plt.show()

# ─── Stats por split ───────────────────────────────────────────────────────
for split in ['train', 'val', 'test']:
    n = len(list((DATA_ROOT / 'images' / split).iterdir()))
    print(f"{split:5s}: {n} imágenes")
```
::::


::::cell{#secA type=markdown role=section}
---
## Sección A: Inferencia con YOLO preentrenado

Antes de fine-tunear nada, vamos a usar el modelo "tal cual viene" para detectar objetos en imágenes arbitrarias. Esto cumple dos roles: por un lado, ver lo impresionante que es un detector moderno preentrenado; por otro, familiarizarnos con la API de `ultralytics` que después usamos para entrenar.
::::


::::cell{#secA-yolo type=markdown role=scaffolding}
### ¿Qué es YOLO y por qué lo usamos?

**YOLO** = "You Only Look Once". Es una familia de detectores propuesta originalmente por Joseph Redmon en 2016 y que evolucionó (con cambios de equipo y arquitectura) hasta las versiones actuales mantenidas por **Ultralytics** (YOLOv8 en 2023, YOLOv11 en 2024).

#### Cómo funciona en términos generales

A diferencia de los detectores **two-stage** (tipo Faster R-CNN), que primero proponen regiones candidatas y después las clasifican, YOLO es **one-stage**: en una sola pasada por la red predice *todas* las cajas y sus clases. De ahí "you only look once" — y de ahí su velocidad.

A grandes rasgos:

1. La imagen se procesa con un **backbone convolucional** (similar a una red de clasificación).
2. Las features se pasan por un **neck** (FPN/PAN) que combina información de varias escalas — necesario para detectar tanto objetos grandes como chicos.
3. La **head** de detección produce, en cada posición de un mapa de features, una predicción de:
   - Si hay un objeto allí (*objectness*),
   - Qué clase es (probabilidades sobre las N clases del dataset),
   - El offset y tamaño de la caja respecto a esa posición.
4. **NMS** se aplica como post-procesamiento para eliminar cajas duplicadas.

Las versiones modernas (v8+) son **anchor-free**: predicen las cajas directamente desde cada posición sin usar anchors prefijados. Esto simplifica el diseño respecto de versiones anteriores.

#### Ventajas

- **Velocidad:** YOLOv11n (la variante "nano") procesa cientos de FPS en GPU; corre en tiempo real incluso en hardware modesto.
- **API muy simple:** entrenar es `model.train(data='dataset.yaml', epochs=20)`. Pocas líneas, mucha funcionalidad.
- **Ecosistema maduro:** documentación, tutoriales, integraciones con Roboflow, exportación a ONNX/TensorRT/CoreML.
- **Versátil:** la misma familia tiene variantes para detección, segmentación, pose estimation y clasificación.

#### Desventajas / cosas a tener en cuenta

- **API de alto nivel:** muchos detalles internos (loss, asignación de targets, augmentation) están encapsulados. Bueno para productividad, malo si querés modificar la arquitectura a fondo.
- **Licencia AGPL-3.0:** restrictiva para uso comercial cerrado.
- **Evolución rápida:** el código y los pesos cambian entre versiones; tutoriales viejos pueden dejar de funcionar.
- **Tamaño del modelo:** las variantes `n` (nano) y `s` (small) son rápidas pero menos precisas que `m`, `l`, `x`. Hay un trade-off velocidad/accuracy que conviene tener presente al elegir variante.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 1 — Inferencia básica
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Primera inferencia con YOLO preentrenado

**Objetivo:** Familiarizarte con la API de `ultralytics` para cargar un modelo preentrenado y correr inferencia sobre una imagen.

**Enunciado:**

1. Cargá el modelo preentrenado `yolo11n.pt` (la variante nano de YOLOv11). La primera vez `ultralytics` descarga los pesos automáticamente.
2. Corré inferencia sobre la imagen de ejemplo `https://ultralytics.com/images/bus.jpg`. El modelo acepta URLs directamente.
3. Visualizá el resultado con las cajas dibujadas usando el método `.plot()` del objeto `Result`. Tené en cuenta que `.plot()` devuelve la imagen en formato BGR (orden de canales de OpenCV), así que si la mostrás con `matplotlib` hay que invertir los canales.
4. Imprimí cada detección con su clase (nombre, no índice) y su nivel de confianza.

> **Pista:** `model(...)` devuelve una lista de `Result` (una entrada por imagen procesada). Cada `Result` tiene `.boxes` (las detecciones), `.names` (diccionario `{índice: nombre}` de clases del modelo) y el método `.plot()`. Cada caja en `.boxes` tiene `.cls` (índice de clase) y `.conf` (confianza), ambos como tensores de un solo elemento.
::::


::::cell{#ej1-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Cargar el modelo preentrenado en COCO ──────────────────────────────────
# yolo11n.pt es la variante "nano" de YOLOv11, entrenada sobre COCO (80 clases
# de objetos cotidianos: persona, auto, bus, perro, etc.). Ultralytics descarga
# los pesos automáticamente la primera vez.
model = YOLO('yolo11n.pt')

# ─── Inferencia ─────────────────────────────────────────────────────────────
# El modelo acepta una URL, un path local, un numpy array o un tensor.
# Devuelve una lista de Results (una entrada por imagen).
img_url = 'https://ultralytics.com/images/bus.jpg'
results = model(img_url)
res = results[0]  # nuestra única imagen

# ─── Visualización ──────────────────────────────────────────────────────────
# .plot() devuelve la imagen con las cajas dibujadas en formato BGR (cv2).
# Para mostrarla con matplotlib invertimos el orden de canales.
plt.figure(figsize=(10, 8))
plt.imshow(res.plot()[:, :, ::-1])
plt.axis('off')
plt.title('YOLOv11n preentrenado en COCO — bus.jpg')
plt.show()

# ─── Listar las detecciones con confianza ──────────────────────────────────
print(f"Detecciones: {len(res.boxes)}")
for box in res.boxes:
    cls_id   = int(box.cls[0])
    cls_name = res.names[cls_id]
    conf     = float(box.conf[0])
    print(f"  {cls_name:15s}  confianza={conf:.2f}")
```
::::


::::cell{#ej1-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirá las clases que detectó el modelo. ¿Por qué no detecta "ventana", "rueda" o "espejo retrovisor", aunque claramente están en la imagen?
::::


::::cell{#ej1-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

El modelo está entrenado sobre **COCO**, que tiene exactamente 80 clases predefinidas (persona, auto, bus, bicicleta, semáforo, etc.). El modelo solo puede predecir clases que vio durante el entrenamiento: el conjunto de salidas posibles está fijado por la arquitectura de la cabeza de clasificación, que tiene 80 neuronas — una por clase.

Una "ventana", una "rueda" o un "espejo" no están en ese vocabulario, así que aunque visualmente estén ahí, el modelo no las "ve" como categoría: no tiene una salida posible que las represente. Esta limitación es una propiedad fundamental de los detectores supervisados de vocabulario cerrado, y es justamente lo que motivan los modelos de vocabulario abierto (zero-shot) que vamos a ver al final del laboratorio.
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 2 — Confianza e IoU del NMS
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Confianza y NMS: efectos del threshold

**Objetivo:** Ver de manera práctica qué controlan el threshold de confianza y el threshold IoU del NMS.

**Enunciado:**

1. Sobre la misma imagen `https://ultralytics.com/images/bus.jpg`, hacé dos inferencias variando el threshold de confianza:
   - `conf=0.10` (laxo: deja pasar muchas cajas)
   - `conf=0.70` (estricto: solo cajas muy seguras)
   Mostrá las dos imágenes lado a lado y reportá cuántas detecciones hay en cada caso.
2. Hacé lo mismo variando el threshold IoU del NMS:
   - `iou=0.30` (estricto: descarta agresivamente cajas superpuestas)
   - `iou=0.80` (laxo: solo descarta si las cajas casi se superponen totalmente)
   Mostrá las dos imágenes lado a lado y reportá la cantidad de cajas en cada caso.

> **Pista:** Tanto `conf` como `iou` se pasan como argumentos al llamar a `model(...)`. Por ejemplo: `model(img_url, conf=0.5, iou=0.5)`. Para visualizar dos imágenes lado a lado usá `plt.subplots(1, 2, figsize=(...))`.
::::


::::cell{#ej2-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
img_url = 'https://ultralytics.com/images/bus.jpg'

# ─── Threshold de confianza ─────────────────────────────────────────────────
# El detector predice cientos de cajas con scores asociados. `conf` es el
# umbral por debajo del cual descartamos detecciones. Bajarlo aumenta recall
# (no perdemos objetos) a costa de precisión (más falsos positivos). Subirlo
# hace lo contrario.
res_low  = model(img_url, conf=0.10, verbose=False)[0]
res_high = model(img_url, conf=0.70, verbose=False)[0]

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs[0].imshow(res_low.plot()[:, :, ::-1])
axs[0].set_title(f'conf=0.10 — {len(res_low.boxes)} detecciones')
axs[0].axis('off')
axs[1].imshow(res_high.plot()[:, :, ::-1])
axs[1].set_title(f'conf=0.70 — {len(res_high.boxes)} detecciones')
axs[1].axis('off')
plt.tight_layout()
plt.show()

# ─── Threshold IoU del NMS ─────────────────────────────────────────────────
# NMS recorre las cajas de mayor a menor confianza y elimina aquellas que
# tengan IoU > umbral con una caja ya seleccionada. `iou` bajo es estricto
# (suprime más); `iou` alto es laxo (suprime menos).
res_strict = model(img_url, iou=0.30, verbose=False)[0]
res_lax    = model(img_url, iou=0.80, verbose=False)[0]

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs[0].imshow(res_strict.plot()[:, :, ::-1])
axs[0].set_title(f'NMS iou=0.30 (estricto) — {len(res_strict.boxes)} cajas')
axs[0].axis('off')
axs[1].imshow(res_lax.plot()[:, :, ::-1])
axs[1].set_title(f'NMS iou=0.80 (laxo) — {len(res_lax.boxes)} cajas')
axs[1].axis('off')
plt.tight_layout()
plt.show()
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

a) Pensá en dos aplicaciones reales: una donde te conviene **bajar** el threshold de confianza y otra donde te conviene **subirlo**. Justificá brevemente.

b) ¿Qué pasaría si pusieras `iou=1.0` en NMS? ¿Y si pusieras `iou=0.0`?
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

**a)** El threshold de confianza determina el balance recall/precisión:

- **Bajarlo (ej. 0.1):** cuando perder un objeto es peor que tener falsos positivos. Por ejemplo, **detección de defectos en una línea de fabricación** donde un humano (o un segundo modelo) revisa después: preferimos que aparezcan algunas piezas falsamente marcadas a que se nos escape una con defecto real. Otro ejemplo: detección de tumores en imágenes médicas como filtro inicial para revisión radiológica.
- **Subirlo (ej. 0.7):** cuando un falso positivo tiene un costo real. Por ejemplo, un **auto autónomo decidiendo frenar** ante un "obstáculo" — un falso positivo causa frenadas bruscas innecesarias y compromete la conducción. Otro ejemplo: detección automática que dispara alertas a un operador humano (si hay muchos falsos positivos, el operador deja de prestar atención).

**b)** El threshold IoU del NMS controla cuán "agresivo" es eliminando duplicados:

- Con `iou=1.0`, NMS solo suprime cajas con IoU **estrictamente mayor** a 1.0, lo cual es imposible. En la práctica, **ninguna caja se descarta**: vemos las cajas crudas del modelo, con muchísimas duplicaciones sobre el mismo objeto.
- Con `iou=0.0`, NMS suprime cualquier caja que se solape **aunque sea mínimamente** con otra de mayor confianza. Esto descarta detecciones de objetos genuinamente cercanos (por ejemplo, dos personas paradas juntas), porque sus cajas se tocan aunque sean objetos distintos. El resultado es muchos falsos negativos en escenas con objetos próximos.
```
::::


::::cell{#secB type=markdown role=section}
---
## Sección B: Fine-tuning sobre BCCD

Hasta acá usamos el modelo "tal cual viene": entrenado en COCO, sirve para detectar las 80 clases de COCO. Pero la mayoría de las aplicaciones reales requieren detectar objetos que no están en COCO. En esta sección vamos a tomar el mismo modelo y **fine-tunearlo** sobre el dataset BCCD para que aprenda a detectar tres tipos de células sanguíneas.
::::


::::cell{#secB-tl type=markdown role=scaffolding}
### Transfer learning aplicado a detección

La idea de transfer learning ya la vimos en clasificación: un modelo entrenado en una tarea grande (ImageNet) tiene un backbone con representaciones útiles que sirven como punto de partida para una tarea más chica.

En detección la lógica es la misma, pero con un detalle adicional: la **cabeza de detección** depende del número de clases del dataset, así que **se reemplaza** al cambiar de tarea. Lo que se preserva es:

- El **backbone** (extractor de features convolucional): aprendió a representar visualmente objetos en COCO. Esas representaciones siguen siendo útiles para BCCD.
- El **neck** (FPN/PAN, combinación multi-escala): la lógica de fusionar features a distintas escalas es independiente de las clases.

Lo que se reinicializa:

- La **cabeza de clasificación** dentro del head (de 80 clases a 3 en nuestro caso).
- Los pesos asociados específicamente a las predicciones de clase.

Ultralytics maneja esto automáticamente cuando le pasás un dataset con un número de clases distinto al original. No hace falta tocar nada manualmente: lee el `data.yaml`, descubre el `nc`, reconstruye la cabeza y entrena end-to-end.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 3 — Modelo COCO sobre BCCD: el dominio importa
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — Probando el modelo COCO sobre microscopía

**Objetivo:** Ver de primera mano por qué un modelo preentrenado no es suficiente para una tarea nueva, antes de hacer el fine-tuning.

**Enunciado:**

1. Tomá la primera imagen del split de **test** del dataset BCCD (`DATA_ROOT / 'images' / 'test'`).
2. Corré inferencia sobre esa imagen con el modelo `yolo11n.pt` preentrenado en COCO (el mismo del Ej. 1).
3. Visualizá el resultado con `.plot()` e imprimí las detecciones (clase + confianza). Bajá el threshold a `conf=0.05` para que aparezca todo lo que el modelo "cree ver".

> **Pista:** El path de la imagen de test podés obtenerlo con `sorted((DATA_ROOT / 'images' / 'test').iterdir())[0]`. Pasalo a `model(...)` como string (`str(...)`).
::::


::::cell{#ej3-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Inferencia con el modelo COCO sobre una imagen BCCD ───────────────────
# El modelo nunca vio imágenes de microscopía durante el entrenamiento, y
# sus 80 clases (persona, auto, perro, etc.) no incluyen tipos de células.
# Bajamos el threshold a 0.05 para ver todo lo que el modelo "cree ver".
sample_path = sorted((DATA_ROOT / 'images' / 'test').iterdir())[0]

model_coco = YOLO('yolo11n.pt')
res = model_coco(str(sample_path), conf=0.05, verbose=False)[0]

plt.figure(figsize=(10, 8))
plt.imshow(res.plot()[:, :, ::-1])
plt.axis('off')
plt.title(f'YOLO COCO sobre microscopía — {sample_path.name}')
plt.show()

print(f"Detecciones: {len(res.boxes)}")
for box in res.boxes:
    cls_id   = int(box.cls[0])
    cls_name = res.names[cls_id]
    conf     = float(box.conf[0])
    print(f"  {cls_name:15s}  conf={conf:.2f}")
```
::::


::::cell{#ej3-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

El modelo prácticamente no detecta nada útil (o detecta cosas absurdas con confianza baja). Suponé que los autores de COCO hubieran incluido la clase `"cell"` (célula) en el conjunto de clases. ¿Eso alcanzaría para resolver nuestra tarea? ¿Por qué sí o por qué no?
::::


::::cell{#ej3-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

No alcanzaría, por dos razones independientes:

1. **Granularidad insuficiente.** Nuestra tarea no es "detectar células" en abstracto; es **distinguir tres tipos** específicos (RBC, WBC, Platelets). Una sola clase `"cell"` no permitiría diferenciarlos. Para que el modelo sirviera, COCO tendría que tener cada tipo como clase aparte, y aún así sería poco probable que coincida con la convención clínica que necesitamos.

2. **Domain gap visual.** Aunque la clase existiera, el modelo no habría visto **imágenes de microscopía teñida con Wright-Giemsa** (la técnica que produce los colores violáceos típicos de BCCD). Las features que aprendió el backbone están sesgadas hacia escenas naturales: fotos cotidianas con iluminación variable, fondos heterogéneos, objetos sólidos. Las texturas, colores y contrastes de la microscopía son un dominio distinto, y el modelo se desempeñaría mal aunque la clase nominal coincidiera.

Estas dos razones (granularidad de clases + domain gap) son exactamente lo que el fine-tuning resuelve: ajustamos la cabeza de detección al esquema de clases que necesitamos, y dejamos que el modelo readapte sus features al nuevo dominio visual entrenando con datos del dominio objetivo.
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 4 — Fine-tuning (sin pregunta de análisis)
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — Fine-tuning sobre BCCD

**Objetivo:** Ejecutar el fine-tuning del modelo preentrenado sobre el dataset BCCD.

**Enunciado:**

1. Instanciá un modelo nuevo a partir de `yolo11n.pt` (no usar `model_coco` del ejercicio anterior por si quedaron cachés de inferencia con thresholds modificados).
2. Llamá a `model.train(...)` con los siguientes argumentos:
   - `data` = path al `data.yaml` que generamos en el setup (recordá que hay que pasarlo como string).
   - `epochs=20`
   - `imgsz=640` (tamaño al que se resizean las imágenes para entrenar)
   - `batch=16`
   - `name='bccd_finetune'` (nombre de la corrida; se guarda en `runs/detect/bccd_finetune/`)
3. Esperá a que termine (~5-10 min en T4). Al final, los pesos del mejor checkpoint quedan en `runs/detect/bccd_finetune/weights/best.pt`.

> **Pista:** Mientras entrena, ultralytics imprime una tabla por epoch con las pérdidas (box, cls, dfl) y métricas (mAP50, mAP50-95) sobre el split de validación. Es normal que las pérdidas oscilen un poco; lo importante es que el mAP50 vaya subiendo.

> **Nota:** Si en algún momento querés re-entrenar desde cero, borrá la carpeta `runs/detect/bccd_finetune` antes (o cambiá el `name`).
::::


::::cell{#ej4-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Fine-tuning sobre BCCD ─────────────────────────────────────────────────
# Partimos del modelo COCO preentrenado. Ultralytics, internamente:
#   1. Lee data.yaml y descubre que el dataset tiene 3 clases (no 80).
#   2. Reemplaza la cabeza de detección para 3 clases, manteniendo backbone y neck.
#   3. Inicializa el resto de los pesos con los del modelo COCO (transfer learning).
#   4. Entrena end-to-end por la cantidad de epochs indicada.
model = YOLO('yolo11n.pt')

results = model.train(
    data=str(DATA_YAML),
    epochs=20,
    imgsz=640,
    batch=16,
    name='bccd_finetune',
)
# Los pesos del mejor checkpoint (según mAP50-95 en val) quedan en:
#   runs/detect/bccd_finetune/weights/best.pt
```
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 5 — Evaluación + conteo
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej5-enunciado type=markdown role=enunciado}
### Ejercicio 5 — Evaluación y conteo por clase

**Objetivo:** Medir la performance del modelo fine-tuneado y aplicarlo a la tarea concreta de **contar células por clase** comparando con el ground truth.

**Enunciado:**

1. Cargá el mejor checkpoint del fine-tuning desde `runs/detect/bccd_finetune/weights/best.pt`.
2. Evaluá el modelo en el split de **test** con `best_model.val(data=..., split='test')`. Imprimí `mAP50` y `mAP50-95` globales y `mAP50` por clase (`metrics.box.maps` devuelve un array indexado por clase).
3. Para las primeras 3 imágenes del split de test, hacé:
   - Predicción con `best_model(...)` (usá `conf=0.25`, `verbose=False`).
   - Conteo de detecciones por clase a partir de `res.boxes`.
   - Conteo de cajas en el ground truth leyendo el archivo `.txt` de etiquetas correspondiente.
   - Imprimí una tabla por imagen comparando predicción vs GT.
   - Mostrá la imagen con las cajas predichas usando `.plot()`.

> **Pista 1:** El nombre del archivo de etiquetas es el mismo que el de la imagen pero con extensión `.txt` y ubicado en `labels/test/` en lugar de `images/test/`. Podés construirlo con `(DATA_ROOT / 'labels' / 'test' / sample_img.stem).with_suffix('.txt')`.

> **Pista 2:** Cada línea del archivo de etiquetas YOLO tiene la forma `clase_idx x_centro y_centro ancho alto` (las últimas cuatro normalizadas). Para contar instancias por clase basta con leer el primer número de cada línea.
::::


::::cell{#ej5-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Cargar el mejor modelo entrenado ───────────────────────────────────────
best_model = YOLO('runs/detect/bccd_finetune/weights/best.pt')

# ─── Evaluación global y por clase en el split de test ─────────────────────
metrics = best_model.val(data=str(DATA_YAML), split='test', verbose=False)

print(f"mAP50 (global):    {metrics.box.map50:.3f}")
print(f"mAP50-95 (global): {metrics.box.map:.3f}")
print("\nmAP50 por clase:")
for i, name in enumerate(class_names):
    # metrics.box.maps es mAP50-95 por clase; metrics.box.ap50 es mAP50 por clase.
    print(f"  {name:12s}: mAP50={metrics.box.ap50[i]:.3f}  mAP50-95={metrics.box.maps[i]:.3f}")

# ─── Conteo predicción vs ground truth en 3 imágenes de test ───────────────
test_imgs = sorted((DATA_ROOT / 'images' / 'test').iterdir())[:3]

for img_path in test_imgs:
    res = best_model(str(img_path), conf=0.25, verbose=False)[0]

    # Conteo a partir de las cajas predichas
    pred_counts = {n: 0 for n in class_names}
    for box in res.boxes:
        pred_counts[class_names[int(box.cls[0])]] += 1

    # Conteo del ground truth a partir del archivo de etiquetas
    lbl_path = (DATA_ROOT / 'labels' / 'test' / img_path.stem).with_suffix('.txt')
    gt_counts = {n: 0 for n in class_names}
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                cls = int(line.split()[0])
                gt_counts[class_names[cls]] += 1

    # Tabla
    print(f"\n--- {img_path.name} ---")
    print(f"  {'clase':12s}  {'pred':>6s}  {'GT':>6s}")
    for n in class_names:
        print(f"  {n:12s}  {pred_counts[n]:>6d}  {gt_counts[n]:>6d}")

    # Visualización de la predicción
    plt.figure(figsize=(8, 6))
    plt.imshow(res.plot()[:, :, ::-1])
    plt.axis('off')
    plt.title(img_path.name)
    plt.show()
```
::::


::::cell{#ej5-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirá los `mAP50` por clase. ¿Hay alguna clase donde el modelo funciona claramente peor que en las otras? Proponé al menos dos razones plausibles y, para cada una, qué cambio harías en el pipeline (datos, hiperparámetros, modelo) para mitigarla.
::::


::::cell{#ej5-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Típicamente, en BCCD la clase con peor mAP es **Platelets** (plaquetas), seguida por **WBC** (glóbulos blancos), siendo **RBC** la mejor. Las razones probables y mitigaciones posibles:

1. **Desbalance de clases.** En BCCD hay muchísimas más instancias de RBC que de Platelets (cada imagen tiene decenas de RBC y unas pocas plaquetas). El modelo "ve" mucho más de unas que de otras durante el entrenamiento.
   - *Mitigación:* aplicar **class weights** en la pérdida, hacer **oversampling** de las imágenes que contienen plaquetas, o usar augmentation focalizada (rotaciones/flips que multipliquen las muestras de la clase minoritaria).

2. **Tamaño del objeto.** Las plaquetas son objetos chicos (en píxeles). La detección de objetos pequeños es un problema general en YOLO: a `imgsz=640`, una plaqueta puede ocupar pocos píxeles de feature map y resulta difícil de localizar.
   - *Mitigación:* aumentar `imgsz` (por ejemplo a 1024) para darle más resolución al modelo, o usar una variante más grande (`yolo11s` o `yolo11m`) que tiene un FPN/PAN con mayor capacidad para escalas chicas.

3. **Calidad de anotación.** En datasets como BCCD las plaquetas suelen estar anotadas de forma menos consistente que las otras clases (a veces no se anotan las que están en el borde o desenfocadas).
   - *Mitigación:* revisar manualmente el ground truth (auditoría de etiquetas) o, si el dataset es muy ruidoso, usar técnicas de label smoothing / loss robustas al ruido.

4. **Pocas epochs.** 20 epochs sobre un dataset chico puede no ser suficiente para que la clase minoritaria converja del todo.
   - *Mitigación:* entrenar más epochs (por ejemplo 50-100) y usar early stopping basado en mAP50-95 de validación.
```
::::


::::cell{#secC type=markdown role=section}
---
## Sección C: Detección zero-shot — qué viene después

Hasta acá hicimos lo que hace la mayoría de los proyectos de detección hoy: tomamos un modelo pre-entrenado y lo fine-tuneamos sobre nuestro dataset etiquetado. Pero hay una línea de modelos más recientes que cambia las reglas del juego: detectores **open-vocabulary** que aceptan las clases como **texto** en tiempo de inferencia, sin necesidad de fine-tuning.
::::


::::cell{#secC-zeroshot type=markdown role=scaffolding}
### Detección open-vocabulary

Modelos como **Grounding DINO**, **OWL-ViT** y **YOLO-World** combinan un backbone visual con un encoder de texto (estilo CLIP). Durante el entrenamiento se les muestran muchísimos pares imagen + descripciones textuales, y aprenden a alinear regiones de la imagen con descripciones en lenguaje natural.

El resultado: en inferencia, podés pasarle una lista de clases como prompt — `["dog", "person wearing red shirt", "blue car"]` — y el modelo detecta esas clases sin haberlas visto explícitamente etiquetadas durante el entrenamiento.

**Cuándo conviene zero-shot:**

- **Prototipado rápido:** querés una demo en un día, sin tiempo de etiquetar nada.
- **Clases que cambian:** un cliente que hoy quiere detectar "botellas" y mañana quiere "frascos de mermelada"; con un modelo zero-shot basta con cambiar el prompt.
- **Clases raras:** la clase de interés tiene tan pocos ejemplos en el mundo que reunir un dataset etiquetado es prohibitivo.
- **Búsqueda visual abierta:** "encontrar todas las imágenes que tengan X" sobre un corpus, donde X es definido por el usuario.

**Cuándo NO conviene:**

- **Producción estable** con clases fijas y necesidad de máximo accuracy: un modelo fine-tuneado sobre un dataset razonable suele superar a uno zero-shot.
- **Dominios visuales muy distintos** del dato de entrenamiento del modelo zero-shot (microscopía especializada, imágenes médicas específicas, sensores no-RGB): el modelo zero-shot tampoco vio esos dominios durante su pre-entrenamiento, así que no compensa el domain gap.

En este laboratorio probamos **YOLO-World**, que es la integración de open-vocabulary detection dentro del paquete `ultralytics`. Es una variante del backbone YOLO con un encoder de texto adicional y una cabeza de detección que produce embeddings en el mismo espacio que el texto.
::::


<!-- ──────────────────────────────────────────────────────────────────────
     EJERCICIO 6 — YOLO-World + pregunta de análisis final
     ────────────────────────────────────────────────────────────────────── -->

::::cell{#ej6-enunciado type=markdown role=enunciado}
### Ejercicio 6 — Detección zero-shot con YOLO-World

**Objetivo:** Ver de primera mano cómo funciona la detección open-vocabulary y comparar el resultado con un modelo de vocabulario cerrado.

**Enunciado:**

1. Importá `YOLOWorld` desde `ultralytics` y cargá el modelo `yolov8s-world.pt`.
2. Definí las clases mediante texto usando `world_model.set_classes([...])`. Probá con clases que **no estén en COCO**, por ejemplo: `["person wearing helmet", "skateboard wheel", "bus window"]`.
3. Corré inferencia sobre la misma imagen del Ej. 1 (`https://ultralytics.com/images/bus.jpg`) con `conf=0.05` (los modelos open-vocabulary suelen necesitar thresholds más bajos).
4. Visualizá el resultado e imprimí las detecciones con clase y confianza.

> **Pista:** La API de `YOLOWorld` es prácticamente idéntica a la de `YOLO`. La única diferencia operativa es la llamada a `set_classes(...)` antes de la inferencia.
::::


::::cell{#ej6-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Cargar YOLO-World ──────────────────────────────────────────────────────
# YOLO-World es un detector "open-vocabulary": en vez de tener un conjunto
# fijo de clases, acepta una lista de nombres de clase como prompt de texto
# y detecta esos objetos sin fine-tuning específico.
from ultralytics import YOLOWorld

world_model = YOLOWorld('yolov8s-world.pt')

# ─── Definir las clases mediante texto ─────────────────────────────────────
# Probemos clases que NO estaban en COCO ni en BCCD: descripciones más
# específicas que combinan objeto + atributo.
world_model.set_classes(["person wearing helmet", "skateboard wheel", "bus window"])

img_url = 'https://ultralytics.com/images/bus.jpg'
res = world_model(img_url, conf=0.05, verbose=False)[0]

plt.figure(figsize=(10, 8))
plt.imshow(res.plot()[:, :, ::-1])
plt.axis('off')
plt.title('YOLO-World con prompt de texto')
plt.show()

print(f"Detecciones: {len(res.boxes)}")
for box in res.boxes:
    cls_id   = int(box.cls[0])
    cls_name = res.names[cls_id]
    conf     = float(box.conf[0])
    print(f"  {cls_name:25s}  conf={conf:.2f}")
```
::::


::::cell{#ej6-pregunta type=markdown role=pregunta}
**Pregunta de análisis (final):**

Para cada uno de los siguientes escenarios, indicá qué familia de detector usarías (YOLO con fine-tuning, YOLO sin fine-tuning, transformer-based, zero-shot/open-vocabulary) y justificá brevemente.

a) Una **app móvil** que detecta personas y autos en tiempo real para asistencia a la conducción.

b) Un **investigador biólogo** que quiere encontrar "células anómalas" en sus imágenes de microscopía pero todavía no tiene un dataset etiquetado.

c) Una **línea de fabricación** que necesita detectar 5 tipos específicos de defectos en piezas. Tienen 3000 imágenes etiquetadas y exigen máximo accuracy.

d) Un **prototipo rápido** para mostrarle a un cliente un demo de "detección de productos en góndola", sin tiempo para etiquetar.
::::


::::cell{#ej6-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis (final):**

a) **YOLO sin fine-tuning** (directamente `yolo11n` o `yolo11s` sobre COCO). "Persona" y "auto" están en COCO, así que no hace falta entrenar nada. Lo crítico es **latencia baja** en hardware modesto (móvil); YOLO en variante nano corre en tiempo real incluso en CPU.

b) **Zero-shot / open-vocabulary** (YOLO-World, Grounding DINO). No hay dataset todavía y el concepto de "anómala" no es trivial de fijar; el investigador puede prototipar con prompts iterativos ("redondas con núcleo grande", "irregulares") sin invertir tiempo en etiquetado. Una vez que entiende qué busca, puede pasar a fine-tuning si necesita más accuracy.

c) **YOLO con fine-tuning** (variante mediana o grande, `yolo11m`/`yolo11l`). Tienen 3000 imágenes etiquetadas — datos suficientes —, las clases son fijas y estables, y la prioridad es **máximo accuracy**. Un modelo fine-tuneado sobre datos del dominio supera a cualquier zero-shot. Ventaja adicional: YOLO se exporta fácil a ONNX/TensorRT para integrarlo en la línea con baja latencia.

d) **Zero-shot.** Es exactamente el escenario para el que están pensados estos modelos: no hay tiempo de etiquetar, las clases son fáciles de describir en texto ("bottle of soda", "cereal box", "bag of chips"), y el resultado no necesita ser perfecto sino convincente para la demo. Si el cliente aprueba, después se hace fine-tuning con datos reales.

**Patrón general:**
- ¿Tenés datos etiquetados y clases fijas? → fine-tuning.
- ¿No tenés datos pero las clases son cotidianas y están en COCO? → modelo preentrenado tal cual.
- ¿No tenés datos y las clases son específicas / abiertas? → zero-shot.
- ¿La latencia es crítica? → YOLO (sobre las otras opciones).
```
::::


::::cell{#checklist type=markdown role=checklist}
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] El fine-tuning corrió completo y el checkpoint quedó guardado en `runs/detect/bccd_finetune/weights/best.pt`.
- [ ] Los `mAP50` por clase tienen valores razonables (no todos en 0.0 ni en 1.0).
- [ ] Las visualizaciones muestran las cajas predichas con sus clases y confianzas.
- [ ] Respondí las cuatro preguntas de análisis (Ej. 1, 2, 3, 5, 6).
- [ ] No modifiqué ninguna celda fuera de las de actividad.
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Completaste tu primer laboratorio de detección de objetos. Practicaste:

- **Inferencia con un detector preentrenado** (YOLOv11 sobre COCO) y los efectos de los thresholds de confianza y NMS.
- **Fine-tuning** sobre un dataset clínico chico (BCCD), con la lógica de transfer learning aplicada a detección.
- **Evaluación** con mAP por clase y aplicación a una tarea concreta (conteo de células).
- **Detección zero-shot** con YOLO-World, viendo cuándo conviene cada familia de detector.

El próximo laboratorio (4b) extiende lo que vimos acá a **segmentación de imágenes**: en lugar de predecir solo una caja alrededor de cada objeto, vamos a predecir el contorno exacto píxel por píxel.
::::
