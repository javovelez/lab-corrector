---
lab: "5b"
title: "Laboratorio n° 5. Parte B: Autoencoders Variacionales"
subject: "Redes Neuronales Profundas"
block: "5 — Modelos generativos"
---

<!-- ════════════════════════════════════════════════════════════════════════
     Laboratorio 5b — Autoencoders Variacionales
     Fuente única. Compilar con: python tools/lab_build.py _TPS/sources/Laboratorio_5b.lab.md
     Genera: _TPS/Laboratorios/Laboratorio_5b.ipynb
             _TPS/Soluciones/Laboratorio_5b_Solucion.ipynb
     ════════════════════════════════════════════════════════════════════════ -->


::::cell{#img-header type=markdown role=header}
![Imgur](https://i.imgur.com/acSOZRh.png)
::::


::::cell{#header type=markdown role=title}
# Laboratorio n° 5. Parte B: Autoencoders Variacionales

**Asignatura:** Redes Neuronales Profundas
**Bloque:** 5 — Modelos generativos

---

## Introducción

Un **Autoencoder Variacional** (*VAE*) es un modelo generativo que aprende a comprimir imágenes en un **espacio latente continuo** y a generar imágenes nuevas muestreando puntos de ese espacio. A diferencia de un autoencoder clásico, el VAE codifica cada imagen como una **distribución de probabilidad** en lugar de un punto único. Esa distinción es lo que hace posible muestrear, interpolar entre dos imágenes o moverse de forma suave por el espacio latente.

En este laboratorio vas a construir un VAE pieza por pieza y entrenarlo sobre un dataset de retratos de jugadores de fútbol del videojuego FIFA. Al final, vas a poder:

- Definir los bloques convolucionales que arman encoder y decoder.
- Implementar un **encoder variacional** que devuelve la media `μ` y la varianza logarítmica `log_var` de la distribución latente, y aplicar el **truco de la reparametrización** para muestrear de forma diferenciable.
- Diseñar un decoder que reconstruye una imagen RGB a partir de un código latente.
- Entrenar el modelo con la pérdida característica del VAE: reconstrucción + divergencia KL.
- Generar imágenes nuevas muestreando del prior y observar qué aprende el modelo.
- Hacer una **interpolación entre dos jugadores** caminando linealmente por el espacio latente y entender qué nos dice el resultado sobre la estructura aprendida.

> **Importante — GPU:** este laboratorio entrena una red convolucional. **Activá la GPU en Colab** antes de empezar: *Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU*. Sin GPU el entrenamiento tarda decenas de minutos.

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
#   - torch / torchvision: módulos, capas, optimizadores y transformaciones.
#   - pandas / matplotlib / numpy: lectura del CSV y visualización.
# Detectamos GPU una sola vez y guardamos el dispositivo en `device`. Esa
# variable es la que vas a usar en todos los `.to(device)` de tus ejercicios.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# ─── Setup: descarga del dataset FIFA ───────────────────────────────────────
# Bajamos un zip con ~600 retratos de jugadores (PNG con canal alfa) y un CSV
# con los metadatos (nombre, equipo, posición, etc.). Es el mismo dataset que
# se usó en la clase de GANs. Esta celda funciona igual en Google Colab y
# en local: el dataset queda en `./fifa_data/` relativo al notebook, y solo
# se baja la primera vez.
import zipfile

try:
    import gdown
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    import gdown

DATA_DIR = "fifa_data"
os.makedirs(DATA_DIR, exist_ok=True)

zip_path    = os.path.join(DATA_DIR, "fifa.zip")
images_dir  = os.path.join(DATA_DIR, "Images")
csv_path    = os.path.join(DATA_DIR, "data.csv")

if not (os.path.isdir(images_dir) and os.path.exists(csv_path)):
    if not os.path.exists(zip_path):
        gdown.download(
            id="1zpT6gHzvbr21_Vq4NjpRY0JGSaQOSANa",
            output=zip_path,
            quiet=False,
        )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)

print(f"Dataset listo en ./{DATA_DIR}/")
```
::::


::::cell{#setup-helpers-data type=code role=setup}
```python
# ─── Setup: lectura, limpieza y dataset ─────────────────────────────────────
# Las imágenes vienen en RGBA con un fondo transparente. Tomamos el canal
# alpha como máscara para separar al jugador del fondo y nos quedamos solo
# con los tres canales RGB.
def remove_noise(image):
    """Usa el canal alfa como máscara binaria y devuelve la imagen RGB."""
    alpha = (image[3, :, :] > 50).type(torch.uint8)
    masked = torch.mul(alpha, image)
    return masked[:3, :, :]


def read_voc_images(voc_dir, n=-1):
    """Lee todas las imágenes del directorio en orden numérico de archivo."""
    files = sorted(
        os.listdir(voc_dir),
        key=lambda i: int(os.path.splitext(i)[0]),
    )
    mode = torchvision.io.image.ImageReadMode.RGB_ALPHA
    features, labels = [], []
    for i, fname in enumerate(files):
        if i == n:
            break
        try:
            img = remove_noise(torchvision.io.read_image(
                os.path.join(voc_dir, fname), mode))
        except Exception:
            continue
        features.append(img)
        labels.append(int(os.path.splitext(fname)[0]))
    return features, labels


class FIFADataset(torch.utils.data.Dataset):
    """Imágenes de jugadores redimensionadas a 64×64 y escaladas a [0, 1]."""
    def __init__(self, img_list, labels, names, size=64):
        super().__init__()
        self.img_list = img_list
        self.labels = labels
        self.names = names
        self.transform = torchvision.transforms.Resize((size, size))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx] / 255.0
        return self.transform(img.type(torch.float32)), self.labels[idx], self.names[idx]


# Leemos las imágenes y los nombres una sola vez.
data = pd.read_csv(csv_path)
imgs_list, labels_list = read_voc_images(images_dir)
names_list = list(data["Name"])

fifa = FIFADataset(imgs_list, labels_list, names_list)

batch_size = 128
# `num_workers=0` evita problemas de pickling de la clase FIFADataset en
# macOS (donde Python 3.8+ usa `spawn` en vez de `fork` para los workers).
# Con un dataset que ya está en memoria, no hay ganancia significativa por
# usar varios workers.
data_iter = torch.utils.data.DataLoader(
    fifa, batch_size=batch_size, shuffle=True, num_workers=0,
)

print(f"Total de imágenes: {len(fifa)}")
print(f"Forma de una imagen del dataset: {fifa[0][0].shape}")
```
::::


::::cell{#setup-helpers-vis type=code role=setup}
```python
# ─── Setup: helpers de visualización ────────────────────────────────────────
# Los usás en los ejercicios de reconstrucción, muestreo e interpolación.
# No hace falta entender el detalle interno: cada uno toma tensores en CPU y
# arma una grilla con matplotlib.
def mostrar_grilla(imgs, num_rows, num_cols, titulos=None, scale=1.5):
    """Muestra una lista de imágenes en una grilla."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        if titulos is not None and i < len(titulos):
            ax.set_title(titulos[i], fontsize=8)
    plt.tight_layout()
    plt.show()


def mostrar_recon(originales, reconstruidas, titulos=None, scale=1.5):
    """Muestra una fila de originales sobre una fila de reconstrucciones."""
    n = originales.shape[0]
    figsize = (n * scale, 2 * scale)
    _, axes = plt.subplots(2, n, figsize=figsize)
    for i in range(n):
        for fila, batch in enumerate([originales, reconstruidas]):
            img = batch[i].detach().cpu().permute(1, 2, 0).numpy()
            axes[fila, i].imshow(np.clip(img, 0, 1))
            axes[fila, i].set_xticks([]); axes[fila, i].set_yticks([])
        if titulos is not None and i < len(titulos):
            axes[0, i].set_title(titulos[i], fontsize=8)
    axes[0, 0].set_ylabel("original",  fontsize=10)
    axes[1, 0].set_ylabel("reconstr.", fontsize=10)
    plt.tight_layout()
    plt.show()


def mostrar_interp(imgs_decoded, orig_inicio, orig_fin,
                   nombre_inicio, nombre_fin, scale=1.5):
    """Muestra una fila con la interpolación entre dos imágenes:
    en los extremos las imágenes originales del dataset, y entre medio las
    decodificaciones de la trayectoria latente. La primera y última de
    `imgs_decoded` son las decodificaciones de μ_inicio y μ_fin."""
    panels = (
        [orig_inicio]
        + [imgs_decoded[i] for i in range(imgs_decoded.shape[0])]
        + [orig_fin]
    )
    n = len(panels)
    figsize = (n * scale, scale)
    _, axes = plt.subplots(1, n, figsize=figsize)
    for ax, img in zip(axes, panels):
        img = img.detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title(f"{nombre_inicio}\n(original)",  fontsize=8)
    axes[1].set_title(f"{nombre_inicio}\n(VAE)",       fontsize=8)
    axes[-2].set_title(f"{nombre_fin}\n(VAE)",         fontsize=8)
    axes[-1].set_title(f"{nombre_fin}\n(original)",    fontsize=8)
    plt.tight_layout()
    plt.show()


def linear_interpolation(v1, v2, n_steps):
    """Interpola linealmente entre dos vectores y devuelve un tensor (n_steps, dim).
    El primer punto coincide con v1 y el último con v2."""
    alphas = torch.linspace(0, 1, n_steps, device=v1.device).unsqueeze(1)
    return (1 - alphas) * v1 + alphas * v2
```
::::


::::cell{#setup-train type=code role=setup}
```python
# ─── Setup: loop de entrenamiento ───────────────────────────────────────────
# Esta función hace el descenso de gradiente estándar para un VAE: por cada
# batch hace forward, calcula la pérdida con `vae_loss` (que vas a definir
# en el Ej 4), backward y step. Reporta la pérdida promedio por imagen.
# El único requisito es que `model(x)` devuelva la tupla `(x_hat, mu, log_var)`.
def train_vae(model, dataloader, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        loss_acumulada, n_imgs = 0.0, 0
        for x, _, _ in dataloader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, log_var = model(x)
            loss = vae_loss(x, x_hat, mu, log_var)
            loss.backward()
            opt.step()
            loss_acumulada += loss.item()
            n_imgs += x.size(0)
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"pérdida promedio por imagen: {loss_acumulada / n_imgs:.2f}")
    return model
```
::::


::::cell{#secA type=markdown role=section}
---
## Sección A: Bloques convolucionales
::::


::::cell{#intro-secA type=markdown role=intro}
Antes de armar el encoder y el decoder vamos a definir los **dos bloques de construcción** que se repiten varias veces en cada uno. Mantener cada bloque en su propia clase hace que la arquitectura del modelo se lea de un vistazo como una secuencia de bloques.

- `DEC_block`: bloque de **bajada de resolución** que va al encoder. Hace `Conv2d` (con `stride=2`, así reduce a la mitad alto y ancho) seguido de `BatchNorm2d` y `LeakyReLU`.
- `AUG_block`: bloque de **subida de resolución** que va al decoder. Hace `ConvTranspose2d` (con `stride=2`, que duplica alto y ancho) seguido de `BatchNorm2d` y `ReLU`.

> **Pista:** la `BatchNorm2d` siempre va **después** de la convolución y **antes** de la activación. Es la convención estándar en arquitecturas convolucionales modernas.
::::


::::cell{#ej1-enunciado type=markdown role=enunciado}
### Ejercicio 1 — Bloques convolucionales del encoder y del decoder

**Objetivo:** Implementar dos bloques reutilizables. El primero baja resolución (lo usás en el encoder); el segundo la sube (lo usás en el decoder).

**Enunciado:**

**Parte A — `DEC_block` (bajada de resolución):**

1. Heredá de `nn.Module`.
2. En `__init__` instanciá tres capas con los argumentos del constructor:
   - `nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding=padding)`.
   - `nn.BatchNorm2d(out_channels)`.
   - `nn.LeakyReLU(alpha)`.
3. En `forward(X)` aplicalas en orden: convolución → batchnorm → activación.

**Parte B — `AUG_block` (subida de resolución):**

1. Igual al anterior pero con `nn.ConvTranspose2d` en lugar de `Conv2d`, y `nn.ReLU` en lugar de `LeakyReLU`.
2. En `forward(X)` aplicá: convolución transpuesta → batchnorm → activación.

> **Pista:** los argumentos por defecto que ya están en la firma (`kernel_size=4`, `strides=2`, `padding=1`) están elegidos para que cada bloque divida o multiplique por 2 las dimensiones espaciales. No hace falta cambiarlos en los ejercicios siguientes.
::::


::::cell{#ej1-code-a type=code role=student-code}
```python
class DEC_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1, alpha=0.2):
        super().__init__()
        # Tu código aquí

    def forward(self, X):
        # Tu código aquí
        pass
```

```python solution
class DEC_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1, alpha=0.2):
        super().__init__()
        # ─── Conv → BatchNorm → LeakyReLU ─────────────────────────────────
        # Conv2d con stride=2 reduce a la mitad alto y ancho (con padding=1
        # y kernel_size=4). BatchNorm2d estabiliza el entrenamiento, y
        # LeakyReLU permite que pasen pendientes negativas chiquitas, lo
        # que ayuda en el encoder donde las activaciones pueden saturar.
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=strides, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, X):
        return self.act(self.bn(self.conv(X)))
```
::::


::::cell{#ej1-code-b type=code role=student-code}
```python
class AUG_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1):
        super().__init__()
        # Tu código aquí

    def forward(self, X):
        # Tu código aquí
        pass
```

```python solution
class AUG_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1):
        super().__init__()
        # ─── ConvTranspose → BatchNorm → ReLU ─────────────────────────────
        # ConvTranspose2d con stride=2 duplica alto y ancho. ReLU es la
        # activación canónica del decoder: como las salidas se usan para
        # reconstruir píxeles en [0, 1], no necesitamos las pendientes
        # negativas que sí ayudan en el encoder.
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=strides, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, X):
        return self.act(self.bn(self.deconv(X)))
```
::::


::::cell{#secB type=markdown role=section}
---
## Sección B: Encoder variacional
::::


::::cell{#intro-secB type=markdown role=intro}
El **encoder variacional** es la pieza clave del VAE: en lugar de mapear cada imagen a un vector único, la mapea a una **distribución de probabilidad** sobre el espacio latente. Concretamente, predice los parámetros de una gaussiana multivariada con matriz de covarianza diagonal:

$$q(z \mid x) = \mathcal{N}(z;\, \mu(x),\, \mathrm{diag}(\sigma^2(x)))$$

Para que esto se pueda entrenar con descenso de gradiente, no muestreamos `z` con `torch.normal(mu, std)` (esa operación no es diferenciable respecto a `mu` y `std`). Aplicamos el **truco de la reparametrización**: sampleamos un ruido `eps ~ N(0, I)` y construimos `z = mu + eps * std`. Así, `z` queda como una función diferenciable de `mu` y `std`, y los gradientes pueden propagarse hasta los pesos del encoder.

> **Detalle numérico:** la red no predice directamente `var` o `std`, sino el logaritmo de la varianza, `log_var`. Trabajar en log evita restringir la salida a valores positivos: `log_var ∈ ℝ` mientras que `var > 0`. Después se recupera `std = exp(0.5 * log_var)`.
::::


::::cell{#ej2-enunciado type=markdown role=enunciado}
### Ejercicio 2 — Encoder variacional

**Objetivo:** Implementar el encoder completo: tronco convolucional + cabezas para `μ` y `log_var` + reparametrización.

**Enunciado:**

1. En `__init__`, definí:
   - `self.trunk` como un `nn.Sequential` con cuatro `DEC_block` consecutivos que llevan los canales `3 → 32 → 64 → 128 → 256` (las dimensiones espaciales bajan `64 → 32 → 16 → 8 → 4`). Terminá el tronco con un `nn.Flatten()` para obtener un vector de tamaño `256*4*4 = 4096`.
   - `self.fc_mu` y `self.fc_log_var`: dos `nn.Linear(256*4*4, latent_dims)`. **Sin activación**: queremos cabezas lineales puras.
2. En `forward(x)`:
   - Pasá `x` por el tronco y obtené `h`.
   - Calculá `mu = self.fc_mu(h)` y `log_var = self.fc_log_var(h)`.
   - Aplicá el **truco de la reparametrización**:
     - `std = torch.exp(0.5 * log_var)`.
     - `eps = torch.randn_like(std)`.
     - `z = mu + eps * std`.
   - Devolvé la tupla `(z, mu, log_var)`.

> **Pista:** las dos cabezas se construyen con el mismo tipo de capa pero pesos independientes — no compartas la misma `nn.Linear` para las dos.
::::


::::cell{#ej2-code type=code role=student-code}
```python
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        # Tu código aquí

    def forward(self, x):
        # Tu código aquí
        pass
```

```python solution
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        # ─── Tronco convolucional: (3, 64, 64) → (4096,) ───────────────────
        # Cada DEC_block divide a la mitad las dimensiones espaciales y
        # duplica los canales: 64→32→16→8→4 en espacio, 3→32→64→128→256 en
        # canales. Al final aplanamos para conectar con las dos cabezas.
        self.trunk = nn.Sequential(
            DEC_block(in_channels=3,   out_channels=32),    # → (32, 32, 32)
            DEC_block(in_channels=32,  out_channels=64),    # → (64, 16, 16)
            DEC_block(in_channels=64,  out_channels=128),   # → (128, 8, 8)
            DEC_block(in_channels=128, out_channels=256),   # → (256, 4, 4)
            nn.Flatten(),                                   # → (4096,)
        )
        # ─── Dos cabezas lineales independientes ───────────────────────────
        # Pesos separados: cada cabeza aprende a producir una cantidad
        # distinta (μ y log_var). NO ponemos activación: μ y log_var son
        # números reales sin restricción.
        self.fc_mu      = nn.Linear(256 * 4 * 4, latent_dims)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dims)

    def forward(self, x):
        h = self.trunk(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        # ─── Truco de la reparametrización ─────────────────────────────────
        # std = exp(0.5 * log_var) recupera el desvío estándar a partir del
        # log de la varianza. eps ~ N(0, I) absorbe toda la aleatoriedad,
        # de modo que z = mu + eps * std es diferenciable respecto a mu y
        # log_var, y el gradiente puede propagarse hasta los pesos del
        # encoder durante el backward pass.
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z   = mu + eps * std

        return z, mu, log_var
```
::::


::::cell{#ej2-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

El `forward` del encoder devuelve **tres tensores**: `z`, `mu` y `log_var`.

1. ¿Qué representa cada uno? ¿Cuál es la diferencia conceptual entre `z` y `mu`?
2. ¿Por qué hace falta el truco de la reparametrización en lugar de simplemente muestrear con `torch.normal(mu, std)`?
::::


::::cell{#ej2-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

1. **Qué representa cada uno.**
   - `mu` y `log_var` son los **parámetros** de la distribución latente que el encoder predice para la imagen `x`: una gaussiana `q(z|x) = N(μ, σ²·I)`. Son la "creencia" de la red sobre dónde vive el código de `x` en el espacio latente y cuánta incertidumbre tiene sobre esa estimación.
   - `z` es **una muestra concreta** de esa distribución, el código que efectivamente entregamos al decoder. Por eso `z` cambia en cada forward pass aunque la imagen sea la misma — `mu` y `log_var` no.
   - Como `z = mu + eps * std`, podés pensarlo así: `mu` es el "centro" del código y `eps * std` es el ruido que el encoder agrega para forzar al decoder a aprender una región del espacio latente y no un solo punto.

2. **Por qué la reparametrización.**
   La operación `torch.normal(mu, std)` muestrea de una gaussiana con esos parámetros, pero la operación de muestreo en sí **no es diferenciable** respecto a `mu` y `std`: PyTorch no sabe propagar gradientes a través de "elegir un número aleatorio". Si entrenáramos así, los pesos del encoder no recibirían señal de gradiente y nunca aprenderían qué `mu` y `log_var` predecir.

   El truco saca la aleatoriedad afuera de los parámetros aprendibles: muestreamos `eps ~ N(0, I)` (sin parámetros) y reconstruimos `z = mu + eps * std`. Esa última expresión es una operación determinista (suma y multiplicación) respecto a `mu` y `std`, así que el gradiente de la pérdida fluye limpiamente hasta el encoder.
```
::::


::::cell{#secC type=markdown role=section}
---
## Sección C: Decoder
::::


::::cell{#intro-secC type=markdown role=intro}
El decoder hace el camino inverso al encoder: toma un código latente `z` (un vector de tamaño `latent_dims`) y reconstruye una imagen RGB de 64×64. La estructura es **espejo del encoder**:

1. Una capa lineal expande `z` al volumen `(256, 4, 4)`, que es donde el encoder había llegado al final del tronco convolucional.
2. Tres `AUG_block` van duplicando las dimensiones espaciales (`4 → 8 → 16 → 32`) y reduciendo los canales (`256 → 128 → 64 → 32`).
3. Una `nn.ConvTranspose2d` final lleva de `(32, 32, 32)` a `(3, 64, 64)`.
4. Una `nn.Sigmoid` mapea cada píxel al rango `[0, 1]`, igual que las imágenes de entrada.
::::


::::cell{#ej3-enunciado type=markdown role=enunciado}
### Ejercicio 3 — Decoder

**Objetivo:** Implementar el decoder que mapea un vector latente a una imagen RGB.

**Enunciado:**

1. En `__init__`:
   - `self.fc = nn.Linear(latent_dims, 256 * 4 * 4)`.
   - `self.deconvs = nn.Sequential(...)` con, en orden:
     - `AUG_block(in_channels=256, out_channels=128)`.
     - `AUG_block(in_channels=128, out_channels=64)`.
     - `AUG_block(in_channels=64,  out_channels=32)`.
     - `nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)`.
     - `nn.Sigmoid()`.
2. En `forward(z)`:
   - Aplicá la lineal: `h = self.fc(z)` (shape `(B, 4096)`).
   - **Reformateá** el vector a un volumen 4D usando `h.view(-1, 256, 4, 4)`.
   - Pasalo por `self.deconvs` y devolvé el resultado.

> **Pista:** el `view(-1, 256, 4, 4)` mantiene el tamaño del batch automáticamente (el `-1` significa "lo que haga falta para que cierre"). Después de eso ya tenés un tensor 4D listo para entrar a las convoluciones transpuestas.
::::


::::cell{#ej3-code type=code role=student-code}
```python
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        # Tu código aquí

    def forward(self, z):
        # Tu código aquí
        pass
```

```python solution
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        # ─── Lineal de expansión: (latent_dims,) → (256*4*4,) ──────────────
        # Llevamos el vector latente al tamaño que ocupa el "cuello" del
        # encoder al final del tronco convolucional. Después lo reshapeamos
        # a 4D y lo pasamos por las convoluciones transpuestas.
        self.fc = nn.Linear(latent_dims, 256 * 4 * 4)

        # ─── Camino convolucional: (256, 4, 4) → (3, 64, 64) ───────────────
        # Tres AUG_block duplican espacio y reducen canales: 4→8→16→32 en
        # espacio, 256→128→64→32 en canales. La ConvTranspose2d final
        # produce los 3 canales RGB y duplica una vez más el espacio
        # (32→64). Sigmoid escala los píxeles al rango [0, 1] que la
        # pérdida BCE espera.
        self.deconvs = nn.Sequential(
            AUG_block(in_channels=256, out_channels=128),  # → (128, 8, 8)
            AUG_block(in_channels=128, out_channels=64),   # → (64, 16, 16)
            AUG_block(in_channels=64,  out_channels=32),   # → (32, 32, 32)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.deconvs(h)
```
::::


::::cell{#ej3-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La última capa del decoder es una `nn.Sigmoid`. ¿Por qué es **necesaria** ahí? ¿Qué pasaría si la sacáramos, o si la reemplazáramos por `nn.Tanh`?
::::


::::cell{#ej3-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Las imágenes de entrada se normalizaron al rango `[0, 1]` cuando armamos el `Dataset` (división por 255). Para que la pérdida de reconstrucción tenga sentido, el decoder tiene que producir imágenes **en el mismo rango**.

- **Con `Sigmoid`**: la salida queda en `(0, 1)`, exactamente lo que necesitamos. Además, en el Ej 4 vamos a usar `binary_cross_entropy` como pérdida de reconstrucción, y esa función **requiere** que sus dos argumentos estén en `[0, 1]` — sin la sigmoide tira `nan` o un error.
- **Sin activación final**: las salidas del decoder serían valores reales sin restricción, y la BCE explotaría. Si cambiáramos la pérdida a MSE el modelo podría entrenarse igual, pero ya no estaríamos modelando los píxeles como probabilidades, lo cual es lo que el VAE estándar asume.
- **Con `Tanh`**: el rango sería `(-1, 1)`. Para que esto sea coherente habría que normalizar las imágenes de entrada también a `[-1, 1]` (típicamente con `(x - 0.5) * 2`) y reemplazar la BCE por MSE. Es una alternativa válida en otras arquitecturas (por ejemplo en GANs es habitual), pero exige cambiar varios eslabones del pipeline.

Resumen: la `Sigmoid` es lo que conecta el rango de salida del decoder con la pérdida BCE y con el rango original de los datos.
```
::::


::::cell{#secD type=markdown role=section}
---
## Sección D: Pérdida del VAE
::::


::::cell{#intro-secD type=markdown role=intro}
La pérdida del VAE se obtiene de un objetivo variacional llamado **ELBO** (*Evidence Lower BOund*), que tiene dos términos:

$$\mathcal{L}_{\text{VAE}}(x) = \underbrace{-\log p(x \mid z)}_{\text{reconstrucción}} \;+\; \underbrace{\mathrm{KL}\!\left(q(z\mid x)\,\Vert\,p(z)\right)}_{\text{regularización}}$$

- El **término de reconstrucción** mide qué tan parecida es la imagen reconstruida a la original. Como modelamos cada píxel como una probabilidad en `[0, 1]`, usamos `binary_cross_entropy` con `reduction='sum'` (sumamos el aporte de todos los píxeles del batch).
- El **término KL** mide qué tan lejos está la posterior aproximada `q(z|x) = N(μ, σ²·I)` del prior `p(z) = N(0, I)`. Para dos gaussianas con covarianza diagonal este término tiene **forma cerrada**:

$$\mathrm{KL}(q\,\Vert\,p) \;=\; -\tfrac{1}{2}\sum_{i=1}^{D} \big(1 + \log \sigma_i^2 - \sigma_i^2 - \mu_i^2 \big)$$

donde `D = latent_dims` y la suma corre sobre las dimensiones del latente. En PyTorch eso se traduce literalmente a una línea con `log_var` y `mu`.

> **Pista:** ambas componentes se suman tal cual. No hace falta dividir por el batch ni normalizar — el optimizador se ocupa de la escala.
::::


::::cell{#ej4-enunciado type=markdown role=enunciado}
### Ejercicio 4 — Pérdida del VAE

**Objetivo:** Implementar la función de pérdida que combina reconstrucción y divergencia KL.

**Enunciado:**

Implementá `vae_loss(x, x_hat, mu, log_var)` que devuelva un escalar:

1. **Reconstrucción**: `F.binary_cross_entropy(x_hat, x, reduction='sum')`.
2. **KL**: aplicá la fórmula cerrada `-0.5 * torch.sum(1 + log_var - log_var.exp() - mu.pow(2))`.
3. Sumá las dos componentes y devolvé el resultado.

> **Pista:** `log_var.exp()` corresponde a `σ²` (no a `σ`). Es importante respetar la fórmula tal como aparece arriba — un signo cambiado y la KL se vuelve negativa, lo que rompe el entrenamiento.
::::


::::cell{#ej4-code type=code role=student-code}
```python
def vae_loss(x, x_hat, mu, log_var):
    # Tu código aquí
    pass
```

```python solution
def vae_loss(x, x_hat, mu, log_var):
    # ─── Término 1: reconstrucción (BCE píxel a píxel) ──────────────────────
    # reduction='sum' acumula la pérdida de todos los píxeles del batch.
    # Esa elección es importante: si usáramos 'mean', la KL quedaría con un
    # peso relativo muy distinto y el balance entre los dos términos sería
    # otro.
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # ─── Término 2: KL[q(z|x) || N(0, I)] en forma cerrada ─────────────────
    # La fórmula vale solo porque tanto la posterior como el prior son
    # gaussianas con covarianza diagonal. Empuja a μ hacia 0 y a σ² hacia 1
    # (con eso 1 + log_var − σ² − μ² alcanza el máximo y la KL se anula).
    kl = -0.5 * torch.sum(1 + log_var - log_var.exp() - mu.pow(2))

    return recon + kl
```
::::


::::cell{#ej4-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

La pérdida del VAE tiene **dos términos** que tiran en direcciones distintas. ¿Qué objetivo busca cada uno y qué le pasaría al modelo si entrenáramos solo con la reconstrucción (es decir, sin el término KL)?
::::


::::cell{#ej4-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

- **Reconstrucción.** Empuja al modelo a que `x_hat` se parezca lo más posible a `x`. Esto sí o sí requiere que el encoder mande imágenes parecidas a códigos parecidos y que el decoder aprenda a invertir esa codificación píxel a píxel. Por sí solo, este término convierte al VAE en un autoencoder cualquiera.

- **KL.** Es el término **regularizador**. Mide la distancia entre la posterior aproximada `q(z|x) = N(μ, σ²)` y el prior `p(z) = N(0, I)`. La empuja a estar centrada cerca del origen (`μ ≈ 0`) y con varianza cercana a 1 (`σ² ≈ 1`). El efecto práctico es que las distribuciones codificadas para distintas imágenes **se solapan** en el espacio latente y cubren la región donde vive el prior, sin huecos.

- **Sin KL.** El modelo degenera a un autoencoder clásico. El encoder es libre de mandar cada imagen a un punto puntual del latente, sin orden estructural ni continuidad. La reconstrucción puede ser excelente, pero el espacio latente queda **lleno de huecos**: si después muestreamos `z ~ N(0, I)` para generar imágenes nuevas, casi cualquier `z` cae en una región donde el decoder nunca aprendió nada coherente y la salida es ruido o un "promedio" sin sentido. Tampoco interpolar entre dos códigos da resultados suaves: la trayectoria atraviesa zonas no entrenadas.

La combinación de los dos términos es lo que distingue al VAE: la KL **paga el precio** de una reconstrucción levemente más borrosa a cambio de un espacio latente continuo y muestreable.
```
::::


::::cell{#secE type=markdown role=section}
---
## Sección E: Ensamble y entrenamiento
::::


::::cell{#intro-secE type=markdown role=intro}
Con encoder, decoder y pérdida ya definidos, lo único que falta es ensamblarlos en un único `nn.Module` y lanzar el entrenamiento. La función `train_vae` ya está provista en el setup: hace el loop estándar de descenso de gradiente y reporta la pérdida promedio por imagen en cada época.

> **Tiempos esperados:** con GPU, 30 épocas tardan unos 5 minutos. Si el dispositivo detectado es `cpu`, el entrenamiento puede llevar más de media hora — volvé al menú de Colab y activá la GPU antes de continuar.
::::


::::cell{#ej5-enunciado type=markdown role=enunciado}
### Ejercicio 5 — Ensamblar y entrenar el VAE

**Objetivo:** Combinar encoder y decoder en un único modelo y entrenarlo.

**Enunciado:**

1. Definí la clase `VariationalAutoencoder(nn.Module)`. En `__init__` instanciá:
   - `self.encoder = VariationalEncoder(latent_dims)`.
   - `self.decoder = Decoder(latent_dims)`.
2. En `forward(x)`:
   - Codificá: `z, mu, log_var = self.encoder(x)`.
   - Decodificá: `x_hat = self.decoder(z)`.
   - Devolvé la tupla `(x_hat, mu, log_var)` (en ese orden — es lo que `train_vae` espera).
3. Instanciá un VAE con `latent_dims = 128`, movelo al `device` y entrenalo durante 30 épocas con `train_vae(vae, data_iter, epochs=30)`.

> **Pista:** la forma en que el `forward` retorna la tupla es importante: `train_vae` y `vae_loss` desempaquetan exactamente esos tres tensores. Si los devolvés en otro orden, el loop falla en silencio.
::::


::::cell{#ej5-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# ─── Instancia y entrenamiento ──────────────────────────────────────────────
# Con latent_dims=128 el espacio latente es lo bastante grande como para
# capturar la variabilidad de los retratos sin volverse desordenado.
latent_dims = 128
vae = VariationalAutoencoder(latent_dims).to(device)
train_vae(vae, data_iter, epochs=30)
```
::::


::::cell{#secF type=markdown role=section}
---
## Sección F: Reconstrucción y muestreo del prior
::::


::::cell{#intro-secF type=markdown role=intro}
Ahora que el modelo está entrenado podemos evaluarlo cualitativamente desde dos ángulos complementarios:

- **Reconstrucción.** Tomamos imágenes reales del dataset, las pasamos por el VAE (encoder → decoder) y comparamos la salida con el original. Es la prueba mínima de que el modelo aprendió a comprimir y descomprimir.
- **Muestreo del prior.** Sampleamos vectores latentes directamente del prior `z ~ N(0, I)` (sin pasar por ninguna imagen real) y los pasamos por el decoder. Es la prueba de que el espacio latente está lo bastante bien estructurado como para que cualquier código razonable produzca algo coherente.

Vas a comparar las dos salidas en la pregunta de análisis.
::::


::::cell{#ej6-enunciado type=markdown role=enunciado}
### Ejercicio 6 — Reconstrucción y muestras del prior

**Objetivo:** Evaluar el modelo entrenado con dos pruebas cualitativas.

**Enunciado:**

**Parte A — Reconstrucción.**

1. Poné el modelo en modo evaluación: `vae.eval()`.
2. Sin construir el grafo de gradientes (usá `with torch.no_grad():`):
   - Tomá un batch del `data_iter`: `x_batch, _, nombres_batch = next(iter(data_iter))`.
   - Quedate con las primeras **8 imágenes** y mandalas al `device`.
   - Pasalas por el VAE para obtener `x_hat`.
3. Mostrá originales y reconstrucciones con `mostrar_recon(x_batch[:8].cpu(), x_hat.cpu(), titulos=nombres_batch[:8])`.

**Parte B — Muestras del prior.**

1. Sin gradientes, generá **8 vectores latentes** muestreados del prior estándar: `z = torch.randn(8, latent_dims, device=device)`.
2. Decodificalos con `vae.decoder(z)` para obtener un batch de imágenes.
3. Mostrá la grilla con `mostrar_grilla(samples.cpu(), num_rows=2, num_cols=4)`.

> **Pista:** llamar `vae.eval()` antes de la inferencia es importante: pone las capas `BatchNorm` en modo determinista (usan las estadísticas acumuladas durante el entrenamiento, no las del batch actual). Sin eso un batch chico puede dar resultados raros.
::::


::::cell{#ej6-code-a type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Parte A: reconstrucción de un batch real ───────────────────────────────
vae.eval()
with torch.no_grad():
    x_batch, _, nombres_batch = next(iter(data_iter))
    x_batch = x_batch[:8].to(device)
    x_hat, _, _ = vae(x_batch)

mostrar_recon(x_batch.cpu(), x_hat.cpu(), titulos=nombres_batch[:8])
```
::::


::::cell{#ej6-code-b type=code role=student-code}
```python
# Tu código aquí
```

```python solution
# ─── Parte B: muestras del prior estándar N(0, I) ───────────────────────────
# Pasamos solo por el decoder: no hay imagen de entrada, solo un código
# latente aleatorio. Lo que veamos es lo que el decoder cree que produce
# una "imagen plausible" cuando le doy un punto típico del prior.
vae.eval()
with torch.no_grad():
    z = torch.randn(8, latent_dims, device=device)
    samples = vae.decoder(z)

mostrar_grilla(samples.cpu(), num_rows=2, num_cols=4)
```
::::


::::cell{#ej6-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Compará visualmente las dos grillas: las **reconstrucciones** (jugadores reales pasados por el VAE) y las **muestras del prior** (`z ~ N(0, I)` decodificado). ¿Cuál de las dos da imágenes más nítidas? ¿Por qué te parece que pasa eso?
::::


::::cell{#ej6-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

Las **reconstrucciones** son notoriamente más nítidas: se reconoce al jugador, se ve el contorno de la cara, los colores de la camiseta. Las **muestras del prior** son más borrosas y "promediadas" — caras genéricas, colores apagados, contornos suaves.

Hay dos razones complementarias:

1. **El decoder fue entrenado a producir esa imagen específica.** En la reconstrucción, el encoder le entrega al decoder un `z` que está muy cerca de `μ(x)`, una región del espacio latente que vio repetidas veces durante el entrenamiento asociada a esa imagen exacta. El decoder responde con la mejor reconstrucción que puede de esa imagen.

2. **El prior cubre regiones poco entrenadas.** Cuando muestreamos `z ~ N(0, I)`, caemos en lugares del latente que no son exactamente donde están concentradas las codificaciones reales (la "agregada" de las posteriors no es exactamente igual al prior). En esas regiones el decoder nunca aprendió a producir una imagen específica, así que devuelve el **promedio** de lo que vio cerca: un retrato "típico" sin detalles.

Más en general, los VAEs producen imágenes más borrosas que las GANs porque la pérdida BCE/MSE penaliza el error píxel a píxel, y ante incertidumbre el decoder elige el promedio (que minimiza el error esperado) en lugar de comprometerse con un detalle concreto. La contraparte es que los VAEs cubren todo el soporte sin colapsar a unos pocos modos, cosa que las GANs sí pueden hacer.
```
::::


::::cell{#secG type=markdown role=section}
---
## Sección G: Interpolación entre jugadores
::::


::::cell{#intro-secG type=markdown role=intro}
Hasta ahora vimos que el VAE reconstruye imágenes y genera muestras desde el prior. Pero la propiedad más interesante del espacio latente continuo aparece cuando **caminamos linealmente entre dos puntos** y observamos qué hace el decoder en cada paso intermedio.

La idea es:

1. Codificar dos jugadores `x_1` y `x_2`. Tomamos la **media** de la posterior, `μ_1` y `μ_2`, y no la muestra `z` con ruido — porque queremos que la trayectoria sea reproducible y determinista.
2. Generar una secuencia de puntos `z_t = (1 - t) μ_1 + t μ_2` con `t ∈ [0, 1]`.
3. Decodificar cada `z_t` y mostrar la secuencia resultante.

Si el VAE aprendió bien, la trayectoria visual es **suave**: cada paso intermedio es un retrato plausible que mezcla rasgos de los dos extremos de manera gradual. Eso solo ocurre si el espacio latente es continuo, que es exactamente lo que la divergencia KL forzó durante el entrenamiento.
::::


::::cell{#ej7-enunciado type=markdown role=enunciado}
### Ejercicio 7 — Interpolación lineal entre dos jugadores

**Objetivo:** Implementar y visualizar una interpolación entre dos imágenes del dataset usando el espacio latente del VAE.

**Enunciado:**

1. Implementá `model_interp(model, idx1, idx2, size=10)` que:
   - Pone el modelo en `eval()`.
   - Toma las dos imágenes de `fifa[idx1]` y `fifa[idx2]`, las manda al `device` y agrega la dimensión de batch (`unsqueeze(0)`).
   - Dentro de un `with torch.no_grad():`:
     - Codifica cada imagen con `model.encoder(...)` y se queda solo con `mu` (descartá `z` y `log_var`).
     - Construye una trayectoria de `size` puntos en el espacio latente con `linear_interpolation(mu_1[0], mu_2[0], size)` (la helper está en el setup).
     - Decodifica todos los puntos en una sola llamada `model.decoder(z_traj)`.
   - Devuelve el batch de imágenes decodificadas.
2. Probá la función con `idx1 = 0`, `idx2 = 1` y `size = 6`. Eso genera **6 puntos** sobre la trayectoria latente: los dos extremos son `μ_inicio` y `μ_fin`, y entre medio quedan 4 puntos intermedios.
3. Mostrá el resultado con `mostrar_interp(imgs, fifa[idx1][0], fifa[idx2][0], fifa.names[idx1], fifa.names[idx2])`. La función pone las imágenes originales del dataset en los extremos y las decodificaciones (incluidas las de los `μ`) entre medio, así se puede comparar lado a lado lo "real" con lo que reconstruye el VAE.

> **Pista:** `mu_1` y `mu_2` tienen shape `(1, latent_dims)` por la dimensión de batch. `linear_interpolation` espera vectores 1D, por eso indexamos con `[0]` para sacar el batch.
::::


::::cell{#ej7-code type=code role=student-code}
```python
# Tu código aquí
```

```python solution
def model_interp(model, idx1, idx2, size=10):
    model.eval()
    img1, _, _ = fifa[idx1]
    img2, _, _ = fifa[idx2]
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    with torch.no_grad():
        # ─── Codificamos a la MEDIA, no a una muestra ──────────────────────
        # El encoder devuelve (z, mu, log_var). z = mu + eps * std incluye
        # ruido aleatorio; mu es la "mejor estimación" determinista del
        # código. Para una interpolación reproducible y suave nos quedamos
        # con mu.
        _, mu_1, _ = model.encoder(img1)
        _, mu_2, _ = model.encoder(img2)

        # ─── Trayectoria lineal en el espacio latente ──────────────────────
        # `size` puntos equiespaciados entre los dos códigos. El primer y
        # último punto coinciden con mu_1 y mu_2.
        z_traj = linear_interpolation(mu_1[0], mu_2[0], size)

        # ─── Decodificamos toda la trayectoria de una sola pasada ──────────
        imgs = model.decoder(z_traj)

    return imgs


# ─── Visualización ──────────────────────────────────────────────────────────
# `imgs` tiene 6 imágenes: las decodificaciones de μ_inicio, 4 puntos
# intermedios y μ_fin. `mostrar_interp` agrega las originales del dataset
# como primer y último panel para comparar.
idx1, idx2 = 0, 1
imgs = model_interp(vae, idx1, idx2, size=6)
img_inicio, _, _ = fifa[idx1]
img_fin,    _, _ = fifa[idx2]
mostrar_interp(imgs, img_inicio, img_fin, fifa.names[idx1], fifa.names[idx2])
```
::::


::::cell{#ej7-pregunta type=markdown role=pregunta}
**Pregunta de análisis:**

Mirá la secuencia de imágenes que produjo la interpolación. La transición entre los dos extremos resulta **suave** (los rasgos cambian gradualmente) en lugar de **abrupta** (se ve el primer jugador, después ruido, después el segundo).

1. ¿Por qué pasa eso? ¿Qué propiedad del espacio latente del VAE lo hace posible?
2. Decidimos interpolar usando `μ` y no `z` muestreado. ¿Qué cambiaría visualmente si interpoláramos con `z` (ruidoso) en lugar de `μ` (determinista)?
::::


::::cell{#ej7-respuesta type=markdown role=student-answer}
*(Escribí tu respuesta acá)*

```markdown solution
**Respuesta a la pregunta de análisis:**

1. **Por qué la transición es suave.**
   El término KL durante el entrenamiento fuerza dos cosas: que las posteriors `q(z|x)` se acerquen al prior `N(0, I)` (centradas, con varianza moderada) y, como consecuencia, que las distribuciones codificadas de imágenes parecidas **se solapen** en el espacio latente. El resultado es que el latente queda **continuo**: no hay regiones grandes vacías entre los códigos. Un punto `z_t = (1-t) μ_1 + t μ_2` a mitad de camino entre `μ_1` y `μ_2` cae en una región que el decoder ya vio (o vio algo muy parecido) y produce un retrato plausible que combina características de los dos extremos.

   En un autoencoder clásico, sin la KL, esa misma trayectoria atravesaría regiones no entrenadas y produciría imágenes ruidosas o deformes en el medio.

2. **Interpolar con `z` en lugar de `μ`.**
   `z = μ + ε · σ` agrega ruido aleatorio en cada llamada al encoder. Si interpoláramos con `z`, los dos extremos de la trayectoria tendrían cada uno un componente aleatorio distinto, y la línea recta entre ellos sería una entre infinitas posibles. Visualmente eso se traduciría en una transición más errática (cada vez que ejecutamos el código sale distinto) y los extremos no coincidirían exactamente con los retratos originales reconstruidos. Usar `μ` da la trayectoria **determinista** y "más representativa" entre los dos jugadores.
```
::::


::::cell{#checklist type=markdown role=checklist}
---
## Antes de entregar

Revisá esta checklist rápida:

- [ ] Ejecuté el laboratorio con la **GPU activa** (`device == "cuda"`).
- [ ] Reinicié el entorno y ejecuté **todas** las celdas de arriba a abajo sin errores (**Entorno de ejecución > Reiniciar y ejecutar todo**).
- [ ] El entrenamiento corrió las 30 épocas y la pérdida promedio bajó consistentemente.
- [ ] Las reconstrucciones del Ej 6A se parecen razonablemente a los originales.
- [ ] La interpolación del Ej 7 muestra una transición visualmente suave entre los dos jugadores.
- [ ] No modifiqué ninguna celda fuera de las de actividad.
::::


::::cell{#footer type=markdown role=footer}
---
## ¡Listo!

Construiste un VAE de punta a punta: bloques convolucionales, encoder con reparametrización, decoder, pérdida ELBO, entrenamiento, y dos formas de explorar el espacio latente aprendido (muestreo del prior e interpolación entre imágenes reales). El truco más importante del lab es probablemente el de la **reparametrización** — separar la fuente de aleatoriedad de los parámetros aprendibles es lo que hace que todo el modelo sea entrenable con descenso de gradiente.

En la próxima clase vamos a ver modelos generativos que abandonan el supuesto gaussiano sobre la posterior y trabajan con dinámicas iterativas: los **modelos de difusión**.
::::
