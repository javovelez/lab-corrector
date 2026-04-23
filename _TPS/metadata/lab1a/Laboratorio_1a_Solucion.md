ej1-enunciado

```python
# ─── Paso 1: tensor de ceros con forma (2, 3, 4) ────────────────────────────
# torch.zeros() crea un tensor con todos sus elementos inicializados a 0.0.
# Es el constructor más común cuando queremos reservar espacio con un valor seguro.
ceros = torch.zeros(2, 3, 4)
print("Tensor de ceros:")
print(ceros)

# ─── Paso 2: tensor aleatorio con la misma forma ─────────────────────────────
# torch.rand_like() genera valores uniformes en [0,1) con EXACTAMENTE
# la misma forma y tipo de dato que el tensor que le pasamos.
# Es más seguro que escribir torch.rand(2, 3, 4) a mano: si cambiamos la
# forma del tensor base, este se actualiza automáticamente.
aleatorio = torch.rand_like(ceros)
print("\nTensor aleatorio:")
print(aleatorio)

# ─── Paso 3: atributos del tensor ────────────────────────────────────────────
print("\n--- Atributos ---")
print(f"Forma (.shape): {aleatorio.shape}")
print(f"Tipo de dato (.dtype): {aleatorio.dtype}")
print(f"Número de dimensiones: {aleatorio.ndim}")
print(f"Total de elementos: {aleatorio.numel()}")
# 2 * 3 * 4 = 24 elementos en total

```

** Respuesta a la pregunta de análisis:**

- `torch.zeros(2, 3, 4)`: crea un tensor **inicializado** con ceros. Sabemos exactamente qué valor tiene cada elemento. Es el constructor a usar cuando el valor inicial importa.
- `torch.empty(2, 3, 4)`: **reserva memoria** para el tensor pero no inicializa los valores. Lo que aparece son los bytes que ya estaban en esa zona de memoria (basura). Es marginalmente más rápido que `zeros`, pero solo se usa cuando *inmediatamente después* vamos a asignar todos los valores (por ejemplo, en implementaciones de capas a mano), ya que trabajar con valores no inicializados puede generar resultados imprevisibles.

ej2-enunciado

```python
torch.manual_seed(0) # fijamos semilla para reproducibilidad

# ─── float32: el tipo por defecto en PyTorch ──────────────────────────────────
# float32 usa 4 bytes por elemento y tiene ~7 dígitos decimales de precisión.
t_f32 = torch.rand(3, 3)
print(f"float32: {t_f32.dtype}")
print(t_f32)

# ─── Convertir a float16 ──────────────────────────────────────────────────────
# float16 usa 2 bytes por elemento y tiene solo ~3-4 dígitos de precisión.
# El método .to() devuelve un NUEVO tensor con el tipo solicitado.
t_f16 = t_f32.to(torch.float16)
print(f"\nfloat16: {t_f16.dtype}")
print(t_f16)

# ─── Diferencia al convertir de vuelta a float32 ─────────────────────────────
# float16 → float32 recupera el tensor en precisión alta,
# pero los valores ya fueron redondeados al pasar por float16.
t_f16_back = t_f16.to(torch.float32)
diferencia = t_f32 - t_f16_back
print(f"\nDiferencia (error de redondeo):")
print(diferencia)
print(f"Error máximo: {diferencia.abs().max().item():.6f}")
```

**Respuesta a la pregunta de análisis:**

`float16` ocupa la mitad de memoria que `float32`, lo que permite procesar lotes más grandes de datos. Sin embargo, su rango numérico es mucho más estrecho (máximo ~65000 vs ~3.4×10³⁸) y tiene muchos menos dígitos de precisión.

El problema en entrenamiento es que los **gradientes** suelen ser números muy pequeños (del orden de 10⁻⁴ o menores). Con `float16`, esos gradientes producen *underflow* a cero, lo que hace que el modelo deje de aprender. Por esta razón, en la práctica se usa **entrenamiento en precisión mixta** (AMP): los pesos se almacenan en `float32` pero los cálculos intermedios en `float16`, con un *loss scaler* que evita el underflow.

ej3-enunciado

```python
# ─── Crear el vector [1, 2, ..., 12] ─────────────────────────────────────────
# torch.arange(start, stop) genera números de start hasta stop-1 inclusive.
x = torch.arange(1, 13)
print("Vector original:", x)

# ─── Reorganizar en (3, 4): 3 filas × 4 columnas ─────────────────────────────
# reshape() devuelve un tensor con la nueva forma.
# PyTorch llena en orden C (row-major): primero completa la fila 0, luego la 1, etc.
mat_3x4 = x.reshape(3, 4)
print("\nMatriz (3, 4):")
print(mat_3x4)
# El número 8 está en la fila 1, columna 3 (índices base 0)
print(f"El número 8 está en: fila={1}, columna={3} → mat_3x4[1, 3] = {mat_3x4[1, 3].item()}")

# ─── Comparar con (4, 3): 4 filas × 3 columnas ───────────────────────────────
mat_4x3 = x.reshape(4, 3)
print("\nMatriz (4, 3):")
print(mat_4x3)
```

** Respuesta a la pregunta de análisis:**

PyTorch llena los elementos en **orden C** (también llamado *row-major* u *orden lexicográfico*). Esto significa que la última dimensión varía más rápido: primero se completa toda la fila 0, luego la fila 1, y así sucesivamente. Es el mismo orden que usa Python para listas anidadas y NumPy por defecto.

El orden opuesto se llama orden F (*column-major* o *Fortran order*), donde la primera dimensión varía más rápido. MATLAB y Fortran usan este orden.

4c5e723d

```python
# ─── Crear el vector [1, 2, ..., 24] ─────────────────────────────────────────
x_4d = torch.arange(1, 25)
print("Vector original:\n", x_4d)

# ─── Reorganizar en (2, 3, 2, 2) ─────────────────────────────
tensor_4d = x_4d.reshape(2, 3, 2, 2)
print("\nTensor 4D:\n", tensor_4d)

# ─── Extraer posición específica ───────────────────────────────
# Lote 1 (índice 1), Canal 2 (índice 2), Fila 0 (índice 0), Columna 1 (índice 1)
val = tensor_4d[1, 2, 0, 1]
print(f"\nEl valor en [1, 2, 0, 1] es: {val.item()}")

```

** Respuesta a la pregunta de análisis:**

El tensor se llena completando las dimensiones de **derecha a izquierda (orden C)**. Esto significa que la dimensión que varía más **rápido es la última (las columnas)**. Los elementos adyacentes en la memoria corresponden a la misma fila.

El orden lógico de llenado es:
1. Variar **columnas** hasta completar una fila.
2. Variar **filas** hasta completar un canal (imagen o matriz 2D).
3. Variar **canales** hasta completar todas las capas de una imagen (ej: R, G, B).
4. Variar **lotes** (batches). Es la dimensión que varía más **lento**.


ej4-enunciado

```python
# ─── Vector [0, 1, ..., 23] ───────────────────────────────────────────────────
y = torch.arange(24)
print("Vector original:", y)

# ─── Reorganizar en (2, 3, 4) usando view() ───────────────────────────────────
# view() es similar a reshape() pero garantiza que no copia datos.
# Si el tensor no es contiguo en memoria, view() falla y hay que usar reshape().
cubo = y.view(2, 3, 4)
print("\nTensor cúbico (2, 3, 4):")
print(cubo)

# ─── Extraer el primer bloque (índice 0) ──────────────────────────────────────
# Al indexar la primera dimensión, obtenemos una "rodaja" del tensor.
primer_bloque = cubo[0]
print(f"\nPrimer bloque (cubo[0]), forma {primer_bloque.shape}:")
print(primer_bloque)

# ─── Posición del número 15 ───────────────────────────────────────────────────
# Con orden C en forma (2, 3, 4):
# - Primer bloque: índices 0-11 → cubo[0]
# - Segundo bloque: índices 12-23 → cubo[1]
# - Dentro del bloque 1: 15 - 12 = 3 → fila 3//4 = 0, columna 3%4 = 3
print(f"\nEl número 15 está en: cubo[1, 0, 3] = {cubo[1, 0, 3].item()}")
```

ej5-enunciado

```python
# ─── Parte A: la vista comparte memoria ──────────────────────────────────────
# Recreamos el vector y su vista (cubo) del ejercicio anterior
y = torch.arange(24, dtype=torch.float32)
cubo = y.view(2, 3, 4)

print("Primer elemento del cubo ANTES de modificar y:", cubo[0, 0, 0].item())

# Modificamos el primer elemento del vector ORIGINAL
y[0] = -99
# Como 'cubo' es una vista de 'y', apuntan a la misma zona de memoria.
# Cualquier cambio en 'y' se refleja inmediatamente en 'cubo'.
print("Primer elemento del cubo DESPUÉS de modificar y:", cubo[0, 0, 0].item())
print("\nTensor cubo completo (notar el -99 en cubo[0,0,0]):")
print(cubo)
```

```python
# ─── Parte B: copia real con .clone() ────────────────────────────────────────
# clone() crea un tensor completamente nuevo en una zona de memoria diferente.
# A partir de este punto, cubo_copia y cubo son independientes.
cubo_copia = cubo.clone()

# Modificamos el segundo elemento del vector original
y[1] = -999
print("Segundo elemento de cubo (vista, sí cambia):", cubo[0, 0, 1].item())
print("Segundo elemento de cubo_copia (copia, NO cambia):", cubo_copia[0, 0, 1].item())
```

** Respuesta a la pregunta de análisis:**

**Ventaja:** Al no copiar datos, operaciones como `view` y `reshape` son instantáneas y consumen cero memoria adicional, sin importar el tamaño del tensor. En deep learning, donde se trabaja con tensores de cientos de millones de elementos, esto es fundamental para el rendimiento.

**Cuándo es un problema:** Cuando necesitamos modificar un tensor derivado sin afectar el original (o viceversa). Esto ocurre frecuentemente al calcular augmentations de datos o cuando se procesan tensores en paralelo. La solución es simple: usar `.clone()` antes de realizar la modificación.

ej6-enunciado

```python
# ─── Preparar tensores ────────────────────────────────────────────────────────
A = torch.ones( 3, 4)
v = torch.tensor([10., 20., 30., 40.])

print(f"Forma de A: {A.shape}")
print(f"Forma de v: {v.shape}")

# ─── Multiplicación con broadcasting ─────────────────────────────────────────
# PyTorch compara de derecha a izquierda:
# A: (3, 4) → dimensión 1: 4
# v: (4,) → dimensión 0: 4 ← coinciden ✓
# → dimensión izquierda de v: NO EXISTE → se trata como 1
# PyTorch "expande" v virtualmente de (1, 4) a (3, 4), multiplicando
# el mismo vector v por cada una de las 3 filas de A.
resultado = A * v
print(f"\nResultado (forma: {resultado.shape}):")
print(resultado)
```

** Respuesta a la pregunta de análisis:**

```
A: 3 4
v: ← 4 (la dimensión izquierda no existe → se trata como 1)
```

La dimensión faltante de `v` se trata como `1`, y como `1` es compatible con cualquier número mediante broadcasting, se "replica" 3 veces. La última dimensión `4 == 4` coincide exactamente. El resultado final tiene forma `(3, 4)`: el máximo de cada dimensión entre los dos tensores.

ej7-enunciado

```python
import torch
A = torch.arange(24).reshape(3, 2, 4)
v = torch.tensor([6, 5, 4, 3, 2, 1], dtype=torch.float32)

print("Tensor A (3, 2, 4):")
print(A)
print(f"\nVector v (forma original): {v.shape} → {v}")

# ─── Análisis del problema ────────────────────────────────────────────────────
# A tiene forma (3, 2, 4): 3 bloques de 2 filas con 4 columnas cada una.
# Queremos multiplicar cada una de las 6 filas (3×2) por un escalar de v.
# Para que el broadcasting alinee v con las dos primeras dimensiones de A,
# le damos a v la forma (3, 2, 1):
# - Dimensión 0 (bloques): 3 == 3 ✓
# - Dimensión 1 (filas): 2 == 2 ✓
# - Dimensión 2 (columnas): 1 → se expande a 4 por broadcasting ✓

v_reshape = v.reshape(3, 2, 1)
print(f"\nVector v reorganizado (3, 2, 1):\n{v_reshape}")

resultado = A * v_reshape
print(f"\nResultado A * v (forma: {resultado.shape}):")
print(resultado)

# Verificación: la primera fila del primer bloque debería ser [0,1,2,3]*6 = [0,6,12,18]
print("\nVerificación fila 0, bloque 0:", resultado[0, 0].tolist(), "(esperado: [0, 6, 12, 18])")
```

ej8-enunciado

```python
import torch
torch.manual_seed(42)
X = torch.rand(32, 3, 28, 28)
medias = torch.tensor([0.5, 0.4, 0.3])

# ─── Paso 1: intentar la resta directa ────────────────────────────────────────
# El error que aparece es algo como:
# RuntimeError: The size of tensor a (28) must match the size of tensor b (3)
# at non-singleton dimension 3
# PyTorch compara de derecha a izquierda:
# X: 32 3 28 28
# medias: 3 ← PyTorch intenta alinear el 3 con el 28 del ancho → ERROR
try:
 X - medias
except RuntimeError as e:
 print("Error al restar directamente:")
 print(e)

# ─── Solución: reorganizar medias con dimensiones de tamaño 1 ─────────────────
# Necesitamos que 'medias' tenga forma (1, 3, 1, 1) para que:
# X: 32 3 28 28
# medias: 1 3 1 1 ← broadcasting expande el 1 en todos los ejes ✓
medias_reshape = medias.view(1, 3, 1, 1)
# Equivalentemente: medias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

print(f"\nForma de medias reorganizado: {medias_reshape.shape}")

X_normalizado = X - medias_reshape
print(f"Forma del resultado: {X_normalizado.shape}")

# Verificación: la media de cada canal debería ser ≈ 0 después de normalizar
# (aproximado porque X fue generado aleatoriamente, no con media exacta 0.5/0.4/0.3)
for c, nombre in enumerate(['Rojo', 'Verde', 'Azul']):
 media_canal = X_normalizado[:, c, :, :].mean().item()
 print(f" Canal {nombre}: media después de normalizar = {media_canal:.4f}")
```

** Respuesta a la pregunta de análisis:**

```
X: 32 3 28 28
medias reshape: 1 3 1 1
─────────────────────────────────
Resultado: 32 3 28 28 ← max(32,1), max(3,3), max(28,1), max(28,1)
```

Las dimensiones con `1` se expanden virtualmente al tamaño del tensor opuesto. PyTorch no copia datos en memoria: simplemente actúa como si la misma fila se repitiera 32 veces (dim 0) y como si el mismo escalar se aplicara a todos los píxeles (dims 2 y 3).

ej9-enunciado

```python
x = torch.arange(1, 6) # [1, 2, 3, 4, 5] — forma: (5,)
y = torch.arange(10, 50, 10) # [10, 20, 30, 40] — forma: (4,)

# ─── Generar la cuadrícula en una línea ───────────────────────────────────────
# Para que el broadcasting expanda en dos ejes al mismo tiempo:
#
# x.unsqueeze(1) → forma (5, 1) → se expande hacia las columnas (→)
# y → forma (4,) equivale a (1, 4)
# → se expande hacia las filas (↓)
#
# Al sumarlos:
# x_col: (5, 1) → se replica 4 veces a la derecha
# y: (1, 4) → se replica 5 veces hacia abajo
# M: (5, 4) → cada posición (i,j) = x[i] + y[j]
M = x.unsqueeze(1) + y

print(f"Forma de M: {M.shape}")
print("\nMatriz M (M[i,j] = x[i] + y[j]):")
print(M)

# Verificación explícita de algunas posiciones
print(f"\nM[0,0] = x[0]+y[0] = 1+10 = {M[0,0].item()} (esperado: 11)")
print(f"M[4,3] = x[4]+y[3] = 5+40 = {M[4,3].item()} (esperado: 45)")
```

ej10-enunciado

```python
import torch

# ─── Paso 1: definir x con seguimiento de gradiente ──────────────────────────
# requires_grad=True le indica a PyTorch: "rastreá todas las operaciones
# sobre este tensor para poder calcular gradientes después".
x = torch.tensor(2.0, requires_grad=True)

# ─── Paso 2: definir la función y = 3x² + 4x + 2 ────────────────────────────
# A medida que ejecutamos estas operaciones, PyTorch construye
# el grafo computacional internamente.
y = 3 * x**2 + 4 * x + 2
print(f"y = 3*{x.item()}² + 4*{x.item()} + 2 = {y.item()}")

# ─── Paso 3: backward — propagación hacia atrás ───────────────────────────────
# .backward() recorre el grafo de derecha a izquierda aplicando la regla de la
# cadena y acumula los gradientes en el atributo .grad de cada tensor hoja.
y.backward()
print(f"\nGradiente calculado por autograd: dy/dx = {x.grad.item()}")

# ─── Paso 4: verificar con la derivada analítica ──────────────────────────────
# dy/dx = 6x + 4 → evaluada en x=2: 6*2 + 4 = 16
derivada_analitica = 6 * 2 + 4
print(f"Derivada analítica: 6*2 + 4 = {derivada_analitica}")
print(f"¿Coinciden? {x.grad.item() == derivada_analitica}")
```

```python
# ─── Paso 5: intentar backward() dos veces ────────────────────────────────────
# Por defecto, PyTorch libera el grafo computacional después de la primera
# llamada a backward() para ahorrar memoria. Si intentamos llamarlo de nuevo:
x2 = torch.tensor(2.0, requires_grad=True)
y2 = 3 * x2**2 + 4 * x2 + 2

y2.backward() # primera llamada: funciona bien
print(f"Primer backward: x2.grad = {x2.grad.item()}")

try:
 y2.backward() # segunda llamada: el grafo ya fue liberado → error
except RuntimeError as e:
 print(f"\nError al llamar backward() dos veces:\n{e}")

# ─── Solución: retain_graph=True ─────────────────────────────────────────────
# Le indica a PyTorch que NO libere el grafo después del primer backward.
x3 = torch.tensor(2.0, requires_grad=True)
y3 = 3 * x3**2 + 4 * x3 + 2

y3.backward(retain_graph=True) # primera llamada, grafo conservado
y3.backward() # segunda llamada, ahora funciona
# Nota: los gradientes se ACUMULAN, así que x3.grad será el doble del valor correcto
print(f"\nCon retain_graph=True, x3.grad = {x3.grad.item()} (acumulado: 16+16=32)")
```

** Respuesta a la pregunta de análisis:**

**1. Grafo computacional:** 
A medida que realizamos operaciones sobre tensores con `requires_grad=True`, PyTorch construye dinámicamente un **grafo acíclico dirigido (DAG)**. Cada nodo del grafo representa una operación (suma, multiplicación, etc.), y las aristas conectan los tensores de entrada con los de salida. Este grafo permite que `.backward()` aplique automáticamente la **regla de la cadena** (backpropagation) para calcular cómo cada tensor de entrada contribuyó al resultado final.

Por defecto, el grafo se **libera de memoria inmediatamente** después de `backward()`. Esto es correcto en el 99% de los casos: una vez calculados los gradientes, el grafo ya no se necesita y mantenerlo sería un desperdicio de RAM.

**2. `retain_graph=True`:** 
Se usa cuando necesitamos llamar a `.backward()` más de una vez sobre el mismo grafo. Esto ocurre en arquitecturas como las **redes generativas adversarias (GANs)**, donde los gradientes del generador y del discriminador se calculan en pasos separados, pero comparten partes del mismo grafo computacional.

o24zqxqq0lc

```python
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class PlatesDataSet(Dataset):
    def __init__(self, csv_file='./data/plates/plates.csv', root_dir='./data/plates',
                 mode='train', transform=None):
        """
        Parámetros:
        csv_file (str): ruta al archivo CSV con las anotaciones
        root_dir (str): directorio raíz que contiene las imágenes
        mode (str): partición a cargar ('train', 'test' o 'val')
        transform: transformaciones opcionales a aplicar a las imágenes
        """
        super().__init__()
        # ─── Cargar el CSV y filtrar por partición ────────────────────────────────
        # reset_index(drop=True) renumera los índices desde 0 para que
        # el acceso por posición en __getitem__ sea coherente.
        df = pd.read_csv(csv_file)
        self.data = df[df['data set'] == mode].reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform

        # ─── Cachear columnas como listas para acceso O(1) ───────────────────────
        # Acceder a un DataFrame por índice es más lento que a una lista Python.
        # Como __getitem__ se llama en cada iteración del entrenamiento,
        # esta optimización tiene un impacto real en los tiempos de carga.
        self.labels = self.data['class id'].tolist()
        self.text_labels = self.data['labels'].tolist()
        self.filepaths = self.data['filepaths'].tolist()

    def __len__(self):
        """Retorna la cantidad de imágenes en la partición."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retorna la imagen y la etiqueta en la posición index.

        Retorna:
        tuple: (imagen, etiqueta_numérica)
        """
        # ─── Construir la ruta completa ───────────────────────────────────────────
        # os.path.join() concatena rutas correctamente en cualquier sistema operativo.
        img_path = os.path.join(self.root_dir, self.filepaths[index])

        # ─── Cargar la imagen ─────────────────────────────────────────────────────
        # PIL.Image.open() es el cargador estándar para PyTorch.
        # .convert('RGB') garantiza exactamente 3 canales, incluso si la imagen
        # original es en escala de grises o tiene canal alfa (RGBA).
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, index):
        """Retorna el nombre del estado para el elemento en index."""
        return self.text_labels[index]
```

```python
set(dataset_train.text_labels)
```

** Respuesta a la pregunta de análisis:**

La separación entre `Dataset` y `DataLoader` es un ejemplo del principio de responsabilidad única: cada clase hace una sola cosa.

- `Dataset` solo sabe cómo *acceder* a un elemento por su índice y cuántos elementos hay en total. No sabe nada de lotes, mezcla ni paralelismo.
- `DataLoader` solo sabe cómo *iterar* sobre un `Dataset`: agrupa los elementos en lotes, los mezcla, y puede cargarlos en paralelo con múltiples workers.

La ventaja práctica es que para cambiar el tamaño de lote, activar el mezclado o agregar workers de carga paralela, no hay que modificar nada de la clase de datos. Simplemente se instancia un nuevo `DataLoader` con los parámetros deseados: `DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)`. El mismo `PlatesDataSet` funciona sin cambios.