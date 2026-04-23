# PyTorch Cheatsheet

Basado en el notebook `01_02_pytorch.ipynb`.

## 1. Creación de Tensores

Los tensores son la estructura de datos principal en PyTorch.

```python
import torch

# Crear un tensor no inicializado de 3x4
x = torch.empty(3, 4)

# Crear tensor con ceros o unos
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)

# Crear tensor con valores aleatorios (0 a 1)
random_tensor = torch.rand(2, 3)

# Crear un tensor a partir de una lista/tupla de Python
t = torch.tensor([[3.14, 2.71], [1.61, 0.007]])

# Crear tensores con la misma forma que otro tensor
empty_like = torch.empty_like(x)
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x)
```

**Atributos importantes:**
- `x.shape`: Devuelve la forma (dimensiones) del tensor.

### Semillas Aleatorias
Fijar la semilla para que los números aleatorios sean reproducibles.
```python
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
```

## 2. Tipos de Datos (dtypes)

Se pueden definir al crear el tensor o convertirlos posteriormente con el método `.to()`.

```python
# Definir al crear
a = torch.ones((2, 3), dtype=torch.int16)

# Modificar el tipo de dato
b = torch.rand((2, 3), dtype=torch.float64)
c = b.to(torch.int32)
```
Tipos comunes: `torch.bool`, `torch.int8`, `torch.int32`, `torch.int64`, `torch.float`, `torch.double`.

## 3. Matemáticas y Operaciones

Las operaciones aritméticas básicas (`+`, `-`, `*`, `/`, `**`) se aplican **elemento a elemento**.

```python
a = torch.ones(2, 2)
b = torch.ones(2, 2) * 2

c = a + b    # Suma
d = b ** 2   # Potenciación
```

**Operaciones comunes de PyTorch:**
```python
torch.abs(a)           # Valor absoluto
torch.ceil(a)          # Redondeo hacia arriba
torch.sin(angles)      # Seno
torch.bitwise_xor(b,c) # XOR lógico
torch.eq(d, e)         # Comparación (igualdad)  
torch.max(d)           # Valor máximo
torch.mean(d)          # Promedio
torch.matmul(m1, m2)   # Multiplicación de matrices (producto punto)
```

### Operaciones In-situ (In-place)
Añadir un guión bajo `_` altera el tensor original en lugar de devolver uno nuevo, ahorrando memoria.
```python
a = torch.ones(2, 2)
b = torch.rand(2, 2)
a.add_(b)  # 'a' se modifica y ahora contiene a + b
```

### Broadcasting
El *broadcasting* es la capacidad de PyTorch de operar matemáticamente sobre tensores de **distintas formas**, sin copiar datos en memoria. PyTorch compara las dimensiones de los dos tensores **de derecha a izquierda** y aplica las siguientes reglas para cada par de dimensiones:

| Situación | Resultado |
|---|---|
| Ambas dimensiones son iguales | Se opera normalmente |
| Una de las dimensiones es `1` | Se "expande" virtualmente para coincidir |
| Una dimensión no existe | Se asume que es `1` y se expande |
| Distintas y ninguna es `1` | ❌ Falla (RuntimeError) |

```python
A = torch.ones(3, 4)
v = torch.tensor([10, 20, 30, 40]) # Forma: (4,)

# Broadcasting en acción: 'v' se resta a cada fila de 'A'
# A: 3 x 4
# v: 1 x 4 (el 1 se asume por la dimensión faltante)
resultado = A - v 
```

## 4. Manipulación de la Forma del Tensor

- **`unsqueeze(dim)`**: Añade una dimensión de longitud 1 en la posición `dim`.
  ```python
  a = torch.rand(3, 226, 226)
  b = a.unsqueeze(0) # Forma: (1, 3, 226, 226)
  ```
- **`squeeze(dim)`**: Elimina una dimensión de longitud 1 en la posición `dim`.
  ```python
  c = b.squeeze(0) # Forma vuelve a: (3, 226, 226)
  ```
- **`reshape(shape)`**: Cambia la forma manteniendo el número total de elementos. Devuelve una vista si es posible.
  ```python
  output3d = torch.rand(6, 20, 20)
  input1d = output3d.reshape(6 * 20 * 20)
  ```

### Copiar tensores
Usar `= ` crea una referencia al mismo objeto en memoria. Usa `clone()` para una copia independiente:
```python
b = a.clone()
```

## 5. Puente con NumPy

Intercambiar datos entre tensores PyTorch y arreglos NumPy. **Comparten la memoria subyacente**: si modificas uno, cambia el otro.

```python
import numpy as np

# NumPy a PyTorch
numpy_array = np.ones((2, 3))
pytorch_tensor = torch.from_numpy(numpy_array)

# PyTorch a NumPy
numpy_rand = pytorch_tensor.numpy()
```

## 6. Datasets y DataLoaders

PyTorch proporciona dos clases principales para manejar datos, lo que permite desacoplar la lógica de procesamiento de los datos del propio modelo:

1. **`Dataset`**: Almacena los ejemplos y sus correspondientes etiquetas (o targets). Para usar datos propios, se debe crear una clase que herede de `Dataset` y sobreescribir tres métodos principales:
   - `__init__`: Para inicializar variables, leer rutas, archivos csv, etc.
   - `__len__`: Devuelve la cantidad total de ejemplos en el conjunto de datos.
   - `__getitem__`: Carga y devuelve un ejemplo y su etiqueta dado un índice.

2. **`DataLoader`**: Envuelve al `Dataset` en un iterador. Su principal función es facilitar el trabajo de entrenamiento al:
   - Entregar los datos estructurados en "minilotes" o *minibatches* (controlado por el parámetro `batch_size`).
   - Mezclar automáticamente los datos del conjunto de entrenamiento en cada iteración (parámetro `shuffle=True`) para evitar el sobreajuste.
   - Acelerar la recuperación a través del multiprocesamiento.

**Ejemplo de uso de DataLoader:**
```python
from torch.utils.data import DataLoader

# Se necesita una instancia de un Dataset (por ejemplo "training_data")
# Crea un iterador que entrega lotes de 64 muestras mezcladas aleatoriamente
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Extraer un lote manualmente
train_features, train_labels = next(iter(train_dataloader))
```

## 7. Fundamentos de Autograd

Autograd rastrea dinámicamente las operaciones de un tensor para calcular automáticamente los gradientes usando retropropagación (backpropagation).

- **Activar cálculo de gradientes:**
  ```python
  a = torch.ones(2, 3, requires_grad=True)
  ```
- **Calcular derivadas (backpropagation):**
  Llama a `.backward()` en un tensor escalar (como la pérdida). El gradiente se guardará en el atributo `.grad` de los tensores de origen.
  ```python
  x = torch.tensor([2.0, 3.0], requires_grad=True)
  
  # Operación (ej. cálculo del error)
  y = (x ** 2).sum() 
  
  # Calcular gradientes automáticamente
  y.backward()
  
  # Imprime las derivadas parciales respecto a 'x' (dy/dx = 2x)
  print(x.grad) # tensor([4., 6.])
  ```
- **Desactivar Autograd:** 
  Útil durante la evaluación o para ahorrar memoria.
  - Explícitamente: `a.requires_grad = False`
  - Bloque temporal: `with torch.no_grad():`
  - Decorador: `@torch.no_grad()`
  - Separar un tensor del historial: `y = x.detach()`

**⚠️ Cuidado con las operaciones in-situ (`_`):** 
No utilices operaciones in-situ en variables que requieran `autograd`, ya que destruirán la información necesaria para derivar.
