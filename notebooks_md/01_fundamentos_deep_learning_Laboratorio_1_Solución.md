![Imgur](https://i.imgur.com/acSOZRh.png)

# Laboratorio N° 1

# Parte 1: Introducción

### Ejercicio 1
Generar un tensor llamado x cuyo contenido sea el siguiente:

![Imgur](https://i.imgur.com/rfaKGUG.png)

El tensor debe construirse sin escribir manualmente los valores.


```python
#inserte su código aquí
import torch

x = torch.arange(0, 60, 2).view(2, 3, 5)
print(x)


```

    tensor([[[ 0,  2,  4,  6,  8],
             [10, 12, 14, 16, 18],
             [20, 22, 24, 26, 28]],
    
            [[30, 32, 34, 36, 38],
             [40, 42, 44, 46, 48],
             [50, 52, 54, 56, 58]]])


### Ejercicio 2
Tienes un tensor $A$ con forma (3, 2, 4), y un vector v con 6 elementos. Tu tarea es hacer los cambios necesarios para multiplicar $A$ por $v$ de manera que:


*   Cada fila de $A$ se multiplique por el elemento correspondiente de $v$.



```python
import torch

A = torch.arange(24).reshape(3, 2, 4)  # Tensor base
v = torch.tensor([6,5,4,3,2,1])  # Vector

print(A)
print(v)

```

    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7]],
    
            [[ 8,  9, 10, 11],
             [12, 13, 14, 15]],
    
            [[16, 17, 18, 19],
             [20, 21, 22, 23]]])
    tensor([6, 5, 4, 3, 2, 1])



#### **Pistas:**
1. Necesitas hacer que `v` tenga una forma compatible con `A`.
2. Usa `reshape()` o `view()` para reorganizar `v`.
3. Piensa en `broadcasting`: PyTorch expandirá dimensiones automáticamente si están en el lugar correcto.



```python
# Inserta aquí los cambios necesarios para poder multiplicar A por v
v = v.reshape(3,2,1)
print(A*v)
```

    tensor([[[ 0,  6, 12, 18],
             [20, 25, 30, 35]],
    
            [[32, 36, 40, 44],
             [36, 39, 42, 45]],
    
            [[32, 34, 36, 38],
             [20, 21, 22, 23]]])


### **Ejercicio 3: Filtrando Imágenes de Bolsos del Dataset FashionMNIST**

**Objetivo:**  
Descargar el dataset `FashionMNIST`, filtrar las imágenes correspondientes a bolsos (label 8) y almacenarlas en un tensor. Luego, visualizar algunas de esas imágenes para comprobar que se ha realizado el filtrado correctamente.

**Instrucciones:**

1. Usa `torchvision` para descargar el dataset `FashionMNIST` y aplicar la transformación adecuada para convertir las imágenes a tensores.
2. Filtra las imágenes que tengan la etiqueta 8, que corresponde a los bolsos.
3. Apila esas imágenes en un solo tensor usando `torch.stack`.
4. Implementa una función que permita visualizar algunas de las imágenes de bolsos almacenadas en el tensor.





```python
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transformación para convertir las imágenes a tensores
transform = transforms.Compose([transforms.ToTensor()])

# Descargar el dataset FashionMNIST
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Filtrar las imágenes de bolsos (label 8)
def obtener_bolsos(dataset):
    bolsos = []
    for img, label in dataset:
        if label == 8:  # Label 8 corresponde a "Bolsos"
            bolsos.append(img)
    return torch.stack(bolsos)

# Función para visualizar algunas imágenes de bolsos
def visualizar_bolsos(tensor_bolsos, num_imagenes=5):
    fig, axes = plt.subplots(1, num_imagenes, figsize=(15, 15))
    for i in range(num_imagenes):
        axes[i].imshow(tensor_bolsos[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.show()

# Obtener las imágenes de bolsos
bolsos_tensor = obtener_bolsos(dataset)

# Mostrar la forma del tensor resultante
print(bolsos_tensor.shape)

# Visualizar algunas imágenes de bolsos
visualizar_bolsos(bolsos_tensor)


```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz


    100%|██████████| 26.4M/26.4M [00:00<00:00, 114MB/s]


    Extracting ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz


    100%|██████████| 29.5k/29.5k [00:00<00:00, 4.69MB/s]


    Extracting ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz


    100%|██████████| 4.42M/4.42M [00:00<00:00, 56.4MB/s]


    Extracting ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz


    100%|██████████| 5.15k/5.15k [00:00<00:00, 17.3MB/s]

    Extracting ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw
    


    


    torch.Size([6000, 1, 28, 28])



    
![png](01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_files/01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_10_11.png)
    



### Ejercicio 4: Creación de un Dataset Personalizado para Reconocimiento de Placas de Matrícula

**Objetivo**

Crear un dataset personalizado en PyTorch para trabajar con imágenes de placas de matrícula de diferentes estados de EE.UU.




```python
## descarga el dataset en la carpeta new plates
!gdown https://drive.google.com/uc?id=1FMkstj2JgQOySU0D6mH7Dtr-hts2cqSD
!unzip plates.zip
```

**Datos**

El archivo CSV (`plates.csv`) tiene la siguiente estructura:
- `class id`: Identificador numérico de la clase (estado)
- `filepaths`: Ruta a las imágenes de las placas
- `labels`: Nombre del estado al que pertenece la placa (por ejemplo, "ALABAMA")
- `data set`: Indica si la imagen pertenece al conjunto de entrenamiento ("train"), prueba ("test") o validación ("val")

**Requisitos**
1. Implementar una subclase de `torch.utils.data.Dataset` llamada `PlatesDataSet`
2. La clase debe implementar los siguientes métodos:
   - `__init__(self, mode)`: Constructor que inicializa el dataset y filtra las imágenes según el modo ('train', 'test', etc.)
   - `__len__(self)`: Retorna la cantidad de elementos en el dataset
   - `__getitem__(self, index)`: Carga la imagen correspondiente al índice dado y retorna un par (imagen, etiqueta)

**Implementación**

El dataset debe:
1. Cargar el archivo CSV utilizando pandas
2. Filtrar las entradas según el modo especificado (train/test)
3. Almacenar las etiquetas y rutas de las imágenes
4. Implementar el método `__getitem__` para cargar las imágenes bajo demanda



```python
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as transforms

class PlatesDataSet(Dataset):
    def __init__(self, csv_file='/content/new plates/plates.csv', root_dir='/content/new plates', mode='train', transform=None):
        """
        Inicializa el dataset de placas de matrícula.

        Args:
            csv_file (str): Ruta al archivo CSV con las anotaciones.
            root_dir (str): Directorio con todas las imágenes.
            mode (str): 'train' o 'test' para filtrar las imágenes.
            transform (callable, opcional): Transformaciones opcionales a aplicar a las imágenes.
        """
        # Leer el archivo CSV
        self.data_plates = pd.read_csv(csv_file)

        # Filtrar por modo (train/test)
        self.data = self.data_plates[self.data_plates['data set'] == mode]

        # Guardar las etiquetas y las rutas
        self.labels = self.data.iloc[:, 0]  # class id
        self.paths = self.data.iloc[:, 1]   # filepaths

        # Labels textuales (nombres de los estados)
        self.text_labels = self.data.iloc[:, 2]  # labels

        self.root_dir = root_dir
        self.transform = transform

        # Para debugging
        print(f"Dataset cargado con {len(self.labels)} imágenes en modo '{mode}'")

    def __len__(self):
        """
        Retorna el número de imágenes en el dataset.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Retorna una imagen y su etiqueta correspondiente.

        Args:
            index (int): Índice del elemento a recuperar.

        Returns:
            tuple: (imagen, etiqueta)
        """
        # Construir la ruta completa a la imagen
        img_path = os.path.join(self.root_dir, self.paths[index])

        # Cargar la imagen
        try:
            # Intentar usando torchvision.io.read_image
            image = read_image(img_path)

            # Convertir a float y normalizar a [0, 1]
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

        except Exception as e:
            # Fallback a PIL si read_image falla
            print(f"Error al leer imagen con torchvision: {e}")
            image = Image.open(img_path).convert('RGB')

            # Convertir PIL a tensor
            if self.transform is None:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                image = transform(image)

        # Aplicar transformaciones si existen
        if self.transform:
            image = self.transform(image)

        # Obtener la etiqueta
        label = self.labels[index]

        return image, label

    def get_class_name(self, index):
        """
        Método adicional para obtener el nombre del estado correspondiente a un índice.

        Args:
            index (int): Índice del elemento.

        Returns:
            str: Nombre del estado (ej. 'ALABAMA')
        """
        return self.text_labels[index]
```

Una vez implementada tu solución del ejercicio, podés correr las siguientes celdas para verificar que tu dataset funciona. Esta herramienta grafica 5 ejemplos de placas del estado seleccionado, permitiendo verificar visualmente que la carga de imágenes y etiquetas funciona correctamente.


```python
# @title Función para mostrar los ejemplos del dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from torch.utils.data import DataLoader

def visualize_state_plates(dataset, state_name, num_examples=5, figsize=(15, 3)):
    """
    Visualiza ejemplos de placas de matrícula de un estado específico.

    Args:
        dataset (PlatesDataSet): Dataset de placas de matrícula
        state_name (str): Nombre del estado (ej. 'ALABAMA')
        num_examples (int): Número de ejemplos a visualizar
        figsize (tuple): Tamaño de la figura

    Returns:
        None: Muestra la visualización directamente
    """
    # Filtrar índices que corresponden al estado solicitado
    state_indices = []

    # Buscar las muestras que corresponden al estado requerido
    for i in range(len(dataset)):
        if dataset.get_class_name(i).upper() == state_name.upper():
            state_indices.append(i)

    if not state_indices:
        print(f"No se encontraron ejemplos para el estado '{state_name}'")
        # Mostrar estados disponibles
        available_states = set(dataset.text_labels)
        print(f"Estados disponibles: {', '.join(sorted(available_states))}")
        return

    # Seleccionar aleatoriamente hasta num_examples
    if len(state_indices) > num_examples:
        selected_indices = random.sample(state_indices, num_examples)
    else:
        selected_indices = state_indices
        print(f"Nota: Solo se encontraron {len(selected_indices)} ejemplos para '{state_name}'")

    # Configurar la figura
    fig, axes = plt.subplots(1, len(selected_indices), figsize=figsize)

    # Asegurar que axes sea siempre una secuencia
    if len(selected_indices) == 1:
        axes = [axes]

    # Mostrar cada imagen
    for i, idx in enumerate(selected_indices):
        # Obtener la imagen y la etiqueta
        image, label = dataset[idx]

        # Convertir tensor a numpy para visualización
        if isinstance(image, torch.Tensor):
            # Si es un tensor, mover a CPU y convertir a numpy
            image_np = image.cpu().numpy()

            # Reorganizar dimensiones si es necesario (C, H, W) -> (H, W, C)
            if image_np.shape[0] == 3:  # Si el canal está en la primera dimensión
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = np.array(image)

        # Mostrar la imagen
        axes[i].imshow(image_np)
        axes[i].set_title(f"{state_name}\nID: {idx}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Ejemplos de placas de matrícula - {state_name}", y=1.05)
    plt.show()
```


```python
dataset = PlatesDataSet(mode='train')
state_name = 'NEW YORK'
visualize_state_plates(dataset, state_name)
```

    Dataset cargado con 8161 imágenes en modo 'train'



    
![png](01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_files/01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_17_1.png)
    




### Ejercicio 5: Exploración del Gradiente con Autograd

**Investiga y visualiza el gradiente de la siguiente función en el rango [0, 2π] utilizando `autograd` de PyTorch.

$$f(x) = \tan(2x+ \frac{\pi}{2}) $$

**Consideraciones:**

* **Tensor 'x':** Define un tensor 'x' adecuado para el rango y la función dada.
* **Gradiente:** Utiliza `autograd` para calcular el gradiente de f(x) con respecto a x.
* **Visualización:** Crea gráficos que muestren tanto la función original como su gradiente. Asegúrate de que ambos gráficos tengan el eje y limitado entre -5 y 5.


```python
#inserte su código aquí
import math
import torch
from matplotlib import pyplot as plt

x = torch.linspace(0, 2*math.pi, steps=25, requires_grad=True)
y = torch.tan(x * 2 + math.pi/2)

plt.plot(x.detach(), y.detach())
plt.ylim(-1.1, 1.1)
plt.show()

s = y.sum()
s.backward()

plt.plot(x.detach(), x.grad.detach())
plt.ylim(0, 100)
plt.show()
```


    
![png](01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_files/01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_19_0.png)
    



    
![png](01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_files/01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_19_1.png)
    


# Parte 2: Redes de Una Capa

En esta parte del notebook vamos a retomar el ejemplo de la red neuronal entrenada para clasificar FashionMNIST. Si bien ese ejemplo tiene todos los pasos necesarios para entrenar la red, varios de esos pasos sirven para entrenar cualquier otro modelo. Así que vamos a tratar de modularizarlo de manera que el código sea reutilizable.

Arranquemos importando los módulos necesarios.


```python
import torch
import torchvision
from IPython import display
from torchvision import transforms
from torch.utils import data
```

Volvemos a definir la función que crea los Datasets y devuelve los DataLoaders para poder iterar sobre ellos.


```python
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=1),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=1))

```

También definimos una función que devuelve la cantidad de aciertos del modelo a partir de un tensor de predicciones y otro de etiquetas.


```python
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

Volvemos a definir el modelo con una capa de 10 neuronas para hacer la clasificación e inicializamos sus pesos aleatoriamente con una distribución gaussiana.


```python
net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

Definimos la entropía cruzada como función de perdida y el descenso de gradiente estocástico como algoritmo de optimización.


```python
loss = torch.nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

Y por último, definimos una función que lleva adelante el entrenamiento completo.


```python
def train(net, train_iter, test_iter, loss, num_epochs, updater):
  '''
  Lleva adelante el entrenamiento completo llamando a funciones internas
  que modularizan el ciclo de entrenamiento.

    Parámetros:
            net: la red neuronal que se va a entrenar
            train_iter: iterador de datos de entrenamiento
            test_iter: iterador de datos de prueba
            loss: función de perdida a minimizar
            num_epoch: cantidad de épocas a entrenar
            updater: algoritmo de optimización

    Salida:
            metrics: una lista de tuplas (una para cada epoch)
              con las siguientes componentes
              - epoch: número de época
              - L: pérdida calculada
              - Acc: accuracy de entrenamiento calculada
              - TestAcc: accuracy de prueba calculada
  '''
  metrics =[]
  for epoch in range(num_epochs):
      L, Acc = train_epoch(net, train_iter, loss, updater)
      TestAcc = test_accuracy(net, test_iter)
      metric = (epoch + 1, L, Acc, TestAcc)
      print(metric)
      metrics.append(metric)
  return metrics

```

## Ejercicio 1: train_epoch()

Implementar la función `train_epoch()` que lleva adelante el entrenamiento de una época.


```python
def train_epoch(net, train_iter, loss, updater):
  '''
  Lleva adelante el entrenamiento de una sola época.

    Parámetros:
            net: la red neuronal que se va a entrenar
            train_iter: iterador de datos de entrenamiento
            loss: función de perdida a minimizar
            updater: algoritmo de optimización

    Salida:
            L: pérdida calculada
            Acc: accuracy de entrenamiento calculada
  '''
  # inserte su código aquí
  L = 0.0
  N = 0
  n = 0
  Acc = 0.0
  for X, y in train_iter:
      updater.zero_grad()
      l = loss(net(X) ,y)
      l.mean().backward()
      updater.step()
      L += l.mean()
      N += l.numel()
      n += 1
      Acc += accuracy(net(X), y)
  L /= 1
  Acc /= N
  return L.detach(), Acc

```

## Ejercicio 2: test_accuracy()

Implementar la función `test_accuracy()` que lleva adelante la evaluación de la performance de la red con los datos de prueba.


```python
def test_accuracy(net, test_iter):
  '''
  Evalúa los resultados del entrenamiento de una sola época.

    Parámetros:
            net: la red neuronal que se va a evaluar
            test_iter: iterador de datos de prueba

    Salida:
            - TestAcc: accuracy de prueba calculada
  '''
  # inserte su código aquí
  TestAcc = 0.0
  N = 0
  for X, y in test_iter:
      N += y.numel()
      TestAcc += accuracy(net(X), y)
  TestAcc /= N
  return TestAcc

```

## Ejercicio 3

Utilizar las funciones anteriores para entrenar efectivamente a la red. Entrenarla por 10 epochs y con lotes de tamaño 256.


```python
#inserte su código aquí
num_epochs = 10
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
metrics = train(net, train_iter, test_iter, loss, num_epochs, trainer)
```

    (1, tensor(185.1004), 0.7643333333333333, 0.7885)
    (2, tensor(133.8725), 0.82105, 0.7828)
    (3, tensor(123.4896), 0.8334, 0.8189)
    (4, tensor(117.8344), 0.8389666666666666, 0.7869)
    (5, tensor(113.8664), 0.84365, 0.8232)
    (6, tensor(111.1461), 0.8466166666666667, 0.828)
    (7, tensor(109.6065), 0.8489833333333333, 0.8261)
    (8, tensor(107.4621), 0.8512666666666666, 0.8283)
    (9, tensor(106.2123), 0.8534333333333334, 0.8279)
    (10, tensor(105.1572), 0.85415, 0.8288)


## Ejercicio 4
Graficar la evolución de los valores de el accuracy de entrenamiento, el accuracy de prueba y la pérdida en función de las épocas.


```python
#inserte su código aquí
num_epochs = 10
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
metrics = train(net, train_iter, test_iter, loss, num_epochs, trainer)
```

    (1, tensor(103.9574), 0.85495, 0.837)
    (2, tensor(103.1612), 0.8565, 0.8341)
    (3, tensor(102.4219), 0.8573166666666666, 0.831)
    (4, tensor(101.8878), 0.8586833333333334, 0.8368)
    (5, tensor(100.8662), 0.85815, 0.8385)
    (6, tensor(100.5844), 0.8596333333333334, 0.8381)
    (7, tensor(100.1003), 0.8607833333333333, 0.8307)
    (8, tensor(99.4679), 0.8608333333333333, 0.8385)
    (9, tensor(99.1231), 0.8618666666666667, 0.8391)
    (10, tensor(98.9044), 0.8629333333333333, 0.8409)


# Parte 3: Redes Multicapa

En este notebook vamos a usar MLPs para generar un modelo clasificador sobre FashionMNIST así que muchas de las funciones que usamos en los ejercicios de la clase 2 te serán muy útiles.

## Ejercicio 1:

Generar un modelo perceptron multicapa con 2 capas ocultas de 512 y 128 neuronas respectivamente para clasificación sobre el dataset FashionMNIST


```python
import torch
from torch import nn

INPUT = 28 * 28 # 28 por 28 pixeles
OUTPUT = 10 # 10 clases
# TODO
HIDDEN1 = 512 # elija los valores a completar
HIDDEN2 = 128 # elija los valores a completar


net1 = nn.Sequential(nn.Flatten(),
                    nn.Linear(INPUT, HIDDEN1),
                    nn.ReLU(),
                    nn.Linear(HIDDEN1, HIDDEN2),
                    nn.ReLU(),
                    nn.Linear(HIDDEN2, OUTPUT))
```

## Ejercicio 2

Entrene el modelo por 10 épocas con un tamaño de lote de 256 y un learning rate de 0.3. (Le recomendamos reutilizar las funciones modularizadas de los ejercicios de la clase 2)


```python
batch_size, lr, num_epochs = 256, 0.3, 10


net1.apply(init_weights);

train_iter, test_iter = load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss(reduction='none')
trainer1 = torch.optim.SGD(net1.parameters(), lr=lr)

train(net1, train_iter, test_iter, loss, num_epochs, trainer1)
```

    (1, tensor(280.3407), 0.5536833333333333, 0.7562)
    (2, tensor(132.3263), 0.8082, 0.8026)
    (3, tensor(108.9377), 0.8471333333333333, 0.8355)
    (4, tensor(98.3007), 0.8660666666666667, 0.8447)
    (5, tensor(91.3857), 0.8774833333333333, 0.8323)
    (6, tensor(86.8454), 0.8855666666666666, 0.8506)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (7, tensor(82.0164), 0.8912, 0.8512)
    (8, tensor(78.9782), 0.9000333333333334, 0.8526)
    (9, tensor(75.5500), 0.90435, 0.8486)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (10, tensor(72.3990), 0.9082666666666667, 0.8586)





    [(1, tensor(280.3407), 0.5536833333333333, 0.7562),
     (2, tensor(132.3263), 0.8082, 0.8026),
     (3, tensor(108.9377), 0.8471333333333333, 0.8355),
     (4, tensor(98.3007), 0.8660666666666667, 0.8447),
     (5, tensor(91.3857), 0.8774833333333333, 0.8323),
     (6, tensor(86.8454), 0.8855666666666666, 0.8506),
     (7, tensor(82.0164), 0.8912, 0.8512),
     (8, tensor(78.9782), 0.9000333333333334, 0.8526),
     (9, tensor(75.5500), 0.90435, 0.8486),
     (10, tensor(72.3990), 0.9082666666666667, 0.8586)]



## Ejercicio 3 :

A partir del modelo anterior, analice que ocurre si en lugar de entrenar 10 épocas, entrena 20


```python
batch_size, lr, num_epochs = 256, 0.3, 20

net1.apply(init_weights);

train(net1, train_iter, test_iter, loss, num_epochs, trainer)
```

    (1, tensor(578.4113), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (2, tensor(578.4470), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (3, tensor(578.4771), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^AssertionError
    : can only test a child process


    (4, tensor(578.4629), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (5, tensor(578.4697), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (6, tensor(578.4310), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (7, tensor(578.4912), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (8, tensor(578.4733), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (9, tensor(578.4990), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (10, tensor(578.4617), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (11, tensor(578.4607), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (12, tensor(578.4597), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (13, tensor(578.4252), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (14, tensor(578.4390), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (15, tensor(578.4631), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (16, tensor(578.4379), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^
    ^  File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (17, tensor(578.4293), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (18, tensor(578.4490), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (19, tensor(578.4711), 0.1, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (20, tensor(578.4344), 0.1, 0.1)





    [(1, tensor(578.4113), 0.1, 0.1),
     (2, tensor(578.4470), 0.1, 0.1),
     (3, tensor(578.4771), 0.1, 0.1),
     (4, tensor(578.4629), 0.1, 0.1),
     (5, tensor(578.4697), 0.1, 0.1),
     (6, tensor(578.4310), 0.1, 0.1),
     (7, tensor(578.4912), 0.1, 0.1),
     (8, tensor(578.4733), 0.1, 0.1),
     (9, tensor(578.4990), 0.1, 0.1),
     (10, tensor(578.4617), 0.1, 0.1),
     (11, tensor(578.4607), 0.1, 0.1),
     (12, tensor(578.4597), 0.1, 0.1),
     (13, tensor(578.4252), 0.1, 0.1),
     (14, tensor(578.4390), 0.1, 0.1),
     (15, tensor(578.4631), 0.1, 0.1),
     (16, tensor(578.4379), 0.1, 0.1),
     (17, tensor(578.4293), 0.1, 0.1),
     (18, tensor(578.4490), 0.1, 0.1),
     (19, tensor(578.4711), 0.1, 0.1),
     (20, tensor(578.4344), 0.1, 0.1)]



## Ejercicio 4

Aumente el learning rate a 1 y entrene nuevamente. ¿Cómo puede explicar lo que pasó?


```python
batch_size, lr, num_epochs = 256, 1, 10

net1.apply(init_weights);
trainer = torch.optim.SGD(net1.parameters(), lr=lr)

train(net1, train_iter, test_iter, loss, num_epochs, trainer)
```

    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (1, tensor(567.6570), 0.2018, 0.0997)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (2, tensor(566.6817), 0.11553333333333334, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (3, tensor(541.3739), 0.1152, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (4, tensor(541.3404), 0.11368333333333333, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (5, tensor(541.3408), 0.11428333333333333, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (6, tensor(541.3362), 0.11371666666666666, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (7, tensor(541.3378), 0.11188333333333333, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (8, tensor(541.3522), 0.11305, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (9, tensor(541.3673), 0.11625, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (10, tensor(541.3515), 0.11296666666666667, 0.1)





    [(1, tensor(567.6570), 0.2018, 0.0997),
     (2, tensor(566.6817), 0.11553333333333334, 0.1),
     (3, tensor(541.3739), 0.1152, 0.1),
     (4, tensor(541.3404), 0.11368333333333333, 0.1),
     (5, tensor(541.3408), 0.11428333333333333, 0.1),
     (6, tensor(541.3362), 0.11371666666666666, 0.1),
     (7, tensor(541.3378), 0.11188333333333333, 0.1),
     (8, tensor(541.3522), 0.11305, 0.1),
     (9, tensor(541.3673), 0.11625, 0.1),
     (10, tensor(541.3515), 0.11296666666666667, 0.1)]



## Ejercicio 5:

Analize el efecto de cambiar las funciones de activación en el accurracy


```python
INPUT = 28 * 28
OUTPUT = 10
HIDDEN1 = 512
HIDDEN2 = 128


net2 = nn.Sequential(nn.Flatten(),
                    nn.Linear(INPUT, HIDDEN1),
                    nn.Sigmoid(),
                    nn.Linear(HIDDEN1, HIDDEN2),
                    nn.Sigmoid(),
                    nn.Linear(HIDDEN2, OUTPUT))
batch_size, lr, num_epochs = 256, 0.3, 10

net2.apply(init_weights);

train_iter, test_iter = load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss(reduction='none')
trainer2 = torch.optim.SGD(net2.parameters(), lr=lr)
train(net2, train_iter, test_iter, loss, num_epochs, trainer2)
```

    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (1, tensor(542.8970), 0.13203333333333334, 0.1)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (2, tensor(459.1227), 0.27021666666666666, 0.4211)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (3, tensor(268.0107), 0.5960833333333333, 0.6115)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive(): 
          ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (4, tensor(209.4846), 0.6658166666666666, 0.6571)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (5, tensor(188.4669), 0.7054, 0.7179)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (6, tensor(170.7083), 0.74255, 0.7083)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (7, tensor(156.1484), 0.7641, 0.7585)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (8, tensor(145.7421), 0.78155, 0.777)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (9, tensor(137.0033), 0.7963666666666667, 0.7732)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (10, tensor(130.7798), 0.80815, 0.7927)





    [(1, tensor(542.8970), 0.13203333333333334, 0.1),
     (2, tensor(459.1227), 0.27021666666666666, 0.4211),
     (3, tensor(268.0107), 0.5960833333333333, 0.6115),
     (4, tensor(209.4846), 0.6658166666666666, 0.6571),
     (5, tensor(188.4669), 0.7054, 0.7179),
     (6, tensor(170.7083), 0.74255, 0.7083),
     (7, tensor(156.1484), 0.7641, 0.7585),
     (8, tensor(145.7421), 0.78155, 0.777),
     (9, tensor(137.0033), 0.7963666666666667, 0.7732),
     (10, tensor(130.7798), 0.80815, 0.7927)]



## Ejercicio 6:

Ahora genere un tercer modelo en donde ambas capas tengan 1024 neuronas. Analice si produjo algún cambio en los rendimientos.


```python
INPUT = 28 * 28
OUTPUT = 10
HIDDEN1 = 1024
HIDDEN2 = 1024
batch_size, lr, num_epochs = 256, 0.3, 20

net3 = nn.Sequential(nn.Flatten(),
                    nn.Linear(INPUT, HIDDEN1),
                    nn.ReLU(),
                    nn.Linear(HIDDEN1, HIDDEN2),
                    nn.ReLU(),
                    nn.Linear(HIDDEN2, OUTPUT))

net3.apply(init_weights);

train_iter, test_iter = load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss(reduction='none')
trainer3 = torch.optim.SGD(net3.parameters(), lr=lr)
train(net3, train_iter, test_iter, loss, num_epochs, trainer3)
```

    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (1, tensor(215.5878), 0.6754, 0.7784)


    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x79f71c0979c0>
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1604, in __del__
        self._shutdown_workers()
      File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1587, in _shutdown_workers
        if w.is_alive():
           ^^^^^^^^^^^^
      File "/usr/lib/python3.11/multiprocessing/process.py", line 160, in is_alive
        assert self._parent_pid == os.getpid(), 'can only test a child process'
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: can only test a child process


    (2, tensor(120.3567), 0.8305333333333333, 0.7617)
    (3, tensor(103.4547), 0.8580333333333333, 0.8339)
    (4, tensor(93.6360), 0.8738333333333334, 0.827)
    (5, tensor(87.7321), 0.8850666666666667, 0.8287)
    (6, tensor(82.2614), 0.8916666666666667, 0.8429)
    (7, tensor(78.3130), 0.8995833333333333, 0.8243)
    (8, tensor(75.1022), 0.9053833333333333, 0.8278)
    (9, tensor(72.4800), 0.90835, 0.8616)
    (10, tensor(69.4758), 0.9145666666666666, 0.8373)
    (11, tensor(68.0699), 0.9177833333333333, 0.8543)
    (12, tensor(65.1365), 0.9208833333333334, 0.8625)
    (13, tensor(63.7211), 0.92525, 0.8707)
    (14, tensor(60.9612), 0.9282166666666667, 0.8585)
    (15, tensor(59.4470), 0.9319333333333333, 0.8835)
    (16, tensor(58.0697), 0.9339166666666666, 0.8118)
    (17, tensor(56.5440), 0.9364666666666667, 0.857)
    (18, tensor(55.7306), 0.9379166666666666, 0.8824)
    (19, tensor(53.5975), 0.9411, 0.8675)
    (20, tensor(51.7493), 0.9432833333333334, 0.8186)





    [(1, tensor(215.5878), 0.6754, 0.7784),
     (2, tensor(120.3567), 0.8305333333333333, 0.7617),
     (3, tensor(103.4547), 0.8580333333333333, 0.8339),
     (4, tensor(93.6360), 0.8738333333333334, 0.827),
     (5, tensor(87.7321), 0.8850666666666667, 0.8287),
     (6, tensor(82.2614), 0.8916666666666667, 0.8429),
     (7, tensor(78.3130), 0.8995833333333333, 0.8243),
     (8, tensor(75.1022), 0.9053833333333333, 0.8278),
     (9, tensor(72.4800), 0.90835, 0.8616),
     (10, tensor(69.4758), 0.9145666666666666, 0.8373),
     (11, tensor(68.0699), 0.9177833333333333, 0.8543),
     (12, tensor(65.1365), 0.9208833333333334, 0.8625),
     (13, tensor(63.7211), 0.92525, 0.8707),
     (14, tensor(60.9612), 0.9282166666666667, 0.8585),
     (15, tensor(59.4470), 0.9319333333333333, 0.8835),
     (16, tensor(58.0697), 0.9339166666666666, 0.8118),
     (17, tensor(56.5440), 0.9364666666666667, 0.857),
     (18, tensor(55.7306), 0.9379166666666666, 0.8824),
     (19, tensor(53.5975), 0.9411, 0.8675),
     (20, tensor(51.7493), 0.9432833333333334, 0.8186)]



# Parte 4: Pytorch Avanzado

## Setup


```python
import sys
```


```python
import sklearn
```


```python
import torch
from torchvision import transforms
from torch.utils import data
import torchvision

```

Definiciones adicionales pra que nuestras figuras se vean "bonitas"


```python
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

## Aprendiendo a quitar el ruido de imágenes

En este notebook vamos a seguir trabajando sobre el dataset FashionMNIST, pero esta vez, en vez de clasificar las imágenes según la prenda que contienen, vamos a modificar las imágenes agregando un ruido aleatorio que una de nuestras redes neuronales aprenderá a quitar.



Arranquemos cargando el dataset.


```python
torch.manual_seed(42)  # fijamos la semilla para generar reproducibilidad
batch_size = 256

# Dataloader para FashionMNIST
mnist_train = torchvision.datasets.FashionMNIST(transform=transforms.ToTensor(),
        root="../data", train=True, download=True)
mnist_test = torchvision.datasets.FashionMNIST(transform=transforms.ToTensor(),
        root="../data", train=False, download=True)
iter_train, iter_valid =  (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=2),
            data.DataLoader(mnist_test, batch_size, shuffle=True,
                            num_workers=2))
```

Luego vamos a crear un bloque que genere ruido mediante la capa Dropout. Esta capa multiplica por cero pixeles aleatorios de las imágenes con una probabilidad igual a un escalar que pasamos como parámetro.


```python
p=0.5 #probabilidad de que un pixel sea eliminado
noise = torch.nn.Sequential(torch.nn.Dropout(p))

images,_ = next(iter(iter_train))
noise_images = noise(images)

n_images = 10
fig = plt.figure(figsize=(n_images * 2, 4))
for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(noise_images[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")



```


    
![png](01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_files/01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_71_0.png)
    


## Denoising Autoencoders

A muy alto nivel, un `autoencoder` (codificador automático) contiene un `encoder` (codificador) y un `decoder` (decodificador). Estas dos partes funcionan automáticamente y dan lugar al nombre de `autoencoder`. El encoder transforma la entrada de alta dimensión en una dimensión más baja (espacio latente, donde la entrada está más comprimida), mientras que un decoder hace el trabajo inverso del encoder en el resultado codificado y reconstruye la imagen original.

![Imgur](https://i.imgur.com/iOp5Vdu.png)

En la tarea de eliminación de ruido, los datos se corrompen de alguna manera para que el modelo pueda aprender a predecir la imagen original. En este caso, la idea es almacenar la salida generada por el encoder como un vector de características de la entrada (llamado vector latente) que está tan comprimido que de alguna manera guarda información solamente de la imagen subyacente y no del ruido. De esta forma, al reconstruir la imagen con el decoder teniendo en cuenta solamente el vector latente como entrada, la salida sería la imagen original sin ruido.

## Ejercicio 1: Construcción del Encoder

En este ejercicio usted deberá crear con pytorch un MLP de 2 capas (con 100 y 30 neuronas respectivamente) que reciba como entrada imágenes de 28*28 y produzca como salida vectores latentes de 30 elementos. No olvide que los MLP necesitan funciones de activación para poder apilar sus capas!!


```python
Input = 28 * 28
Hidden1 = 100
Hidden2 = 30
encoder = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(Input, Hidden1),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden1, Hidden2),
    torch.nn.ReLU()
)
```


```python
#@title Test N° 1
#@markdown Ejecutar para confirmar que su código es correcto
images,_ = next(iter(iter_train))

try:
  latentes = encoder(images)
  assert latentes.shape[1] == 30, "La salida de su red no es un vector de 30 elementos"
  print("Al parecer está todo bien. Puedes avanzar al siguiente test")
except:
  print("Su encoder no generó una salida válida.\nLa entrada no pudo recorrer todo el camino hasta el final de su red.\nRevise que la dimensionalidad de sus capas sean compatibles")


```

    Al parecer está todo bien. Puedes avanzar al siguiente test



```python
#@title Test N° 2
#@markdown Ejecutar para confirmar que su código es correcto
assert encoder[1].weight.shape == torch.Size([100, 784]), "Las dimensiones de su primera capa densa están mal"
assert encoder[3].weight.shape == torch.Size([30, 100]), "Las dimensiones de su segunda capa densa están mal"
print("Al parecer está todo bien. Puedes avanzar al siguiente ejercicio")
```

    Al parecer está todo bien. Puedes avanzar al siguiente ejercicio


## Ejercicio 2: Construcción del Decoder

En este ejercicio usted deberá crear con pytorch un MLP de 2 capas que sea inverso al decoder que construyo en el ejercicio anterior (con 30 y 100 neuronas respectivamente) que reciba como entrada vectores latentes de 30 elementos y produzca como salida imágenes de 28*28 . No olvide que los MLP necesitan funciones de activación para poder apilar sus capas!!
Tip: para revertir un Flatten() debe usar Unflatten()


```python
decoder = torch.nn.Sequential(
    torch.nn.Linear(Hidden2, Hidden1),
    torch.nn.ReLU(),
    torch.nn.Linear(Hidden1, Input),
    torch.nn.Unflatten(-1, torch.Size([28, 28]))
)
```


```python
#@title Test N° 1
#@markdown Ejecutar para confirmar que su código es correcto
try:
  salidas = decoder(latentes)
  assert salidas.shape[1] == 28 and salidas.shape[2] == 28, "La salida de su red no es una imagen de 28*28"
  print("Al parecer está todo bien. Puedes avanzar al siguiente test")
except:
  print("Su encoder no generó una salida válida.\nLa entrada no pudo recorrer todo el camino hasta el final de su red.\nRevise que la dimensionalidad de sus capas sean compatibles")



```

    Al parecer está todo bien. Puedes avanzar al siguiente test



```python
#@title Test N° 2
#@markdown Ejecutar para confirmar que su código es correcto
assert decoder[0].weight.shape == torch.Size([100, 30]), "Las dimensiones de su primera capa densa están mal"
assert decoder[2].weight.shape == torch.Size([784, 100]), "Las dimensiones de su segunda capa densa están mal"
print("Al parecer está todo bien. Puedes avanzar al siguiente ejercicio")
```

    Al parecer está todo bien. Puedes avanzar al siguiente ejercicio


## Ejercicio 3: Crear un autoencoder
En este ejercicio deberás crear un bloque que consista en los bloques noise, encoder y decoder creados anteriormente encadenados. Tanto la slaida como la entrada de este bloque deben ser imágenes de 28*28


```python
net = torch.nn.Sequential(noise, encoder, decoder)
```


```python
#@title Test N° 1
#@markdown Ejecutar para confirmar que su código es correcto
try:
  salidas = net(images)
  assert salidas.size == salidas.size, "La salida de su red no tiene el mismo tamaño que la entrada"
  print("Al parecer está todo bien. Puedes avanzar al siguiente test")
except:
  print("Su encoder no generó una salida válida.\nLa entrada no pudo recorrer todo el camino hasta el final de su red.\nRevise que la dimensionalidad de sus capas sean compatibles")



```

    Al parecer está todo bien. Puedes avanzar al siguiente test



```python
#@title Test N° 2
#@markdown Ejecutar para confirmar que su código es correcto
assert net[0] == noise, "Tu primer bloque no es el correcto"
assert net[1] == encoder, "Tu segundo bloque no es el correcto"
assert net[2] == decoder, "Tu tercer bloque no es el correcto"
print("Al parecer está todo bien. Puedes avanzar al siguiente test")

```

    Al parecer está todo bien. Puedes avanzar al siguiente test


## Ejercicio 4: Entrenar el Autoencoder

Antes de entrenar debemos definir la pérdida. Nuestro objetivo es que la salida de la red sea exactamente igual que la entrada. Por lo tanto, debemos establecer como etiquetas a las imágenes de entrada y compararlas mediante el error cuadrático medio.


```python
loss = torch.nn.MSELoss()
```

Para obtener un mejor rendimiento usaremos Adam como algoritmo de optimización en lugar de SGD. En la última clase explicaremos algunas diferencias entre ambos.


```python
trainer = torch.optim.Adam(net.parameters())

```

Un último detalle es que la capa Dropout se comporta diferente si la red está entrenando o prediciendo. Así que debemos indicarle a PyTorch que la red está en modo entrenamiento con la función `train()`


```python
net.train()
```




    Sequential(
      (0): Sequential(
        (0): Dropout(p=0.5, inplace=False)
      )
      (1): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=784, out_features=100, bias=True)
        (2): ReLU()
        (3): Linear(in_features=100, out_features=30, bias=True)
        (4): ReLU()
      )
      (2): Sequential(
        (0): Linear(in_features=30, out_features=100, bias=True)
        (1): ReLU()
        (2): Linear(in_features=100, out_features=784, bias=True)
        (3): Unflatten(dim=-1, unflattened_size=torch.Size([28, 28]))
      )
    )



En la siguiente celda deberá entrenar la red por 50 épocas





```python
#ingrese su código aquí
num_epochs = 50

# Ciclo de entrenamiento
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
for epoch in range(num_epochs):
    L = 0.0
    N = 0
    for X, _ in iter_train:
        Y = X.squeeze().detach().clone()
        #Y += 0.1 * torch.randn_like(X)
        l = loss(net(X), Y)
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()
        L += l.sum()
        N += l.numel()
    print(f'epoch {epoch + 1}, loss {(L/N):f}')


```

    epoch 1, loss 0.073742
    epoch 2, loss 0.046431
    epoch 3, loss 0.039437
    epoch 4, loss 0.034312
    epoch 5, loss 0.029578
    epoch 6, loss 0.027943
    epoch 7, loss 0.027204
    epoch 8, loss 0.026524
    epoch 9, loss 0.025960
    epoch 10, loss 0.025368
    epoch 11, loss 0.024773
    epoch 12, loss 0.024267
    epoch 13, loss 0.023853
    epoch 14, loss 0.023526
    epoch 15, loss 0.023208
    epoch 16, loss 0.022878
    epoch 17, loss 0.022633
    epoch 18, loss 0.022368
    epoch 19, loss 0.022108
    epoch 20, loss 0.021876
    epoch 21, loss 0.021697
    epoch 22, loss 0.021557
    epoch 23, loss 0.021409
    epoch 24, loss 0.021290
    epoch 25, loss 0.021220
    epoch 26, loss 0.021106
    epoch 27, loss 0.021010
    epoch 28, loss 0.020937
    epoch 29, loss 0.020834
    epoch 30, loss 0.020758
    epoch 31, loss 0.020697
    epoch 32, loss 0.020624
    epoch 33, loss 0.020568
    epoch 34, loss 0.020508
    epoch 35, loss 0.020460
    epoch 36, loss 0.020405
    epoch 37, loss 0.020357
    epoch 38, loss 0.020333
    epoch 39, loss 0.020284
    epoch 40, loss 0.020218
    epoch 41, loss 0.020173
    epoch 42, loss 0.020159
    epoch 43, loss 0.020102
    epoch 44, loss 0.020064
    epoch 45, loss 0.020051
    epoch 46, loss 0.020021
    epoch 47, loss 0.019981
    epoch 48, loss 0.019927
    epoch 49, loss 0.019907
    epoch 50, loss 0.019866



```python
#@title Grafique Predicciones de Validación
# Codigo adicional para generar imágenes.
import numpy as np


def plot_reconstructions(model, images=iter_valid, n_images=10):
    noise = torch.nn.Sequential(torch.nn.Dropout(0.5))
    noise.train()
    input = noise(images)
    noise.eval()
    model.eval()
    reconstructions = np.clip(input[:n_images].squeeze().detach(), 0, 1)
    reconstructions = model(reconstructions).squeeze().detach()
    fig = plt.figure(figsize=(n_images * 2, 4))
    for image_index in range(n_images):
        plt.subplot(3, n_images, 1 + image_index)
        plt.imshow(images[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(3, n_images, 1 + n_images + image_index)
        plt.imshow(input[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(3, n_images, 1 + 2 * n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")

net.eval()
plot_reconstructions(net, next(iter(iter_valid))[0])
plt.show()
```


    
![png](01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_files/01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_97_0.png)
    


## Ejercicio 5: Modificar la fuente del Ruido

Intentemos usar un modelo similar, pero esta vez generemos un error gaussiano sobre las imágenes en vez de simplemente eliminar píxeles al azar. La siguiente función altera las imágenes agregando un error gaussiano.


```python
def addGaussianNoise(tensor, mean, std):
  return tensor + torch.randn_like(tensor) * std + mean

images,_ = next(iter(iter_train))
noise_images = addGaussianNoise(images,0,0.15)

n_images = 10
fig = plt.figure(figsize=(n_images * 2, 4))
for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(noise_images[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
```


    
![png](01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_files/01_fundamentos_deep_learning_Laboratorio_1_Solucio%CC%81n_99_0.png)
    


Ahora generemos una nueva red que en vez de eliminar un ruido producido por dropout, elimine un ruido gaussiano producido por la función `addGaussianNoise()`. Para hacerlo tenga en cuenta los siguientes detalles:


1.   El paquete nn.Module nos permite crear tanto capas como modelos completos personalizados. Utilice alguna de estas opciones para implementar este modelo.
2.   Tenga en cuenta que el ruido gaussiano solo debe agregarse a la imagen si el modelo está siendo entrenado. El atributo `training` de cualquier bloque que herede de nn.Module funciona como bandera que se pone en `True` si el modelo está entrenando y en `False` si ya está entrenada y se la usa para generar predicciones. El método `train()` pone esta bandera en `True` y el método `eval()` la pone en `False`




```python
## ingrese su código aquí
## Solución 1: generar una capa para addGaussianNoise()
from torch import nn

class GaussianLayer(nn.Module):
    def __init__(self, std, mean):
        super().__init__()
        self.std = std
        self.mean = mean
    def forward(self, X):
        if self.training:
          return addGaussianNoise(X,self.std,self.mean)
        else:
          return X

gaussian = GaussianLayer(0,0.15)
netGaussian1 = nn.Sequential(gaussian, encoder, decoder)
```


```python
## Solución 2: generar un autoencoder con un forward personalizado
class GaussianAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, std, mean):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.std = std
        self.mean = mean
    def forward(self, X):
        if self.training:
          X_noise = addGaussianNoise(X,self.std,self.mean)
        else:
          X_noise = X
        return decoder(encoder(X_noise))
netGaussian2 = GaussianAutoencoder(encoder,decoder,0,0.15)
```

## Ejercicio 6: Entrenar el modelo con ruido gaussiano

Entrene el modelo creado en el ejercicio anterior


```python
#ingrese su código aquí
num_epochs = 50

# Ciclo de entrenamiento
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net = netGaussian1
net.apply(init_weights);
for epoch in range(num_epochs):
    L = 0.0
    N = 0
    for X, _ in iter_train:
        Y = X.squeeze().detach().clone()
        #Y += 0.1 * torch.randn_like(X)
        l = loss(net(X), Y)
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()
        L += l.sum()
        N += l.numel()
    print(f'epoch {epoch + 1}, loss {(L/N):f}')
```


```python
#@title Grafique Predicciones de Validación
# Codigo adicional para generar imágenes.
import numpy as np


def plot_reconstructions(model, images=iter_valid, n_images=10):
    input = addGaussianNoise(images,0,0.15)
    model.eval()
    reconstructions = np.clip(input[:n_images].squeeze().detach(), 0, 1)
    reconstructions = model(reconstructions).squeeze().detach()
    fig = plt.figure(figsize=(n_images * 2, 4))
    for image_index in range(n_images):
        plt.subplot(3, n_images, 1 + image_index)
        plt.imshow(images[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(3, n_images, 1 + n_images + image_index)
        plt.imshow(input[image_index].squeeze(),
                   cmap="binary")
        plt.axis("off")
        plt.subplot(3, n_images, 1 + 2 * n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")

net.eval()
plot_reconstructions(netGaussian1, next(iter(iter_valid))[0])
plt.show()
```

# Parte 5: Selección de Modelos

Para llevar adelante los ejercicios de este notebook vamos a recuperar los modelos que construimos en la clase 3 para clasificación sobre FashionMNIST.


```python
import torch
from torch import nn

INPUT = 28 * 28
OUTPUT = 10

net1 = nn.Sequential(nn.Flatten(),
                    nn.Linear(INPUT, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, OUTPUT))

net2 = nn.Sequential(nn.Flatten(),
                    nn.Linear(INPUT, 512),
                    nn.Sigmoid(),
                    nn.Linear(512, 128),
                    nn.Sigmoid(),
                    nn.Linear(128, OUTPUT))

net3 = nn.Sequential(nn.Flatten(),
                    nn.Linear(INPUT, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, OUTPUT))
```

Ahora repetiremos la evaluación que hicimos en esa clase, pero llevándola adelante de manera más exhaustiva con K-fold Cross Validation. Para eso, cargaremos **solo los datos de testeo** de FasionMNIST y fingiremos que esos 10000 ejemplos son todos los que tenemos.


```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

data_iter = datasets.FashionMNIST(
        root="../data", train=False, transform=transforms.ToTensor(), download=True)
```

## Ejercicio
Verifique cuál de los modelos anteriores es el mejor llevando adelante proceso de cross validation con 3 folds y entrenando por 20 epochs. Reutilice todas las funciones que necesite de los notebooks de teoría de la clase 5 y de todos los ejercicios anteriores.

Nota: le recomendamos que guarde las accuracy tanto de testeo como de entrenamiento porque le servirán para más adelante.


```python
#### Funciones necesarias de otros notebooks

def reset_weights(m):
  if type(m) == nn.Linear:
      nn.init.normal_(m.weight, std=0.01)

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def test_accuracy(fold,model, loss, device, test_loader):
  # inserte su código aquí
  TestAcc = 0.0
  N = 0
  for X, y in test_loader:
      X, y = X.to(device), y.to(device)
      N += y.numel()
      TestAcc += accuracy(model(X), y)
  print('\nTest set for fold {}:  Accuracy: {}/{} ({:.0f}%)'.format(
        fold, TestAcc, N,
        (100. * TestAcc) / N))
  return TestAcc / N

def train_accuracy(fold,model, loss, device, test_loader):
  # inserte su código aquí
  TestAcc = 0.0
  N = 0
  for X, y in test_loader:
      X, y = X.to(device), y.to(device)
      N += y.numel()
      TestAcc += accuracy(model(X), y)
  print('\nTrain set for fold {}:  Accuracy: {}/{} ({:.0f}%)'.format(
        fold, TestAcc, N,
        (100. * TestAcc) / N))
  return TestAcc / N

def train(fold, model, device, loss, train_loader, optimizer, epoch):

  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      l = loss(model(data), target).mean()
      l.backward()
      optimizer.step()
```


```python
from sklearn.model_selection import KFold

### Función que lleva adelante el proceso de kfold cross validation
def train_kfold(model, dataset, n_fold, epochs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  loss = torch.nn.CrossEntropyLoss(reduction='none')
  optimizer = torch.optim.Adam(model.parameters())
  batch_size=32
  folds=n_fold
  train_acc = []
  acc = []
  kfold=KFold(n_splits=n_fold,shuffle=True)
  for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
    print('------------fold no---------{}----------------------'.format(fold))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=test_subsampler)

    model.apply(reset_weights)

    fold_acc = 0
    for epoch in range(1, epochs + 1):
      train(fold, model, device, loss, trainloader, optimizer, epoch)
      fold_train_acc = train_accuracy(fold,model, loss, device, trainloader)
      fold_acc = test_accuracy(fold,model, loss, device,  testloader)
    train_acc.append(fold_train_acc)
    acc.append(fold_acc)
  return train_acc, acc
```


```python
##### Lleva adelante los entrenamientos
train_acc1, acc_1 = train_kfold(net1, data_iter, 3, 20)
train_acc2, acc_2 = train_kfold(net2, data_iter, 3, 20)
train_acc3, acc_3 = train_kfold(net3, data_iter, 3, 20)

```


```python
import numpy as np
print("Modelo 1:  entrenamiento ", np.array(train_acc1).mean(), ", validación: ", np.array(acc_1).mean() )
print("Modelo 2:  entrenamiento ", np.array(train_acc2).mean(), ", validación: ", np.array(acc_2).mean() )
print("Modelo 3:  entrenamiento ", np.array(train_acc3).mean(), ", validación: ", np.array(acc_3).mean() )

```


```python
print("Modelo 3:  entrenamiento ", np.array(train_acc3).mean(), ", validación: ", np.array(acc_3).mean() )
```
