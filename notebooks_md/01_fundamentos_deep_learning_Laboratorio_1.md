![Imgur](https://i.imgur.com/acSOZRh.png)

# Laboratorio N° 1

# Parte 1: Introducción

### Ejercicio 1: Creando Tensores Complejos
Generar un tensor llamado x cuyo contenido sea el siguiente:

![Imgur](https://i.imgur.com/rfaKGUG.png)

El tensor debe construirse sin escribir manualmente los valores.


```python
#inserte su código aquí


```




    7



### Ejercicio 2: Trabajando con Broadcasting
Tienes un tensor $A$ con forma (3, 2, 4), y un vector v con 6 elementos. Tu tarea es hacer los cambios necesarios para multiplicar $A$ por $v$ de manera que:


*   Cada fila de $A$ se multiplique por el elemento correspondiente de $v$. Es decir, la primera fila debería multiplicarse por 6, la segunda por 5 y así sucesivamente.



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
    tensor([1, 2, 3, 4, 5, 6])


**Pistas:**
1. Necesitas hacer que `v` tenga una forma compatible con `A`.
2. Usa `reshape()` o `view()` para reorganizar `v`.
3. Piensa en `broadcasting`: PyTorch expandirá dimensiones automáticamente si están en el lugar correcto.



```python
# Inserta aquí los cambios necesarios para poder multiplicar A por v
```

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
    #inserte su código aquí

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
        #ingrese su código aquí

    def __len__(self):
        """
        Retorna el número de imágenes en el dataset.
        """
        #ingrese su código aquí

    def __getitem__(self, index):
        """
        Retorna una imagen y su etiqueta correspondiente.

        Args:
            index (int): Índice del elemento a recuperar.

        Returns:
            tuple: (imagen, etiqueta)
        """
        # Ingrese su código aquí

    def get_class_name(self, index):
        """
        Método adicional para obtener el nombre del estado correspondiente a un índice.

        Args:
            index (int): Índice del elemento.

        Returns:
            str: Nombre del estado (ej. 'ALABAMA')
        """
        #ingrese su código aquí
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



### Ejercicio 5: Exploración del Gradiente con Autograd

**Investiga y visualiza el gradiente de la siguiente función en el rango [0, 2π] utilizando `autograd` de PyTorch.

$$f(x) = \tan(2x+ \frac{\pi}{2}) $$

**Consideraciones:**

* **Tensor 'x':** Define un tensor 'x' adecuado para el rango y la función dada.
* **Gradiente:** Utiliza `autograd` para calcular el gradiente de f(x) con respecto a x.
* **Visualización:** Crea gráficos que muestren tanto la función original como su gradiente. Asegúrate de que ambos gráficos tengan el eje y limitado entre -5 y 5.


```python
#inserte su código aquí

```

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

## Ejercicio 1: `train_epoch()`

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



```

## Ejercicio 2: `test_accuracy()`

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

```

## Ejercicio 3: Entrenar la red

Utilizar las funciones anteriores para entrenar efectivamente a la red. Entrenarla por 10 epochs y con lotes de tamaño 256.


```python
#inserte su código aquí

```

## Ejercicio 4: Graficar
Graficar la evolución de los valores de el accuracy de entrenamiento, el accuracy de prueba y la pérdida en función de las épocas.


```python
#inserte su código aquí

```

# Parte 3: Redes Multicapa

En este notebook vamos a usar MLPs para generar un modelo clasificador sobre FashionMNIST así que muchas de las funciones que usamos en los ejercicios de la clase 2 te serán muy útiles.

## Ejercicio 1: Modelo Base

Generar un modelo perceptron multicapa con 2 capas ocultas de 512 y 128 neuronas respectivamente para clasificación sobre el dataset FashionMNIST


```python
net1 = #ingresa tu código aquí
```

## Ejercicio 2: Menos Learning Rate

Entrene el modelo por 10 épocas con un tamaño de lote de 256 y un learning rate de 0.3. (Le recomendamos reutilizar las funciones modularizadas de los ejercicios de la clase 2)


```python
#ingresa tu código aquí

```

## Ejercicio 3: Más épocas

A partir del modelo anterior, analice que ocurre si en lugar de entrenar 10 épocas, entrena 20


```python
#ingresa tu código aquí
```

## Ejercicio 4: Más Learning Rate

Aumente el learning rate a 1 y entrene nuevamente. ¿Cómo puede explicar lo que pasó?


```python
#ingresa tu código aquí
```

## Ejercicio 5: Otras Funciones de Activación

Analize el efecto de cambiar las funciones de activación en el accurracy


```python
#ingresa tu código aquí
```

## Ejercicio 6: Más Neuronas

Ahora genere un tercer modelo en donde ambas capas tengan 1024 neuronas. Analice si produjo algún cambio en los rendimientos.


```python
#ingresa tu código aquí

```

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


    
![png](01_fundamentos_deep_learning_Laboratorio_1_files/01_fundamentos_deep_learning_Laboratorio_1_71_0.png)
    


## Denoising Autoencoders

A muy alto nivel, un `autoencoder` (codificador automático) contiene un `encoder` (codificador) y un `decoder` (decodificador). Estas dos partes funcionan automáticamente y dan lugar al nombre de `autoencoder`. El encoder transforma la entrada de alta dimensión en una dimensión más baja (espacio latente, donde la entrada está más comprimida), mientras que un decoder hace el trabajo inverso del encoder en el resultado codificado y reconstruye la imagen original.

![Imgur](https://i.imgur.com/iOp5Vdu.png)

En la tarea de eliminación de ruido, los datos se corrompen de alguna manera para que el modelo pueda aprender a predecir la imagen original. En este caso, la idea es almacenar la salida generada por el encoder como un vector de características de la entrada (llamado vector latente) que está tan comprimido que de alguna manera guarda información solamente de la imagen subyacente y no del ruido. De esta forma, al reconstruir la imagen con el decoder teniendo en cuenta solamente el vector latente como entrada, la salida sería la imagen original sin ruido.

## Ejercicio 1: Construcción del Encoder

En este ejercicio usted deberá crear con pytorch un MLP de 2 capas (con 100 y 30 neuronas respectivamente) que reciba como entrada imágenes de 28*28 y produzca como salida vectores latentes de 30 elementos. No olvide que los MLP necesitan funciones de activación para poder apilar sus capas!!


```python
#ingrese su código aquí
encoder = #mantenga este nombre de variable durante todo el notebook para que funcionen los tests
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


```python
#@title Test N° 2
#@markdown Ejecutar para confirmar que su código es correcto
assert encoder[1].weight.shape == torch.Size([100, 784]), "Las dimensiones de su primera capa densa están mal"
assert encoder[3].weight.shape == torch.Size([30, 100]), "Las dimensiones de su segunda capa densa están mal"
print("Al parecer está todo bien. Puedes avanzar al siguiente ejercicio")
```

## Ejercicio 2: Construcción del Decoder

En este ejercicio usted deberá crear con pytorch un MLP de 2 capas que sea inverso al decoder que construyo en el ejercicio anterior (con 30 y 100 neuronas respectivamente) que reciba como entrada vectores latentes de 30 elementos y produzca como salida imágenes de 28*28 . No olvide que los MLP necesitan funciones de activación para poder apilar sus capas!!
Tip: para revertir un Flatten() debe usar Unflatten()


```python
#ingrese su código aquí
decoder = #mantenga este nombre de variable durante todo el notebook para que funcionen los tests
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


```python
#@title Test N° 2
#@markdown Ejecutar para confirmar que su código es correcto
assert decoder[0].weight.shape == torch.Size([100, 30]), "Las dimensiones de su primera capa densa están mal"
assert decoder[2].weight.shape == torch.Size([784, 100]), "Las dimensiones de su segunda capa densa están mal"
print("Al parecer está todo bien. Puedes avanzar al siguiente ejercicio")
```

## Ejercicio 3: Crear un autoencoder
En este ejercicio deberás crear un bloque que consista en los bloques noise, encoder y decoder creados anteriormente encadenados. Tanto la slaida como la entrada de este bloque deben ser imágenes de 28*28


```python
#ingrese su código aquí
net = #mantenga este nombre de variable durante todo el notebook para que funcionen los tests
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


```python
#@title Test N° 2
#@markdown Ejecutar para confirmar que su código es correcto
assert net[0] == noise, "Tu primer bloque no es el correcto"
assert net[1] == encoder, "Tu segundo bloque no es el correcto"
assert net[2] == decoder, "Tu tercer bloque no es el correcto"
print("Al parecer está todo bien. Puedes avanzar al siguiente test")

```

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

En la siguiente celda deberá entrenar la red por 50 épocas


```python
#ingrese su código aquí



```


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


    
![png](01_fundamentos_deep_learning_Laboratorio_1_files/01_fundamentos_deep_learning_Laboratorio_1_99_0.png)
    


Ahora generemos una nueva red que en vez de eliminar un ruido producido por dropout, elimine un ruido gaussiano producido por la función `addGaussianNoise()`. Para hacerlo tenga en cuenta los siguientes detalles:


1.   El paquete nn.Module nos permite crear tanto capas como modelos completos personalizados. Utilice alguna de estas opciones para implementar este modelo.
2.   Tenga en cuenta que el ruido gaussiano solo debe agregarse a la imagen si el modelo está siendo entrenado. El atributo `training` de cualquier bloque que herede de nn.Module funciona como bandera que se pone en `True` si el modelo está entrenando y en `False` si ya está entrenada y se la usa para generar predicciones. El método `train()` pone esta bandera en `True` y el método `eval()` la pone en `False`




```python
## ingrese su código aquí

```

## Ejercicio 6: Entrenar el modelo con ruido gaussiano

Entrene el modelo creado en el ejercicio anterior


```python
#ingrese su código aquí
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
#inserte su código aquí
```
