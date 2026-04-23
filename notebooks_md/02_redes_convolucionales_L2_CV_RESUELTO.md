![Imgur](https://i.imgur.com/acSOZRh.png)

# Laboratorio N° 2


# Parte 1: Fundamentos de Redes Convolucionales

## Ejercicio 1


#Formas, Tamaños y Salidas

Los siguientes ejercicios están hechos para practicar, entender y aprender como cambian los tamaños de las salidas y entradas al aplicar padding y strides


```python
import torch
from torch import nn
```

1. Defina un tensor de $15 \times 15$ y aplique una convolución con kernel $5 \times 5$ de tal manera que a la salida tenga un solo mapa de características de $3 \times 3$. No debe utilizar padding
> NOTA: Recuerde que las convoluciones espera a la entrada tensores de la forma:
`X = [tamaño de minilote, numero de canales, alto en píxeles, ancho en píxeles]`


```python
#####SOLUCIÓN#######

x = torch.randn(1, 1, 15, 15)              # 1×1×15×15
conv1 = nn.Conv2d(in_channels=1,
                  out_channels=1,
                  kernel_size=5,
                  stride=5,     # ← clave
                  padding=0)

y = conv1(x)
print(y.shape)   # torch.Size([1, 1, 3, 3])

#####SOLUCIÓN#######

```

    torch.Size([1, 1, 3, 3])


2. Defina un tensor de $7 \times 7$ y aplique una convolución con kernel $3 \times 3$ de tal manera que a la salida tenga un solo mapa de características de $3 \times 3$.


```python
#####SOLUCIÓN#######

x = torch.randn(1, 1, 7, 7)                # 1×1×7×7
conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0)
y = conv2(x)
print(y.shape)   # torch.Size([1, 1, 3, 3])

#####SOLUCIÓN#######


```

    torch.Size([1, 1, 3, 3])


3. Dado un mapa de carácteristicas de la forma $100 \times 100$ y una convolución con kernel $7 \times 7$. ¿Cual era el tamaño original de la imagen de entrada?


```python
#####SOLUCIÓN#######

x = torch.randn(1, 1, 106, 106) #<---------RESPUESTA
conv2 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=0)
y = conv2(x)
print(y.shape)

#####SOLUCIÓN#######



```

    torch.Size([1, 1, 100, 100])


4. Para una imagen de $16\times16$, al aplicar una ventana de pooling de $2\times2$, ¿Cual es el tamaño esperado a la salida?


```python
#####SOLUCIÓN#######

x = torch.randn(1, 1, 16, 16)
pool = nn.MaxPool2d(kernel_size=2)
y = pool(x)
print(y.shape)   # torch.Size([1, 1, 8, 8])

#####SOLUCIÓN#######
```

    torch.Size([1, 1, 8, 8])


5. Para una imagen de $16\times16$, si aplicamos una ventana de pooling de $2\times2$, ¿Podemos obtener un salida de $4\times4$ usando strides?


```python
#####SOLUCIÓN#######

x = torch.randn(1, 1, 16, 16)
pool = nn.MaxPool2d(kernel_size=2, stride=4)
y = pool(x)
print(y.shape)   # torch.Size([1, 1, 8, 8])

#####SOLUCIÓN#######
```

    torch.Size([1, 1, 4, 4])


## Ejercicio 2

Entrenando kernels preexistentes

En la clase presentamos un pequeño pipeline para mostrar como entrenar un kernel. Nuestra intención en este ejercicio es replicar eso resultados para los siguientes ejemplos:

Es posible que necesite unas 100 épocas de entrenamiento para resolver el ejercicio.

### Operador Laplaciano


```python
X = torch.Tensor.uniform_(torch.Tensor(1,1,64,64))
K = torch.Tensor([  [0,  1, 0],
                    [1, -4, 1],
                    [0,  1, 0], ])

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[-2] - h + 1, X.shape[-1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[0][0][i:i + h, j:j + w] * K).sum() # producto de Haddamar
    return Y

Y = corr2d(X,K)

#####SOLUCIÓN#######

conv2d = nn.LazyConv2d(1, kernel_size=3, bias=False)

lr = 3e-1  # Learning rate

for i in range(100):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2 ## minimos cuadrados
    conv2d.zero_grad()
    l.mean().backward()
    # actualizamos los pesos
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 20 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data)
print(K)

#####SOLUCIÓN#######

```

    epoch 20, loss 918.581
    epoch 40, loss 121.710
    epoch 60, loss 16.184
    epoch 80, loss 2.160
    epoch 100, loss 0.289
    tensor([[[[ 8.4668e-04,  9.9256e-01, -1.0056e-03],
              [ 9.9378e-01, -3.9745e+00,  9.9316e-01],
              [-3.8226e-04,  9.9379e-01,  1.7304e-03]]]])
    tensor([[ 0.,  1.,  0.],
            [ 1., -4.,  1.],
            [ 0.,  1.,  0.]])


### Suavizador Gaussiano


```python
X = torch.Tensor.uniform_(torch.Tensor(1,1,64,64))
K = torch.Tensor([  [1.0, 2.0, 1.0],
                    [2.0, 4.0, 2.0],
                    [1.0, 2.0, 1.0], ])
K /= 16.0

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[-2] - h + 1, X.shape[-1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[0][0][i:i + h, j:j + w] * K).sum() # producto de Haddamar
    return Y

Y = corr2d(X,K)

#####SOLUCIÓN#######

conv2d = nn.LazyConv2d(1, kernel_size=3, bias=False)

lr = 3e-1  # Learning rate

for i in range(100):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2 ## minimos cuadrados
    conv2d.zero_grad()
    l.mean().backward()
    # actualizamos los pesos
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 20 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data)
print(K)

#####SOLUCIÓN#######

```

    epoch 20, loss 12.960
    epoch 40, loss 1.627
    epoch 60, loss 0.205
    epoch 80, loss 0.026
    epoch 100, loss 0.003
    tensor([[[[0.0628, 0.1258, 0.0610],
              [0.1244, 0.2505, 0.1245],
              [0.0647, 0.1242, 0.0622]]]])
    tensor([[0.0625, 0.1250, 0.0625],
            [0.1250, 0.2500, 0.1250],
            [0.0625, 0.1250, 0.0625]])



## Ejercicio 3

## Mejorando LeNet

Sabemos que LeNet fue un gran hito en su momento, pero también sabemos que hay modificaciones que podemos hacer a LeNet para mejorar su rendimiento.

Apliquelas, usando de referencia el Pipeline usado en la clase teórica.




```python
import torch
from torch import nn


```


```python
def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```


```python
NUM_CHANNEL1 = 6
NUM_CHANNEL2 = 16
NUM_MLP1 = 120
NUM_MLP2 = 84
num_classes = 10

## Este es la arquitectura original de LeNet. Aplique las mejoras comentadas en clase
model = nn.Sequential(
            nn.LazyConv2d(NUM_CHANNEL1, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(NUM_CHANNEL2, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(NUM_MLP1), nn.Sigmoid(),
            nn.LazyLinear(NUM_MLP2), nn.Sigmoid(),
            nn.LazyLinear(num_classes))

```


```python
#####SOLUCIÓN#######

model = nn.Sequential(
            nn.LazyConv2d(NUM_CHANNEL1, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(NUM_CHANNEL2, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(NUM_MLP1), nn.ReLU(),
            nn.LazyLinear(NUM_MLP2), nn.ReLU(),
            nn.LazyLinear(num_classes))

#####SOLUCIÓN#######
```

Revise cada una de las salidas de cada capa de su modelo.


```python
def layer_summary(net, X_shape):
    X = torch.randn(*X_shape)
    print("Entrada original:\t", X.shape)
    for layer in net:
        X = layer(X)
        print("Salida tras "+layer.__class__.__name__+':\t', X.shape)

layer_summary(model, (1, 1, 28, 28))
```

    Entrada original:	 torch.Size([1, 1, 28, 28])
    Salida tras Conv2d:	 torch.Size([1, 6, 28, 28])
    Salida tras ReLU:	 torch.Size([1, 6, 28, 28])
    Salida tras MaxPool2d:	 torch.Size([1, 6, 14, 14])
    Salida tras Conv2d:	 torch.Size([1, 16, 10, 10])
    Salida tras ReLU:	 torch.Size([1, 16, 10, 10])
    Salida tras MaxPool2d:	 torch.Size([1, 16, 5, 5])
    Salida tras Flatten:	 torch.Size([1, 400])
    Salida tras Linear:	 torch.Size([1, 120])
    Salida tras ReLU:	 torch.Size([1, 120])
    Salida tras Linear:	 torch.Size([1, 84])
    Salida tras ReLU:	 torch.Size([1, 84])
    Salida tras Linear:	 torch.Size([1, 10])


### Cargando los datos.

Como dijimos, vamos a trabajar con Fashion MNIST. Para ello cargaremos le dataset desde la biblioteca de torch.


```python
import torchvision
from torchvision import transforms
from torch.utils import data

def load_data_fashion_mnist(batch_size):
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    length = len(mnist_train)
    stop = int(len(mnist_train) * 0.7)
    mnist_val = [mnist_train[i] for i in range(stop,length)]
    mnist_train = [mnist_train[i] for i in range(stop)]
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=1),
            data.DataLoader(mnist_val, batch_size, shuffle=True,
                            num_workers=1),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=1))

batch_size = 1024
iter_train, iter_val, iter_test = load_data_fashion_mnist(batch_size)

```

También calcularemos el accuracy de nuestro modelo.


```python
def binary_accuracy(preds, y):

    # aproximamos al entero más cercano
    preds = torch.argmax(preds, dim=1)
    correct = (preds == y).float() #convertimos a flotante para la división
    acc = correct.sum() / len(correct)
    return acc
```

Definiremos una función de entrenamiento y evaluación como las que habíamos usado antes.


```python
def train(model, iterator, optimizer, criterion, device):

    model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch in iterator:

        image, label = batch
        image, label  = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        acc = binary_accuracy(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```


```python
def evaluate(model, iterator, criterion, device):

    model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    with torch.no_grad():

      for batch in iterator:

          image, label = batch
          image, label  = image.to(device), label.to(device)
          optimizer.zero_grad()
          output = model(image)
          loss = criterion(output, label)
          acc = binary_accuracy(output, label)
          epoch_loss += loss.item()
          epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

y una función para calcular el tiempo de cálculo


```python
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

###Entrenamiento


```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

```


```python
import math
N_EPOCHS = 20

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, iter_train, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, iter_val, criterion, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train acc.: {train_acc:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. acc.: {valid_acc:.3f}')
```

    Epoch: 01 | Time: 0m 1s
    	Train Loss: 1.804 | Train acc.: 0.411
    	 Val. Loss: 0.981 |  Val. acc.: 0.611
    Epoch: 02 | Time: 0m 0s
    	Train Loss: 0.796 | Train acc.: 0.700
    	 Val. Loss: 0.723 |  Val. acc.: 0.715
    Epoch: 03 | Time: 0m 0s
    	Train Loss: 0.687 | Train acc.: 0.733
    	 Val. Loss: 0.639 |  Val. acc.: 0.755
    Epoch: 04 | Time: 0m 0s
    	Train Loss: 0.612 | Train acc.: 0.761
    	 Val. Loss: 0.598 |  Val. acc.: 0.767
    Epoch: 05 | Time: 0m 0s
    	Train Loss: 0.563 | Train acc.: 0.782
    	 Val. Loss: 0.558 |  Val. acc.: 0.790
    Epoch: 06 | Time: 0m 0s
    	Train Loss: 0.529 | Train acc.: 0.798
    	 Val. Loss: 0.535 |  Val. acc.: 0.800
    Epoch: 07 | Time: 0m 0s
    	Train Loss: 0.508 | Train acc.: 0.812
    	 Val. Loss: 0.510 |  Val. acc.: 0.812
    Epoch: 08 | Time: 0m 0s
    	Train Loss: 0.491 | Train acc.: 0.818
    	 Val. Loss: 0.544 |  Val. acc.: 0.781
    Epoch: 09 | Time: 0m 0s
    	Train Loss: 0.482 | Train acc.: 0.821
    	 Val. Loss: 0.519 |  Val. acc.: 0.789
    Epoch: 10 | Time: 0m 0s
    	Train Loss: 0.460 | Train acc.: 0.833
    	 Val. Loss: 0.472 |  Val. acc.: 0.828
    Epoch: 11 | Time: 0m 0s
    	Train Loss: 0.439 | Train acc.: 0.842
    	 Val. Loss: 0.456 |  Val. acc.: 0.831
    Epoch: 12 | Time: 0m 0s
    	Train Loss: 0.429 | Train acc.: 0.845
    	 Val. Loss: 0.464 |  Val. acc.: 0.835
    Epoch: 13 | Time: 0m 0s
    	Train Loss: 0.427 | Train acc.: 0.848
    	 Val. Loss: 0.426 |  Val. acc.: 0.846
    Epoch: 14 | Time: 0m 0s
    	Train Loss: 0.399 | Train acc.: 0.857
    	 Val. Loss: 0.415 |  Val. acc.: 0.850
    Epoch: 15 | Time: 0m 0s
    	Train Loss: 0.391 | Train acc.: 0.862
    	 Val. Loss: 0.417 |  Val. acc.: 0.852
    Epoch: 16 | Time: 0m 0s
    	Train Loss: 0.386 | Train acc.: 0.865
    	 Val. Loss: 0.406 |  Val. acc.: 0.853
    Epoch: 17 | Time: 0m 0s
    	Train Loss: 0.365 | Train acc.: 0.870
    	 Val. Loss: 0.383 |  Val. acc.: 0.864
    Epoch: 18 | Time: 0m 0s
    	Train Loss: 0.366 | Train acc.: 0.868
    	 Val. Loss: 0.389 |  Val. acc.: 0.861
    Epoch: 19 | Time: 0m 0s
    	Train Loss: 0.380 | Train acc.: 0.869
    	 Val. Loss: 0.439 |  Val. acc.: 0.838
    Epoch: 20 | Time: 0m 0s
    	Train Loss: 0.393 | Train acc.: 0.860
    	 Val. Loss: 0.528 |  Val. acc.: 0.812



```python

model.load_state_dict(torch.load('tut2-model.pt'))

test_loss, test_acc = evaluate(model, iter_test, criterion ,device)

print(f'\t Test. acc: {test_loss:.3f} |  test. acc: {test_acc:.3f}')
```

    	 Test. acc: 0.393 |  test. acc: 0.860


## Ejercicio 4

## Cambiando el dataset.

Dado los resultados que tenemos con Fashion MNIST, es una buena idea tratar de probar otro dataset.



```python
import torch
from torch import nn

```


```python
def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

Como vamos a trabajar con imágenes en colores (3 canales RGB rojo, verde y azul), lo primero que haremos será aumentar el número de canales. Además, dado que CIFAR10 tiene imágenes de $32×32$ eliminaremos el padding de la primera capa convolucional para obtener mapas receptivos similares a los de FashionMNIST. Tambien cambiaremos el número de salidas de la primera capa densa


```python
NUM_CHANNEL1 = 20
NUM_CHANNEL2 = 50
NUM_MLP1 = 200
NUM_MLP2 = 80
num_classes = 10

## Defina aquí su modelo.

```


```python
def layer_summary(net, X_shape):
    X = torch.randn(*X_shape)
    print("Entrada original:\t", X.shape)
    for layer in net:
        X = layer(X)
        print("Salida tras "+layer.__class__.__name__+':\t', X.shape)

layer_summary(model, (1, 3, 32, 32))
```

    Entrada original:	 torch.Size([1, 3, 32, 32])
    Salida tras Conv2d:	 torch.Size([1, 20, 28, 28])
    Salida tras ReLU:	 torch.Size([1, 20, 28, 28])
    Salida tras MaxPool2d:	 torch.Size([1, 20, 14, 14])
    Salida tras Conv2d:	 torch.Size([1, 50, 10, 10])
    Salida tras ReLU:	 torch.Size([1, 50, 10, 10])
    Salida tras MaxPool2d:	 torch.Size([1, 50, 5, 5])
    Salida tras Flatten:	 torch.Size([1, 1250])
    Salida tras Linear:	 torch.Size([1, 200])
    Salida tras ReLU:	 torch.Size([1, 200])
    Salida tras Linear:	 torch.Size([1, 80])
    Salida tras ReLU:	 torch.Size([1, 80])
    Salida tras Linear:	 torch.Size([1, 10])


## Ejercicio 5



### Cargando los datos.

Ahora trabajaremos con CIFAR10 un dataset que tiene imagenes de en colores de $32\times32$. Defina una función que genere iteradores de entrenamiento, validación y prueba para este dataset


```python
import torchvision
from torchvision import transforms
from torch.utils import data

def load_data_fashion_mnist(batch_size):
  ## inserte aquí su código

batch_size = 1024
iter_train, iter_val, iter_test = load_data_fashion_mnist(batch_size)

```

    Files already downloaded and verified
    Files already downloaded and verified


También calcularemos el accuracy de nuestro modelo.


```python
def binary_accuracy(preds, y):

    # aproximamos al entero más cercano
    preds = torch.argmax(preds, dim=1)
    correct = (preds == y).float() #convertimos a flotante para la división
    acc = correct.sum() / len(correct)
    return acc
```

Definiremos una función de entrenamiento y evaluación como las que habíamos usado antes.


```python
def train(model, iterator, optimizer, criterion, device):

    model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch in iterator:

        image, label = batch
        image, label  = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        acc = binary_accuracy(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```


```python
def evaluate(model, iterator, criterion, device):

    model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    with torch.no_grad():

      for batch in iterator:

          image, label = batch
          image, label  = image.to(device), label.to(device)
          optimizer.zero_grad()
          output = model(image)
          loss = criterion(output, label)
          acc = binary_accuracy(output, label)
          epoch_loss += loss.item()
          epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

y una función para calcular el tiempo de cálculo


```python
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

###Entrenamiento


```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

```


```python
import math
N_EPOCHS = 50

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, iter_train, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, iter_val, criterion, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train acc.: {train_acc:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. acc.: {valid_acc:.3f}')
```

    Epoch: 01 | Time: 0m 1s
    	Train Loss: 2.062 | Train acc.: 0.230
    	 Val. Loss: 1.851 |  Val. acc.: 0.321
    Epoch: 02 | Time: 0m 1s
    	Train Loss: 1.757 | Train acc.: 0.364
    	 Val. Loss: 1.701 |  Val. acc.: 0.373
    Epoch: 03 | Time: 0m 1s
    	Train Loss: 1.639 | Train acc.: 0.403
    	 Val. Loss: 1.620 |  Val. acc.: 0.409
    Epoch: 04 | Time: 0m 1s
    	Train Loss: 1.560 | Train acc.: 0.432
    	 Val. Loss: 1.526 |  Val. acc.: 0.445
    Epoch: 05 | Time: 0m 1s
    	Train Loss: 1.501 | Train acc.: 0.453
    	 Val. Loss: 1.516 |  Val. acc.: 0.452
    Epoch: 06 | Time: 0m 1s
    	Train Loss: 1.440 | Train acc.: 0.480
    	 Val. Loss: 1.464 |  Val. acc.: 0.477
    Epoch: 07 | Time: 0m 1s
    	Train Loss: 1.401 | Train acc.: 0.494
    	 Val. Loss: 1.417 |  Val. acc.: 0.488
    Epoch: 08 | Time: 0m 1s
    	Train Loss: 1.373 | Train acc.: 0.504
    	 Val. Loss: 1.371 |  Val. acc.: 0.508
    Epoch: 09 | Time: 0m 1s
    	Train Loss: 1.385 | Train acc.: 0.505
    	 Val. Loss: 1.355 |  Val. acc.: 0.511
    Epoch: 10 | Time: 0m 1s
    	Train Loss: 1.300 | Train acc.: 0.535
    	 Val. Loss: 1.368 |  Val. acc.: 0.511
    Epoch: 11 | Time: 0m 1s
    	Train Loss: 1.280 | Train acc.: 0.541
    	 Val. Loss: 1.297 |  Val. acc.: 0.530
    Epoch: 12 | Time: 0m 1s
    	Train Loss: 1.243 | Train acc.: 0.553
    	 Val. Loss: 1.248 |  Val. acc.: 0.557
    Epoch: 13 | Time: 0m 1s
    	Train Loss: 1.217 | Train acc.: 0.569
    	 Val. Loss: 1.274 |  Val. acc.: 0.551
    Epoch: 14 | Time: 0m 1s
    	Train Loss: 1.195 | Train acc.: 0.577
    	 Val. Loss: 1.299 |  Val. acc.: 0.536
    Epoch: 15 | Time: 0m 1s
    	Train Loss: 1.179 | Train acc.: 0.581
    	 Val. Loss: 1.284 |  Val. acc.: 0.539
    Epoch: 16 | Time: 0m 1s
    	Train Loss: 1.169 | Train acc.: 0.582
    	 Val. Loss: 1.223 |  Val. acc.: 0.570
    Epoch: 17 | Time: 0m 1s
    	Train Loss: 1.130 | Train acc.: 0.602
    	 Val. Loss: 1.226 |  Val. acc.: 0.568
    Epoch: 18 | Time: 0m 1s
    	Train Loss: 1.114 | Train acc.: 0.608
    	 Val. Loss: 1.166 |  Val. acc.: 0.590
    Epoch: 19 | Time: 0m 1s
    	Train Loss: 1.081 | Train acc.: 0.618
    	 Val. Loss: 1.162 |  Val. acc.: 0.594
    Epoch: 20 | Time: 0m 1s
    	Train Loss: 1.058 | Train acc.: 0.628
    	 Val. Loss: 1.132 |  Val. acc.: 0.604
    Epoch: 21 | Time: 0m 1s
    	Train Loss: 1.051 | Train acc.: 0.631
    	 Val. Loss: 1.124 |  Val. acc.: 0.606
    Epoch: 22 | Time: 0m 1s
    	Train Loss: 1.023 | Train acc.: 0.640
    	 Val. Loss: 1.099 |  Val. acc.: 0.617
    Epoch: 23 | Time: 0m 1s
    	Train Loss: 1.014 | Train acc.: 0.645
    	 Val. Loss: 1.108 |  Val. acc.: 0.611
    Epoch: 24 | Time: 0m 1s
    	Train Loss: 0.995 | Train acc.: 0.651
    	 Val. Loss: 1.087 |  Val. acc.: 0.622
    Epoch: 25 | Time: 0m 1s
    	Train Loss: 0.995 | Train acc.: 0.652
    	 Val. Loss: 1.140 |  Val. acc.: 0.597
    Epoch: 26 | Time: 0m 1s
    	Train Loss: 0.986 | Train acc.: 0.653
    	 Val. Loss: 1.073 |  Val. acc.: 0.627
    Epoch: 27 | Time: 0m 1s
    	Train Loss: 0.965 | Train acc.: 0.660
    	 Val. Loss: 1.079 |  Val. acc.: 0.624
    Epoch: 28 | Time: 0m 1s
    	Train Loss: 0.946 | Train acc.: 0.667
    	 Val. Loss: 1.057 |  Val. acc.: 0.635
    Epoch: 29 | Time: 0m 1s
    	Train Loss: 0.933 | Train acc.: 0.674
    	 Val. Loss: 1.070 |  Val. acc.: 0.627
    Epoch: 30 | Time: 0m 1s
    	Train Loss: 0.925 | Train acc.: 0.675
    	 Val. Loss: 1.038 |  Val. acc.: 0.640
    Epoch: 31 | Time: 0m 1s
    	Train Loss: 0.906 | Train acc.: 0.684
    	 Val. Loss: 1.029 |  Val. acc.: 0.643
    Epoch: 32 | Time: 0m 1s
    	Train Loss: 0.888 | Train acc.: 0.688
    	 Val. Loss: 1.082 |  Val. acc.: 0.622
    Epoch: 33 | Time: 0m 1s
    	Train Loss: 0.878 | Train acc.: 0.690
    	 Val. Loss: 1.049 |  Val. acc.: 0.635
    Epoch: 34 | Time: 0m 1s
    	Train Loss: 0.874 | Train acc.: 0.692
    	 Val. Loss: 1.033 |  Val. acc.: 0.642
    Epoch: 35 | Time: 0m 1s
    	Train Loss: 0.847 | Train acc.: 0.702
    	 Val. Loss: 1.065 |  Val. acc.: 0.633
    Epoch: 36 | Time: 0m 1s
    	Train Loss: 0.844 | Train acc.: 0.707
    	 Val. Loss: 1.045 |  Val. acc.: 0.640
    Epoch: 37 | Time: 0m 1s
    	Train Loss: 0.817 | Train acc.: 0.711
    	 Val. Loss: 1.018 |  Val. acc.: 0.649
    Epoch: 38 | Time: 0m 1s
    	Train Loss: 0.811 | Train acc.: 0.717
    	 Val. Loss: 1.038 |  Val. acc.: 0.641
    Epoch: 39 | Time: 0m 1s
    	Train Loss: 0.810 | Train acc.: 0.716
    	 Val. Loss: 0.986 |  Val. acc.: 0.663
    Epoch: 40 | Time: 0m 1s
    	Train Loss: 0.791 | Train acc.: 0.722
    	 Val. Loss: 1.093 |  Val. acc.: 0.623
    Epoch: 41 | Time: 0m 1s
    	Train Loss: 0.808 | Train acc.: 0.715
    	 Val. Loss: 1.134 |  Val. acc.: 0.617
    Epoch: 42 | Time: 0m 1s
    	Train Loss: 0.792 | Train acc.: 0.724
    	 Val. Loss: 0.998 |  Val. acc.: 0.659
    Epoch: 43 | Time: 0m 1s
    	Train Loss: 0.767 | Train acc.: 0.733
    	 Val. Loss: 1.026 |  Val. acc.: 0.647
    Epoch: 44 | Time: 0m 1s
    	Train Loss: 0.751 | Train acc.: 0.737
    	 Val. Loss: 0.992 |  Val. acc.: 0.659
    Epoch: 45 | Time: 0m 1s
    	Train Loss: 0.735 | Train acc.: 0.743
    	 Val. Loss: 1.024 |  Val. acc.: 0.654
    Epoch: 46 | Time: 0m 1s
    	Train Loss: 0.725 | Train acc.: 0.746
    	 Val. Loss: 0.993 |  Val. acc.: 0.660
    Epoch: 47 | Time: 0m 1s
    	Train Loss: 0.717 | Train acc.: 0.748
    	 Val. Loss: 1.024 |  Val. acc.: 0.653
    Epoch: 48 | Time: 0m 1s
    	Train Loss: 0.703 | Train acc.: 0.755
    	 Val. Loss: 1.009 |  Val. acc.: 0.659
    Epoch: 49 | Time: 0m 1s
    	Train Loss: 0.694 | Train acc.: 0.757
    	 Val. Loss: 0.998 |  Val. acc.: 0.665
    Epoch: 50 | Time: 0m 1s
    	Train Loss: 0.673 | Train acc.: 0.766
    	 Val. Loss: 1.045 |  Val. acc.: 0.654



```python

model.load_state_dict(torch.load('tut2-model.pt'))

test_loss, test_acc = evaluate(model, iter_test, criterion ,device)

print(f'\t Test. loss: {test_loss:.3f} |  test. acc: {test_acc:.3f}')
```

    	 Test. acc: 0.986 |  test. acc: 0.661


# Parte 2: Arquitecturas Convolucionales Modernas

## Ejercicio 1

En este ejercicio vas a construir **ResNet-34** **exclusivamente** a partir de las clases auxiliares que ya vienen definidas en las celdas siguientes del notebook.  

El objetivo es que practiques cómo **componer** una red compleja reutilizando bloques previamente implementados, en lugar de volver a escribir todo desde cero.

---

El esquema (imagen de la primera celda) condensa la arquitectura original publicada en *He et al., 2015*:

| Símbolo | Significado | Ejemplo del diagrama |
|---------|-------------|----------------------|
| `7×7 conv, 64, /2` | Convolución 7×7 – 64 canales de **salida** – *stride* 2 | capa inicial (color crema) |
| `3×3 conv, 256` | Convolución 3×3 – 256 canales – *stride* 1 | cualquier bloque intermedio |
| Flecha curva continua | **Skip connection** simple (no cambia # de canales) | líneas sólidas |
| Flecha curva punteada | Skip con proyección **1×1 conv** (ajusta canales / stride) | líneas punteadas |


---

![img](https://i.imgur.com/5TiH4Qu.png)

#### 2.  Clases que **tenés que usar sí o sí**

| Clase | Rol dentro de la red | ¿Debo modificarla? |
|-------|----------------------|--------------------|
| `ResidualBlock` | Implementa un bloque residual *básico* (2 convs 3×3) con opción de proyección 1×1 | **No** |
| `Base` | Capa de entrada → `Conv7×7 + BN + ReLU + MaxPool` | **No** |
| `ResModule` | Agrupa *n* `ResidualBlock` y maneja automáticamente la proyección del **primer** bloque cuando cambia el # de canales o el *stride* | **No** |
|`ResNet`|Implementa una ResNet a partir de una lista de tuplas que define la arquitectura|**No**|
|`ResNet34`|Implementa una ResNet34 invocando a la clase `ResNet` |**Sí**|

---





```python
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```


```python
class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, X):
        return self.net(X)
```


```python
class ResModule(nn.Module):
    def __init__(self, num_residuals, num_channels, first_module=False):
        super().__init__()
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_module:
                blk.append(ResidualBlock(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(ResidualBlock(num_channels))
        self.net = nn.Sequential(*blk)

    def forward(self, X):
        return self.net(X)

```


```python
class ResNet(nn.Module):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(Base())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', ResModule(*b, first_module=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(init_cnn)

    def forward(self, X):
        return self.net(X)
```


```python
class ResNet34(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        #####SOLUCIÓN#######
        super().__init__(((3, 64), (4, 128), (6, 256), (3, 512)),
                       lr, num_classes)
        #####SOLUCIÓN#######


    def forward(self, X):
        return self.net(X)
```
