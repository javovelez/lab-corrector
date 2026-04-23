<a href="https://colab.research.google.com/github/institutohumai/cursos-python/blob/master/DeepLearning/5_Evaluacion_Modelos/2_Seleccion_Modelos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

## Selección de modelos

En el aprendizaje automático, generalmente seleccionamos nuestro modelo final después de evaluar varios modelos candidatos. Este proceso se llama *selección de modelo*. A veces los modelos sujetos a comparación
son de naturaleza fundamentalmente diferente
(por ejemplo, árboles de decisión frente a modelos lineales). En otras ocasiones, estamos comparando miembros de la misma clase de modelos que han sido entrenados con diferentes configuraciones de hiperparámetros.

Con los MLP, por ejemplo, es posible que deseemos comparar modelos con diferentes números de capas ocultas, diferentes números de unidades ocultas y varias opciones de funciones de activación aplicadas a cada capa oculta. Para determinar cuál es el mejor entre nuestros modelos candidatos, generalmente emplearemos un conjunto de datos de validación.

### Conjunto de datos de validación

En principio, no deberíamos tocar nuestro conjunto de prueba hasta que hayamos elegido todos nuestros hiperparámetros.
Si utilizáramos los datos de prueba en el proceso de selección del modelo, existe el riesgo de que podamos sobreajustar los datos de prueba. Entonces estaríamos en serios problemas. Si sobreajustamos nuestros datos de entrenamiento, siempre existe la evaluación de los datos de prueba para mantenernos honestos. Pero si sobreajustamos los datos de prueba, ¿cómo lo sabríamos?

Por lo tanto, nunca debemos confiar en los datos de prueba para la selección del modelo. Y, sin embargo, tampoco podemos confiar únicamente en los datos de entrenamiento para la selección del modelo porque no podemos estimar el error de generalización en los mismos datos que usamos para entrenar el modelo.


En aplicaciones prácticas, la imagen se vuelve más turbia. Si bien, idealmente, solo tocaríamos los datos de prueba una vez, para evaluar el mejor modelo o para comparar una pequeña cantidad de modelos entre sí, los datos de prueba del mundo real rara vez se descartan después de un solo uso. Rara vez podemos permitirnos un nuevo conjunto de prueba para cada ronda de experimentos.

La práctica común para abordar este problema
es dividir nuestros datos de tres maneras, incorporando un *conjunto de datos de validación* (o *conjunto de validación*) además de los conjuntos de datos de entrenamiento y prueba. 

![Imgur](https://i.imgur.com/jyEPbG9.png)

Un buen ejemplo para distinguir entre conjunto de prueba y de validación es lo que hace la plataforma Kaggle en sus competencias de aprendizaje automático. En sus inicios, Kaggle era solamente una plataforma de concursos donde las empresas publican problemas y los participantes compiten para construir el mejor algoritmo, generalmente con premios en efectivo. La organización d elos concursos consiste en:
1. el organizador debe separar su dataset en un conjunto de entrenamiento (que será publicado) y un conjunto de prueba (cuyas features serán publicadas, pero las etiquetas permanecerán ocultas). 
2. Los participantes podrán descargar los datos de entrenamiento y deberán elegir un modelo para presentar en la competencia. Para eso, deberán llevar adelante una selección de modelos generando un conjunto de validación a partir de los datos de entrenamiento.
3. Una vez seleccionado el modelo que mejor funcione con los datos de validación, se alimenta dicho modelo con las features del conjunto de prueba para obtener las etiquetas de prueba predichas por el modelo.
4. Se entregan las etiquetas de prueba predichas y el organizador las compara con las reales. El ganador es el modelo que menos erroes haya cometido. 
![Imgur](https://i.imgur.com/qA88YkJ.png)

De esta manera, los conjuntos de prueba y validación están bien diferenciados. El primero se usa para elegir el mejor modelo y el segundo se usa para evaluar el modelo elegido con datos que nunca vio en el entrenamiento.

A menos que se indique explícitamente lo contrario, en los experimentos de este curso en realidad estamos trabajando con lo que correctamente debería llamarse datos de entrenamiento y datos de validación, sin verdaderos conjuntos de prueba. Por lo tanto, reportado en cada experimento es realmente un accuracy de validación y no un verdadero accuracy del conjunto de pruebas.

### $K$*-fold cross-validation*

Cuando los datos de entrenamiento son escasos, es posible que ni siquiera podamos permitirnos mantener suficientes datos para constituir un conjunto de validación adecuado. Una solución popular a este problema es emplear $K$*-fold cross-validation*. Aquí, los datos de entrenamiento originales se dividen en $K$ subconjuntos que no se superponen. Luego, el entrenamiento y la validación del modelo se ejecutan $K$ veces, cada vez entrenando en $K-1$ subconjuntos y validando en un subconjunto diferente (el que no se usó para entrenar en esa ronda).
Finalmente, los errores de entrenamiento y validación se estiman promediando los resultados de los experimentos de $K$.

![Imgur](https://i.imgur.com/SpOFGyK.png)


```python
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader,ConcatDataset

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

```

#### Model
Definamos una red neuronal simple para el conjunto de datos MNIST.


```python
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

#### Función para reiniciar pesos 
Necesitamos restablecer los pesos del modelo para que cada fold de cross validation comience desde un estado inicial aleatorio y no aprenda de los folds anteriores. Podemos llamar a reset_weights() en todos los módulos hijos.


```python
def reset_weights(m):
  if type(m) == nn.Linear:
      nn.init.normal_(m.weight, std=0.01)
```

Modificamos ligeramente los pipelines de entrenamiento para que sea más ordenado... Todas las lineas para calcular la pérdida y mejorar los parámetros las ponemos en la función train y todas las que se encargan de calcular el accuracy, en la función test.


```python
def train(fold, model, device, loss, train_loader, optimizer, epoch):

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        l = loss(model(data), target).mean()
        l.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Fold/Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                fold,epoch, batch_idx * len(data), len(train_loader.sampler.indices),
                100. * batch_idx / len(train_loader), l.item()/len(target)))

```


```python
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
  print('\nTest set for fold {}:  Accuracy: {}/{} ({:.0f}%)\n'.format(
        fold, TestAcc, N,
        (100. * TestAcc) / N))
  return TestAcc / N

```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```

    Using device: cuda


#### Dataset
Necesitamos concatenar las partes de entrenamiento y prueba del dataset MNIST, que usaremos para entrenar el modelo. Hacer K-fold implica que nosotros mismos generemos las divisiones, por lo que no queremos que PyTorch lo haga por nosotros.


```python
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
```


```python
dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)

dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw
    



```python
dataset=ConcatDataset([dataset1,dataset2])
```

#### Clase KFold

KFold es una clase de la librería sklearn que nos puede ayudar a hacer cross validation. Para eso debemos instanciar el objeto kfold indicando la cantidad de folds que queremos en el atributo n_splits del constructor.


```python
kfold=KFold(n_splits=5,shuffle=True)



```

La clase KFold tiene un método llamado split() que es un iterator que recibe el dataset a separar y devuelve un tupla con dos listas de índices. La primera es la lista de índices de entrenamiento y la segunda es la lista de índices de testeo de ese fold.
	


```python
for train_idx,test_idx in kfold.split(dataset):
  print("train indices", len(train_idx), train_idx)
  print("test indices", len(test_idx), test_idx)
```

    train indices 56000 [    0     2     3 ... 69997 69998 69999]
    test indices 14000 [    1     6     9 ... 69990 69991 69996]
    train indices 56000 [    0     1     3 ... 69996 69997 69999]
    test indices 14000 [    2    10    16 ... 69982 69992 69998]
    train indices 56000 [    0     1     2 ... 69996 69997 69998]
    test indices 14000 [    3     7    25 ... 69987 69989 69999]
    train indices 56000 [    1     2     3 ... 69996 69998 69999]
    test indices 14000 [    0     5    13 ... 69983 69986 69997]
    train indices 56000 [    0     1     2 ... 69997 69998 69999]
    test indices 14000 [    4     8    21 ... 69993 69994 69995]


Ahora podemos generar los folds y entrenar nuestro modelo. Lo vamos a hacer definiendo un loop que itere sobre los folds especificando la lista de identificadores de los ejemplos de entrenamiento y validación para ese fold en particular. 

Dentro del loop hacemos un print del id del fold. Después, entrenamos muestreando los elementos de train y test con un SubsetRandomSampler. A esta clase se le puede pasar una lista con los índices de los elementos que debe muestrear del dataset.




```python
model = net1.to(device)
model.apply(reset_weights)
loss = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adadelta(model.parameters())
```


```python

batch_size=32
folds=5
epochs=5
acc = []
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
    fold_acc = test_accuracy(fold,model, loss, device,  testloader)
  acc.append(fold_acc)

  
```

    ------------fold no---------0----------------------
    Train Fold/Epoch: 0/1 [0/56000 (0%)]	Loss: 0.072632
    Train Fold/Epoch: 0/1 [16000/56000 (29%)]	Loss: 0.002493
    Train Fold/Epoch: 0/1 [32000/56000 (57%)]	Loss: 0.006823
    Train Fold/Epoch: 0/1 [48000/56000 (86%)]	Loss: 0.002683
    
    Test set for fold 0:  Accuracy: 13543.0/14000 (97%)
    
    Train Fold/Epoch: 0/2 [0/56000 (0%)]	Loss: 0.001721
    Train Fold/Epoch: 0/2 [16000/56000 (29%)]	Loss: 0.003222
    Train Fold/Epoch: 0/2 [32000/56000 (57%)]	Loss: 0.000781
    Train Fold/Epoch: 0/2 [48000/56000 (86%)]	Loss: 0.004093
    
    Test set for fold 0:  Accuracy: 13553.0/14000 (97%)
    
    Train Fold/Epoch: 0/3 [0/56000 (0%)]	Loss: 0.001739
    Train Fold/Epoch: 0/3 [16000/56000 (29%)]	Loss: 0.000202
    Train Fold/Epoch: 0/3 [32000/56000 (57%)]	Loss: 0.000488
    Train Fold/Epoch: 0/3 [48000/56000 (86%)]	Loss: 0.000310
    
    Test set for fold 0:  Accuracy: 13683.0/14000 (98%)
    
    Train Fold/Epoch: 0/4 [0/56000 (0%)]	Loss: 0.000024
    Train Fold/Epoch: 0/4 [16000/56000 (29%)]	Loss: 0.012890
    Train Fold/Epoch: 0/4 [32000/56000 (57%)]	Loss: 0.001868
    Train Fold/Epoch: 0/4 [48000/56000 (86%)]	Loss: 0.004358
    
    Test set for fold 0:  Accuracy: 13671.0/14000 (98%)
    
    Train Fold/Epoch: 0/5 [0/56000 (0%)]	Loss: 0.000007
    Train Fold/Epoch: 0/5 [16000/56000 (29%)]	Loss: 0.000599
    Train Fold/Epoch: 0/5 [32000/56000 (57%)]	Loss: 0.001722
    Train Fold/Epoch: 0/5 [48000/56000 (86%)]	Loss: 0.001244
    
    Test set for fold 0:  Accuracy: 13647.0/14000 (97%)
    
    ------------fold no---------1----------------------
    Train Fold/Epoch: 1/1 [0/56000 (0%)]	Loss: 0.070966
    Train Fold/Epoch: 1/1 [16000/56000 (29%)]	Loss: 0.014835
    Train Fold/Epoch: 1/1 [32000/56000 (57%)]	Loss: 0.011153
    Train Fold/Epoch: 1/1 [48000/56000 (86%)]	Loss: 0.001521
    
    Test set for fold 1:  Accuracy: 13271.0/14000 (95%)
    
    Train Fold/Epoch: 1/2 [0/56000 (0%)]	Loss: 0.002682
    Train Fold/Epoch: 1/2 [16000/56000 (29%)]	Loss: 0.000077
    Train Fold/Epoch: 1/2 [32000/56000 (57%)]	Loss: 0.006677
    Train Fold/Epoch: 1/2 [48000/56000 (86%)]	Loss: 0.004831
    
    Test set for fold 1:  Accuracy: 13550.0/14000 (97%)
    
    Train Fold/Epoch: 1/3 [0/56000 (0%)]	Loss: 0.006551
    Train Fold/Epoch: 1/3 [16000/56000 (29%)]	Loss: 0.004006
    Train Fold/Epoch: 1/3 [32000/56000 (57%)]	Loss: 0.000227
    Train Fold/Epoch: 1/3 [48000/56000 (86%)]	Loss: 0.000017
    
    Test set for fold 1:  Accuracy: 13609.0/14000 (97%)
    
    Train Fold/Epoch: 1/4 [0/56000 (0%)]	Loss: 0.001416
    Train Fold/Epoch: 1/4 [16000/56000 (29%)]	Loss: 0.000058
    Train Fold/Epoch: 1/4 [32000/56000 (57%)]	Loss: 0.008581
    Train Fold/Epoch: 1/4 [48000/56000 (86%)]	Loss: 0.000414
    
    Test set for fold 1:  Accuracy: 13633.0/14000 (97%)
    
    Train Fold/Epoch: 1/5 [0/56000 (0%)]	Loss: 0.000285
    Train Fold/Epoch: 1/5 [16000/56000 (29%)]	Loss: 0.000096
    Train Fold/Epoch: 1/5 [32000/56000 (57%)]	Loss: 0.000000
    Train Fold/Epoch: 1/5 [48000/56000 (86%)]	Loss: 0.000004
    
    Test set for fold 1:  Accuracy: 13630.0/14000 (97%)
    
    ------------fold no---------2----------------------
    Train Fold/Epoch: 2/1 [0/56000 (0%)]	Loss: 0.075589
    Train Fold/Epoch: 2/1 [16000/56000 (29%)]	Loss: 0.000583
    Train Fold/Epoch: 2/1 [32000/56000 (57%)]	Loss: 0.002770
    Train Fold/Epoch: 2/1 [48000/56000 (86%)]	Loss: 0.003739
    
    Test set for fold 2:  Accuracy: 13531.0/14000 (97%)
    
    Train Fold/Epoch: 2/2 [0/56000 (0%)]	Loss: 0.001064
    Train Fold/Epoch: 2/2 [16000/56000 (29%)]	Loss: 0.000200
    Train Fold/Epoch: 2/2 [32000/56000 (57%)]	Loss: 0.007270
    Train Fold/Epoch: 2/2 [48000/56000 (86%)]	Loss: 0.001145
    
    Test set for fold 2:  Accuracy: 13599.0/14000 (97%)
    
    Train Fold/Epoch: 2/3 [0/56000 (0%)]	Loss: 0.000017
    Train Fold/Epoch: 2/3 [16000/56000 (29%)]	Loss: 0.001019
    Train Fold/Epoch: 2/3 [32000/56000 (57%)]	Loss: 0.004924
    Train Fold/Epoch: 2/3 [48000/56000 (86%)]	Loss: 0.003546
    
    Test set for fold 2:  Accuracy: 13616.0/14000 (97%)
    
    Train Fold/Epoch: 2/4 [0/56000 (0%)]	Loss: 0.000115
    Train Fold/Epoch: 2/4 [16000/56000 (29%)]	Loss: 0.000201
    Train Fold/Epoch: 2/4 [32000/56000 (57%)]	Loss: 0.005122
    Train Fold/Epoch: 2/4 [48000/56000 (86%)]	Loss: 0.003086
    
    Test set for fold 2:  Accuracy: 13673.0/14000 (98%)
    
    Train Fold/Epoch: 2/5 [0/56000 (0%)]	Loss: 0.000026
    Train Fold/Epoch: 2/5 [16000/56000 (29%)]	Loss: 0.000023
    Train Fold/Epoch: 2/5 [32000/56000 (57%)]	Loss: 0.000010
    Train Fold/Epoch: 2/5 [48000/56000 (86%)]	Loss: 0.001503
    
    Test set for fold 2:  Accuracy: 13693.0/14000 (98%)
    
    ------------fold no---------3----------------------
    Train Fold/Epoch: 3/1 [0/56000 (0%)]	Loss: 0.074771
    Train Fold/Epoch: 3/1 [16000/56000 (29%)]	Loss: 0.000467
    Train Fold/Epoch: 3/1 [32000/56000 (57%)]	Loss: 0.001439
    Train Fold/Epoch: 3/1 [48000/56000 (86%)]	Loss: 0.006780
    
    Test set for fold 3:  Accuracy: 13403.0/14000 (96%)
    
    Train Fold/Epoch: 3/2 [0/56000 (0%)]	Loss: 0.006133
    Train Fold/Epoch: 3/2 [16000/56000 (29%)]	Loss: 0.000120
    Train Fold/Epoch: 3/2 [32000/56000 (57%)]	Loss: 0.001035
    Train Fold/Epoch: 3/2 [48000/56000 (86%)]	Loss: 0.000251
    
    Test set for fold 3:  Accuracy: 13660.0/14000 (98%)
    
    Train Fold/Epoch: 3/3 [0/56000 (0%)]	Loss: 0.000125
    Train Fold/Epoch: 3/3 [16000/56000 (29%)]	Loss: 0.001907
    Train Fold/Epoch: 3/3 [32000/56000 (57%)]	Loss: 0.000069
    Train Fold/Epoch: 3/3 [48000/56000 (86%)]	Loss: 0.000079
    
    Test set for fold 3:  Accuracy: 13626.0/14000 (97%)
    
    Train Fold/Epoch: 3/4 [0/56000 (0%)]	Loss: 0.001781
    Train Fold/Epoch: 3/4 [16000/56000 (29%)]	Loss: 0.000037
    Train Fold/Epoch: 3/4 [32000/56000 (57%)]	Loss: 0.000238
    Train Fold/Epoch: 3/4 [48000/56000 (86%)]	Loss: 0.000378
    
    Test set for fold 3:  Accuracy: 13693.0/14000 (98%)
    
    Train Fold/Epoch: 3/5 [0/56000 (0%)]	Loss: 0.000816
    Train Fold/Epoch: 3/5 [16000/56000 (29%)]	Loss: 0.001517
    Train Fold/Epoch: 3/5 [32000/56000 (57%)]	Loss: 0.004022
    Train Fold/Epoch: 3/5 [48000/56000 (86%)]	Loss: 0.000001
    
    Test set for fold 3:  Accuracy: 13692.0/14000 (98%)
    
    ------------fold no---------4----------------------
    Train Fold/Epoch: 4/1 [0/56000 (0%)]	Loss: 0.079523
    Train Fold/Epoch: 4/1 [16000/56000 (29%)]	Loss: 0.017008
    Train Fold/Epoch: 4/1 [32000/56000 (57%)]	Loss: 0.006527
    Train Fold/Epoch: 4/1 [48000/56000 (86%)]	Loss: 0.001305
    
    Test set for fold 4:  Accuracy: 13476.0/14000 (96%)
    
    Train Fold/Epoch: 4/2 [0/56000 (0%)]	Loss: 0.006741
    Train Fold/Epoch: 4/2 [16000/56000 (29%)]	Loss: 0.003389
    Train Fold/Epoch: 4/2 [32000/56000 (57%)]	Loss: 0.000578
    Train Fold/Epoch: 4/2 [48000/56000 (86%)]	Loss: 0.001750
    
    Test set for fold 4:  Accuracy: 13631.0/14000 (97%)
    
    Train Fold/Epoch: 4/3 [0/56000 (0%)]	Loss: 0.001076
    Train Fold/Epoch: 4/3 [16000/56000 (29%)]	Loss: 0.000317
    Train Fold/Epoch: 4/3 [32000/56000 (57%)]	Loss: 0.000239
    Train Fold/Epoch: 4/3 [48000/56000 (86%)]	Loss: 0.004561
    
    Test set for fold 4:  Accuracy: 13674.0/14000 (98%)
    
    Train Fold/Epoch: 4/4 [0/56000 (0%)]	Loss: 0.000714
    Train Fold/Epoch: 4/4 [16000/56000 (29%)]	Loss: 0.000081
    Train Fold/Epoch: 4/4 [32000/56000 (57%)]	Loss: 0.000414
    Train Fold/Epoch: 4/4 [48000/56000 (86%)]	Loss: 0.002501
    
    Test set for fold 4:  Accuracy: 13663.0/14000 (98%)
    
    Train Fold/Epoch: 4/5 [0/56000 (0%)]	Loss: 0.000166
    Train Fold/Epoch: 4/5 [16000/56000 (29%)]	Loss: 0.000272
    Train Fold/Epoch: 4/5 [32000/56000 (57%)]	Loss: 0.001221
    Train Fold/Epoch: 4/5 [48000/56000 (86%)]	Loss: 0.000299
    
    Test set for fold 4:  Accuracy: 13694.0/14000 (98%)
    



```python
print('El accuracy de cada fold es el siguiente {} y el accuracy promedio del modelo es {}'.format(
                acc, np.array(acc).mean()))
```

    El accuracy de cada fold es el siguiente [0.9747857142857143, 0.9735714285714285, 0.9780714285714286, 0.978, 0.9781428571428571] y el accuracy promedio del modelo es 0.9765142857142857

