<a href="https://colab.research.google.com/github/institutohumai/cursos-python/blob/master/DeepLearning/2_RedesDeUnaCapa/1_regresion_lineal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# Redes Neuronales de Una Capa

# Regresión lineal desde cero

Si estás acá en principio sabés que es una regresión lineal. Una regresión lineal una forma de analizar como ciertas variables dependen de otras.



![Linear_regression.svg](https://i.imgur.com/P3WseWv.png)

Una regresión lineal, nos relaciona variables independientes o features con una variable dependiente objetivo. Por medio de sumás y productos de las diferentes cantidades independientes, se busca obtener la variable dependiente.

Consideremos la llamada fórmula de Dulong. La formula de Dulong es un resultado experimental surgido de analizar la energía liberda en la combustión de combustibles fósiles. La formula predice el valor de energía liberada en función la proporción de cada elemento en el combustible.

$$E\left[\frac{kJ}{g}\right] = 38,2C + 84,9 (H − \frac{O}{8})  − 0,5l -  0,62s$$

Donde $C$ es la proporción en masa de carbono en el combustible, $H$ la proporción de hidrógeno, $O$ la de oxígeno, $l$ vale 1 solamente si el combustible es líquido, $s$ vale 1 solamente si el combustible es sólido

Por ejemplo, al quemar gas metáno se obtiene las siguientes proporciones, $C = 0.75$, $H = 0.25$ y $O = 0.00$

En nuestra fórmula eso nos devuelve $49.87\frac{kJ}{g}$ frente al valor $50.01\frac{kJ}{g}$ reportado en tablas.

> Nota: esos valores provienen de lo siguiente:
> * Fórmula del metano: $CH_4$
> * Masa molar del metano: $16 \frac{g}{mol}$
> * Masa molar del carbono: $12 \frac{g}{mol}$
> * Masa molar del hidrógeno: $1 \frac{g}{mol}$
> * $C = \frac{12}{16} = 0.75$
> * $C = 4\times\frac{1}{16} = 0.25$ porque hay 4 hidrógenos.

El punto de este comentario no es discutir un resultado de termódinámica química, sino señalar que la regresión lineal es una técnica usada desde hace años en areas de las más variopintas. Sin ir más lejos, la formula de Dulong es un resultado del siglo XIX que sigue teniendo utilidad. Tal es así que los coeficientes de la versión que hemos presentado corresponden a un resultado de [2016](https://www.sciencedirect.com/science/article/pii/S0378382016302995?via%3Dihubtps://) La única diferencia con el otro modelo presentado es que aquí hemos decidido usar una notación con one-hot vectors para el estado del combustible.

Siguiento el ejemplo anterior consideremos que hemos estudiado solo dos combustibles, el gas metano y el alcohol etílico.

||$C$|$H$|$O$|gas|líquido|sólido|$E$
|---|---|---|---|---|---|---|---|
metano|0.75|0.25|0.00|1|0|0|50.01|
alcohol etílico ($C_2H_5OH$)|0.52|0.13|0.35|0|1|0|26.70|

Notar que para repersentar el estado de agregación del combustible, hemos usado una codificación de one hot vectors.

En la columna $E$ reportamos el valor medido en laboratorio. Consideremos por ahora solo los valores nuestras variables independientes y pongamos la dentro de una matriz:

\begin{align}
     X = \left[\begin{array}{cccccc}
     0.75&0.25&0.00&1&0&0\\
     0.52&0.13&0.35&0&1&0
     \end{array}\right]
     \end{align}

Consideremos también un vector donde guardaremos los valores reales o los ground truth de nuestos datos

\begin{align}
     y = \left[\begin{array}{c}
     50.01\\
     26.7
     \end{array}\right]
     \end{align}

Esta matriz $X$, se conoce como matriz de diseño y nos permite guardar toda la información de los ejemplos que quiesieramos estudiar. Además, la matriz $X$ tiene una propiedad interesante. Condiremos lo coeficientes de la fórmula de Dulong y guardemoslos en un vector $w$

\begin{align}
     w^T = \left[\begin{array}{cccccc}
     38.2&84.9&-10.6125&0&-0.5&-0.62
     \end{array}\right]
     \end{align}

por que $-\frac{84.9}{8} = -1.6125$

Consideremos ahora el vector:

$$\hat{y} = Xw$$
\begin{align}
     \hat{y} = \left[\begin{array}{cccccc}
     0.75&0.25&0.00&1&0&0\\
     0.52&0.13&0.35&0&1&0
     \end{array}\right]\left[\begin{array}{c}
     38.2\\84.9\\-10.6125\\0\\-0.5\\-0.62
     \end{array}\right]
     \end{align}
\begin{align}
     \hat{y} = \left[\begin{array}{cc}
     49.87\\
     26.69
     \end{array}\right]
     \end{align}

Al guardar nuestros datos y nuestros parametros en matrices, ahora podemos calcular nuestras predicciones como una mera multiplicación de matrices. Además podemos estimar el error absoluto entre nuestra predicción y el valor real como una operación matricial.

$$e=\hat{y} - y$$

De más está decir que de lo anterior podemos tratar de calcular la varianza de los valores en el vector $e$, suponiendo que un buen modelo debe tener dispersión media igual a 0 con respecto a los valore reales.

$$Var(e)$$

La formula anterior y todo el analisis es para señalar que es una neurona artificial





La fórmula de Dulong, como toda regresión lineal, tiene una serie de variables independientes que influyen en el resultado de una variable dependiente. En el caso anterior tenemos, proporcion de diferentes átomos y estado de agregación del combustible (sólido, líquido, gas).

Esta estructura es similar a una neurona que recibe 6 estimulos y produce una única respuesta. Es por esto que decimos que es una neurona. 

Minimizar la varianza de nuestro error abosoluto es equivalente a pedir un ajuste lineal por mínimos cuadrados. Es decir, el entrenamiento de nuestra neurona, consistirá en encontrar los parametos $w$ que al aplicarlos sobre nuestros datos $X$ nos permitan obtener las mejores predicciones $\hat{y}$ de tal manera que se acerque lo más posible a nuestros valores reales $y$ En este contexto, encontrar esos parametros es equivalente a minimizar la varianza. A la cantidad a minimizar la llamaremos **función de pérdida**

Hemos elegido un caso de análisis como es el calor liberado por un combustible en función de sus constituyentes para señalar que el principio con el que opera la técnica es tan general que puede aplicarse a un monton de otras areas. Sin ir más lejos, podríamos ir un paso más alla y en lugar de entrenar solo una neurona para que nos dé la energía liberada, podríamos entrenar una segunda neurona que nos dé la masa molar del combustible. O una tercer neurona para que nos entregue otra propiedad del combustible, como por ejemplo el índice de refracción.

En caso de tener varias neuronas, nuestros parámetros deberían ser almacenados en una matriz de pesos. Es decir:

$$\hat{y} = XW$$
\begin{align}
     \hat{y} = \left[\begin{array}{cccccc}
     0.75&0.25&0.00&1&0&0\\
     0.52&0.13&0.35&0&1&0
     \end{array}\right]\left[\begin{array}{cc}
     38.2&1\\84.9&1\\-1.6125&1\\0&-1\\-0.5&1\\-0.62&2
     \end{array}\right]
     \end{align}
\begin{align}
     \hat{y} = \left[\begin{array}{cc}
     49.87&0\\
     26.69&2
     \end{array}\right]
     \end{align}

Donde la segunda columna corresponde a una segunda neurona que aprendió a devolver alguna otra propiedad de los combustibles

Esperamos que con lo aquí descripto, se entienda que es una neurona, red de una capa de neuronas y en que consiste encontrar los parametros óptimos al problema.

## Presentando el pipeline

Ahora, lo que describiremos será como es el proceso general para entrenar una red:

1. Carga de los datos
1. Separación de los datos en lotes
1. Inicialización de parámetros
1. Definición del modelo
1. Definición de la función de pérdida
1. Definición del algoritmo de optimización

Para nuestro primer ejemplo lo que haremos será trabajar con datos sintéticos. Es decir, tomaremos los datos generados de un modelo lineal. Nuestra intención con esto es múltiple:

* Queremos mostrar un ejemplo que nos permita entender el significado de los parámetros de nuestro modelo.
* Queremos saber que tan buenas son nuestras estimaciones
* Queremos usar un modelo sencillo que nos permita analizar cada paso del pipeline. 

Este último punto es el principal motivo de esta parte. Por lo general, los frameworks de deep learning tiene multiples herramientas que nos permite simplificar cada uno de los pasos. Sin embargo, también es común que necesitemos ajustar detalles del modelo que usaremos. Es en este sentido que "reinventar la rueda", nos puede ayudar entender como funcionan las herramientas preexistentes en los frameworks que usaremos.


```python
%matplotlib inline
import random
import torch
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
```

### "Dataset"

Comencemos con un modelo lineal sencillo al que añadiremos ruido gaussiano

**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**

$$\mathbf{w} = [2, -3.4]^\top$$
$$b = 4.2$$



```python
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
```


```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

Es importante ver cual es la dimensionalidad de nuestros features y nuestras etiquetas.



```python
print('features:', features.shape,'\nlabel:', labels.shape)
```

    features: torch.Size([1000, 2]) 
    label: torch.Size([1000, 1])


Podemos graficar la etiqueta y una de las features para ver este comportamiento lineal



```python
backend_inline.set_matplotlib_formats('svg')
plt.rcParams['figure.figsize'] = (3.5, 2.5)
# Punto y coma para mostrar solo la figura
plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);

```


    
![svg](01_02_redes_de_una_capa_02_01_regresion_lineal_files/01_02_redes_de_una_capa_02_01_regresion_lineal_23_0.svg)
    


### Cargando los datos

Dado que el entrenamiento se hace usando muchos mini-lotes de datos, es conveniente tener una función que se encarga de generar estos minilotes segun los necesitemos. 

Para esto necesitamos una función que tome nuestra matriz de diseño, tome nuestras etiquetas y nos genere lotes para el entrenamiento de un tamaño dado.




```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # aleatorizamos el orden de los datos
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```


```python
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

    tensor([[ 0.2888, -0.5761],
            [-3.7024,  0.2970],
            [ 1.0895,  0.0453],
            [-0.8273, -0.8839],
            [-1.4767,  1.2754],
            [-0.4595, -0.3420],
            [-1.5775, -1.8259],
            [-1.0123,  0.8654],
            [ 1.8892, -0.8451],
            [ 0.0615,  0.4675]]) 
     tensor([[ 6.7351],
            [-4.2164],
            [ 6.2268],
            [ 5.5470],
            [-3.0910],
            [ 4.4381],
            [ 7.2601],
            [-0.7705],
            [10.8543],
            [ 2.7222]])


### Valores iniciales de nuestro modelo

Dado que estamos estamos buscando mínimos de una fución de pérdida, podemos elegir iniciar con cualquier valor. Luego nuestro optimizador se encargará de encontrar los mínimos adecuados.


```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

Con estos parametros iniciales, estamos en condiciones de empezar a entrenar nuestra red. Es decir, buscar los parámetros que mejor representen el comportamiento de nuestros datos.

Debemos recordar que detras del uso de descenso por gradiente, usamos herramientas de diferenciación automática para nuestros problemas.


### Definiendo el modelo.

En este caso, nuestro modelo será análogo al usado para generar nuestros datos, es decir: 

$$\mathbf{y} = \mathbf{Xw} + b$$




```python
def linreg(X, w, b):
    return torch.matmul(X, w) + b
```

### Definiendo la función de pérdida

Como estamos haciendo una regresión lineal, sabemos que lo que minimizamos son es la cantidad llamada **mínimos cuadrados**


```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### Definiendo el algoritmo de optimización

A continuación mostramos un pequeño ejemplo de funciona descenso gradiente estocástico 



```python
def sgd(params, lr, batch_size):
    """Usamos minilotes"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

### Entrenamiento

A continuación esbozamos como es nuestro ciclo de entrenamiento

* Iniciamos nuestros parametros. $(\mathbf{w}, b)$
* Repetimos hasta concluir
    * Calculamos la función de pérdida ${L} \leftarrow \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Calculamos el gradiente con minilotes $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} L$
    * actualizamos los parámetros. $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
    > Nota: estos últimos dos pasos pasos los hace la función `sdg` definida arriba.

Llamamos **época** a cada vez que iteramos sobre todos nuestros datos. Por otro lado el parametro $η$ es lo que llamamos **tasa de aprendizaje** o **learning rate**. Este valor nos dice que tanto nos moveremos en la dirección hacia donde esta el mínimo. Tanto la cantidad de épocas a recorrer como la tasa de aprendizaje son **hiperparametros**. Encontrar lo hiperparámetros apropiados para nuestros datos y modelos no es una tarea sencilla. Por ahora daremos valores arbitrarios, pero aprender a encontrar valores correctos es todo un arte.



```python
lr = 0.03
num_epochs = 5
net = linreg
loss = squared_loss
```


```python
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # Funcion de perdida para nuestro minilote con `X` e `y`
        l = loss(net(X, w, b), y)  
        # Gradiente de la funcion l con respeto a [`w`, `b`]
        # Recordar w es un "vector" de parametros
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

    epoch 1, loss 0.000053
    epoch 2, loss 0.000053
    epoch 3, loss 0.000053
    epoch 4, loss 0.000053
    epoch 5, loss 0.000053


Como nuestros datos son sintéticos, podemos comparar nuestras estimaciones con los valores reales.



```python
print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

    error in estimating w: tensor([-0.0002,  0.0007], grad_fn=<SubBackward0>)
    error in estimating b: tensor([0.0006], grad_fn=<RsubBackward1>)


# Regresión lineal concisa.

En el notebook anterior, vimos un ejemplo de como implementar una red neuronal desde cero. Sin embargo, hacer esto es una mala idea. La principal razón por la que es una mala idea, es que muchas de las cosas que hicimos consisten en "reinventar la rueda". Hay bibliotecas que ya tienen herramientas para hacer lo que ya hicimos. Además, nuestra implementación puede no ser la más eficiente. Es decir: la implementación usada puede generar tiempos de espera que podrían ser evitados si nuestro código estuviera implementado de manera distinta. Por esta razón, es siempre recomendable usar las bibliotecas preexistentes.

Recordemos que el ejemplo anterior estaba pensado para que le perdamos el miedo a las bibliotecas preexistentes, para que entendamos como funcionan y para aprender a implementar cosas nuevas (si llegamos a necesitarlo)

Veamos entonces como implementariamos todo lo anterior haciendo uso de la biblioteca ``pytorch``

## Datos sintéticos



```python
import numpy as np
import torch
from torch.utils import data
```

Misma función que usamos anteriormente


```python
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
```


```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

## Cargando nuestros datos

En este caso, podemos enviar nuestros datos diferentes metodos preexistentes de `pytorch` para generar nuestro minilotes.

Además podemos pedir nos mezcle nuestros datos o que los deje tal cual



```python
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```


```python
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Queremos ver como se generan nuestros minilotes. Para esto debemos poder imprimierlos por pantalla. A diferencia de la implementación anterior, el método `DataLoader`, no genera un iterable, por esto debemos convertirlo en uno y recorrerlo segun necesitemos



```python
next(iter(data_iter))
```




    [tensor([[ 0.8042,  1.2084],
             [ 0.0308, -2.8017],
             [-1.0535, -1.1924],
             [-1.1401,  0.7265],
             [-1.7502,  0.4037],
             [-1.1868,  1.3315],
             [-0.9479,  0.8272],
             [ 0.4404,  0.3488],
             [ 0.1959, -0.0737],
             [-0.8688,  0.2807]]), tensor([[ 1.7018],
             [13.7689],
             [ 6.1498],
             [-0.5491],
             [-0.6562],
             [-2.7091],
             [-0.5121],
             [ 3.8959],
             [ 4.8421],
             [ 1.5083]])]



## Definiendo el modelo

A continuación presentamos la versión concisa de nuestro modelo, luego la discutiremos.



```python
# `nn` = neural networks, redes neuronales
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
```

La clase `Sequential` define todas las capas a aplicar de manera secuencial en nuestro modelo. Por ahora, como trabajamos con regresión lineal solo usaremos una capa. Sin embargo esta capa es lo que se llama una capa *totalmente conectada*. Es decir, esta representada por una matriz que aplica sobre vector de features. Al aplicar esta matriz encontramos la salida de nuestra neurona. En este caso, este tipo de capas se las conoce como `Linear` y reciben como entrada `(<numero_de_entradas>, <numero_de_salidas>)`. Para nuestro modelo, esto son nuestras 2 features y nuestra etiqueta.


¿Que es una capa densa?
------------------


una capa densa o completamente conectada es la forma más básica de una red neuronal. Cada entrada influencia a cada salida de acuerdo a los pesos. Si nuestro modelo tiene $m$ entradas y $n$ salidas, la matriz de pesos sera $m \times n$. De igual modo el vector de sesgos o bias tendra dimensión $n$


```python
import torch

lin = torch.nn.Linear(2, 1)
x = torch.rand(1, 2)
print('entrada:')
print(x)

print('\n\nPesos y parametros:')
for param in lin.named_parameters():
    print(param)

y = lin(x)
print('\n\nsalida')
print(y)

# Al hacer la multiplicacion matricial correspondiente obtenemos nuesta salida.
x @ lin.weight.T + lin.bias
```

    entrada:
    tensor([[0.0123, 0.9261]])
    
    
    Pesos y parametros:
    ('weight', Parameter containing:
    tensor([[0.5161, 0.6590]], requires_grad=True))
    ('bias', Parameter containing:
    tensor([0.4101], requires_grad=True))
    
    
    salida
    tensor([[1.0267]], grad_fn=<AddmmBackward0>)





    tensor([[1.0267]], grad_fn=<AddBackward0>)



## Inicialización de parametros de nuestro modelo.

Por lo general, los frameworks prexistentes tienen implementaciones por defecto para inicializar los parámetros. Sin embargo, queremos iniciarlos de manera similar a la anterior.

Para ellos accedemos a la primera (y única capa) usando `net[0]`. Luego accedemos a los pesos y los sesgos con `weight.data` and `bias.data`. Finalmente rellenamos los valores con lo que teníamos pensado usar.



```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```




    tensor([0.])



## Definiendo la función de pérdida



```python
loss = nn.MSELoss()
```

## Definiendo el algoritmo de optimización


La principal diferencia con lo que hicimos antes, es que solamente debemos pasarle a nuestro `SDG`, los parametros a optimizar. El resto de los detalles ya son manejados por la implementación de `pytorch`. En este caso también estamos pasando la tasa de aprendizaje, pero la clase `SGD` de `pytorch` ya incluye un valor por defecto.


```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

Un optimizador en `torch` tiene por defecto una serie de métodos. Sin embargo ahora mismo solo nos interesan 2 de ellos, pues son los que más usaremos.

* `Optimizer.step`
  > Este es el método es el que propiamente aplica el algoritmo SGD, o cualquier otro algoritmo que fueramos a implementar.
* `Optimizer.zero_grad`
  > Por defecto, `Optimizer` suma los sucesivos gradientes calculados. Esto hace que al principio de cada época de el entrenamiento, debamos setear el gradiente en 0. Es por esto que este método existe dentro de la clase `Optimizer`

## Entrenamiento

Hasta aquí veníamos reduciendo lineas de código de manera impresionante. Sin embargo, nuestro ciclo de entrenamiento será casi identico a lo que habíamos visto antes.
* Repetimos hasta concluir
    * Calculamos la función de pérdida
    * Calculamos el gradiente con minilotes 
    * Actualizamos los parámetros. 



```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

    epoch 1, loss 0.000278
    epoch 2, loss 0.000097
    epoch 3, loss 0.000096



```python
w = net[0].weight.data
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

    error in estimating w: tensor([-0.0002,  0.0006])
    error in estimating b: tensor([-0.0002])


Hasta aquí hemos trabajado con el problema de la regresión. Sin embargo, muchas veces lo que deseamos es clasificar segun clases discretas. De hecho, más adelante veremos que los grandes logros de las redes neuronales son en el area de clasificación. Para esto, a continuación hablaremos de Regresión Softmax y su aplicación en clasificación.
