<a href="https://colab.research.google.com/github/institutohumai/cursos-python/blob/master/DeepLearning/3_Redes_Multicapa/2_backpropagation_autograd.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# Backpropagation y Autograd


```python
import torch
```

Autograd durante el entrenamiento
--------------------

Hemos echado un breve vistazo a cómo funciona Autograd, pero ¿cómo se ve cuando se usa para el propósito previsto? Definamos un modelo pequeño y examinemos cómo cambia después de un solo lote de entrenamiento. Primero, definimos algunas constantes, nuestro modelo y algunos sustitutos para entradas y salidas:


```python
BATCH_SIZE = 16
DIM_IN = 784
HIDDEN_SIZE = 256
DIM_OUT = 10

net = torch.nn.Sequential(torch.nn.Linear(DIM_IN, HIDDEN_SIZE),
                    torch.nn.ReLU(),
                    torch.nn.Linear(HIDDEN_SIZE, DIM_OUT))

# features aleatorias
some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
# etiquetas aleatorias
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = net
```

Fijemos nos que no hizo falta agregar
``requires_grad=True`` a las capas de modelo esto es por que la clase ``torch.nn.Module`` supone que siempre usaremos el gradiente para entrenar el modelo

Sin embargo, al momento de inicial los valores del modelo, el gradiente no se calcula, hasta que lo pidamos.




```python
print(model[2].weight[0][0:10]) # solo algunos son mostrados
print(model[2].weight.grad)
```

    tensor([ 0.0389, -0.0137,  0.0299, -0.0603, -0.0114, -0.0389, -0.0509,  0.0555,
            -0.0331, -0.0127], grad_fn=<SliceBackward0>)
    None


Veamos que ocurre ahora si entrenamos. 

Consideremos como función de perdida la distancia cuadrática media entre nuestra ``prediction`` y las etiquetas, ``ideal_output``

En este caso usaremos SGD como algoritmos de optimización.





```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum()
print(loss)
```

    tensor(186.4553, grad_fn=<SumBackward0>)


Hasta que no llamemos  ``loss.backward()`` los gradientes no se calculan.





```python
print(model[2].weight[0][0:10]) # solo algunos son mostrados
print(model[2].weight.grad)
```

    tensor([ 0.0389, -0.0137,  0.0299, -0.0603, -0.0114, -0.0389, -0.0509,  0.0555,
            -0.0331, -0.0127], grad_fn=<SliceBackward0>)
    None



```python
loss.backward()
print(model[2].weight[0][0:10])
print(model[2].weight.grad[0][0:10])
```

    tensor([ 0.0389, -0.0137,  0.0299, -0.0603, -0.0114, -0.0389, -0.0509,  0.0555,
            -0.0331, -0.0127], grad_fn=<SliceBackward0>)
    tensor([-1.2933,  2.1021,  1.0937, -3.7679, -2.0949, -1.8788, -5.4912,  0.5143,
            -2.8695,  1.9576])


Por ahora solo hemos calculados los gradientes, pero no los hemos usada para actualizar los pesos. Esto es porque debemos ejecutar ``optimizer.step()``





```python
print(model[2].weight[0][0:10]) # solo algunos son mostrados
print(model[2].weight.grad[0][0:10])
```

    tensor([ 0.0389, -0.0137,  0.0299, -0.0603, -0.0114, -0.0389, -0.0509,  0.0555,
            -0.0331, -0.0127], grad_fn=<SliceBackward0>)
    tensor([-1.2933,  2.1021,  1.0937, -3.7679, -2.0949, -1.8788, -5.4912,  0.5143,
            -2.8695,  1.9576])



```python
optimizer.step()
print(model[2].weight[0][0:10])
print(model[2].weight.grad[0][0:10])
```

    tensor([ 0.0402, -0.0158,  0.0288, -0.0566, -0.0093, -0.0370, -0.0454,  0.0550,
            -0.0303, -0.0147], grad_fn=<SliceBackward0>)
    tensor([-1.2933,  2.1021,  1.0937, -3.7679, -2.0949, -1.8788, -5.4912,  0.5143,
            -2.8695,  1.9576])


Vemos ahora que los valores de ``model[2]`` han cambiado

Un detalle que no dedemos ignorar es que debemos llamar a la función ``optimizer.zero_grad()`` despues de llamar
``optimizer.step()``. De no hacer esto cada vez que llamemos  ``loss.backward()`` la suma de los gradientes se acumulará.





```python
print(model[2].weight.grad[0][0:10])

for i in range(0, 5):
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()
    
print(model[2].weight.grad[0][0:10])

optimizer.zero_grad()

print(model[2].weight.grad[0][0:10])
```

    tensor([-1.2933,  2.1021,  1.0937, -3.7679, -2.0949, -1.8788, -5.4912,  0.5143,
            -2.8695,  1.9576])
    tensor([-12.3276,  19.2645,   7.7500, -11.1583,  -7.6493,  -1.7553, -12.9434,
              2.4429, -10.6501,  11.7848])
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


Contenido adicional: Más información sobre Autograd
-----------------------------------------------------------

En principio, ya conocíamos la noción de gradiente. Sabíamos que para una toma vectores m-dimensionales y devuelve un único valor (un escalar), $l=g\left(\vec{y}\right)$ existe el gradiente. Esto es un vector que nos dice como varía una función conforme cambian los valores del vector de entrada $\vec{y}$ 


$$v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}$$

En general, si tenemos una función que toma vectores n-dimensionales como entrada y tiene como salida vectores m-dimensionales, $\vec{y}=f(\vec{x})$, la idea de gradiente no permite abarcar todas las posibles variaciones. En este sentido se necesita una generalización de la idea de gradiente. Esta generalización es una matriz conocida como el 
*Jacobiano:*

\begin{align}J
     =
     \left(\begin{array}{ccc}
     \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
     \vdots & \ddots & \vdots\\
     \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
     \end{array}\right)\end{align}

Sin embargo, la función de pérdida nuestros modelos más sencillos son en realidad una combinación de las dos cosas. 

$$l=g\left(\vec{y}\right)$$
$$\vec{y}=f(\vec{x})$$
$$l=g\left(f(\vec{x})\right)$$

Puede demostrarse, sin embargo, que para obtener el gradiente de $l$, respecto de $\vec{x}$ solo debemos hacer una multiplicación matricial

$$\vec{\nabla_x} l=J^{T}\cdot v$$

\begin{align}J^{T}\cdot v=\left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)\left(\begin{array}{c}
   \frac{\partial l}{\partial y_{1}}\\
   \vdots\\
   \frac{\partial l}{\partial y_{m}}
   \end{array}\right)=\left(\begin{array}{c}
   \frac{\partial l}{\partial x_{1}}\\
   \vdots\\
   \frac{\partial l}{\partial x_{n}}
   \end{array}\right)\end{align}

Del mismo modo, a la salida de cada capa, tenemos un Jacobiano distinto. De tal manera que nuestro gradiente en realdiad tendra la forma:

$$\vec{\nabla_x} l=J_{1}^{T} J_{2}^{T} J_{3}^{T} J_{4}^{T}\cdot v$$


**``torch.autograd`` es la herramienta que computa todas estas dependencias por medio de productos matriciales** Además de guardar la relación entre cada salida y cada entrada de cada capa




Para más información consultar

<https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api>
