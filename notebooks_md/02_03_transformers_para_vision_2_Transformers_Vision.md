<a href="https://colab.research.google.com/github/institutohumai/cursos-python/blob/master/CV/3_CNN_Modernas/2_Transformers_Vision.ipynb"> <img src='https://colab.research.google.com/assets/colab-badge.svg' /> </a>


# Transformers para Visión

La arquitectura del tranformer se propuso inicialmente para el aprendizaje de secuencia a secuencia, como la traducción automática. Con gran eficacia, los tranformers se convirtieron posteriormente en el modelo de elección en diversas tareas de procesamiento del lenguaje natural.
Sin embargo, en el campo de la visión artificial la arquitectura dominante
se había basado en las CNN. *¿Podemos adaptar tranformers para modelar datos de imagen*? Esta pregunta ha despertado un gran interés en la comunidad de visión artificial. Un [paper del 2020](https://arxiv.org/pdf/1911.03584.pdf) demostró teóricamente que la autoatención puede aprender a comportarse de manera similar a la convolución. Empíricamente, se tomaron parches de $2 \times 2$ de las imágenes como entrada, pero el pequeño tamaño del parche hace que el modelo solo sea aplicable a datos de imágenes con resoluciones bajas.

Sin restricciones específicas sobre el tamaño del parche, los *tranformers de visión* (ViT) extraen parches de las imágenes y los introducen en un encoder de tranformer para obtener una representación global, que finalmente se transformará para la clasificación. En particular, los tranformers muestran una mejor escalabilidad que las CNN: cuando se entrenan modelos más grandes en conjuntos de datos más grandes, los tranformers de visión superan a los ResNet por un margen significativo. Similar al panorama del diseño de arquitectura de red en el procesamiento del lenguaje natural, los tranformers también cambiaron las reglas del juego en la visión por computadora.




## Modelo



La siguiente figura representa la arquitectura modelo de los tranformers de visión. Esta arquitectura consta de una base que parchea las imágenes, un cuerpo basado en el encoder de un tranformer multicapa y una cabeza que transforma la representación global en la etiqueta de salida.

![Imgur](https://i.imgur.com/4qOAWmXl.png)




Considere una imagen de entrada con altura $h$, ancho $w$ y $c$ canales. Especificando la altura y el ancho del parche como $p$,
la imagen se divide en una secuencia de $m = hw/p^2$ parches,
donde cada parche se aplana a un vector de longitud $cp^2$.
De esta forma, los parches de imagen pueden ser tratados de manera similar a los tokens en secuencias de texto por encoderes de tranformeres. Un token especial “&lt;cls&gt;” (clasificación) y los $m$ parches de imagen aplanados se proyectan linealmente en una secuencia de vectores $m+1$, sumados con embeddings posicionales que se pueden aprender. El encoder de tranformer multicapa transforma los vectores de entrada $m+1$ en la misma cantidad de representaciones de vectores de salida de la misma longitud. Funciona exactamente de la misma manera que el encoder del tranformer original, solo que difiere en la posición de normalización. Dado que el token “&lt;cls&gt;”  atiende a todos los parches de imagen a través de la autoatención su representación desde la salida del encoder del tranformer
se transformará en la etiqueta de salida.


```python
import torch
from torch import nn
```

## Patch Embedding

Para implementar un transformer de visión, comencemos con los embeddings de los parches. Dividir una imagen en parches y proyectar linealmente estos parches aplanados se puede simplificar como una sola operación de convolución, donde tanto el tamaño del kernel como el tamaño del stride se establecen en el tamaño del parche.



```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        img_size, patch_size = (img_size,img_size), (patch_size,patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

En el siguiente ejemplo, tomando imágenes con una altura y un ancho de `img_size` como entrada, se generan `(img_size//patch_size)**2` parches que se proyectan linealmente en vectores de longitud `num_hiddens`.



```python
def check_shape(a, shape):
    """Defined in :numref:`sec_rnn-scratch`"""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = torch.randn(batch_size, 3, img_size, img_size)
check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Encoder del Transformer de Vision

El MLP del encoder transformer de visión es ligeramente diferente del a la red feed forward posicional del encoder del transformer original. Primero, aquí la función de activación usa la unidad lineal de error gaussiano (GELU), que puede considerarse como una versión más suave de ReLU. En segundo lugar, el dropout se aplica a la salida de cada capa densa en el MLP para la regularización.


```python
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

La implementación del bloque del encoder del transformer de visión simplemente sigue el diseño de prenormalización, donde la normalización se aplica justo *antes* de la atención multiples cabezales o el MLP. A diferencia de la posnormalización, donde la normalización se coloca justo *después* de las conexiones residuales, la prenormalización conduce a un entrenamiento más efectivo o eficiente para los transformers.



```python
# @markdown class MultiHeadAttentionLayer
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
```


```python
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, device):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape).to(device)
        self.attention = MultiHeadAttentionLayer(num_hiddens, num_heads,
                                                dropout, device).to(device)
        self.ln2 = nn.LayerNorm(norm_shape).to(device)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout).to(device)

    def forward(self, X, mask=None):
        X = self.ln1(X)
        att_output , _ = self.attention(X, X, X, mask)
        return X + self.mlp(self.ln2(X + att_output))
```

Igual que en el transformer original, cualquier bloque de encoder de transformer de visión no cambia su forma de entrada.



```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = torch.ones((2, 100, 24)).to(device)
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5, device)
encoder_blk.eval()
check_shape(encoder_blk(X), X.shape)
```

## Juntar todo

El paso hacia adelante de los transformers de visión es sencillo. Primero, las imágenes de entrada se introducen en una instancia `PatchEmbedding`, cuya salida se concatena con el embedding del token “&lt;cls&gt;”. Se suman los embeddings posicionales aprendibles antes del dropout. Luego, la salida se alimenta al encoder del transformer que apila las instancias `num_blks` de la clase `ViTBlock`. Finalmente, la representación del token “&lt;cls&gt;” token es proyectado por la cabeza de la red.



```python
class ViT(nn.Module):
    """Vision transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout,device, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout,device))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

## Training

Training a vision transformer on the Fashion-MNIST dataset is just like how CNNs were trained in :numref:`chap_modern_cnn`.



```python
import torchvision
from torchvision import transforms
from torch.utils import data

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
    
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

```


```python
def train_FashionMNIST_classifier(model, lr, num_epochs, resize=None):
  batch_size= 128
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  loss = nn.CrossEntropyLoss(reduction='none')
  trainer = torch.optim.Adam(model.parameters())
  train_iter, test_iter = load_data_fashion_mnist(batch_size,resize=resize)

  for epoch in range(num_epochs):
      L = 0.0
      N = 0
      Acc = 0.0
      TestAcc = 0.0
      TestN = 0
      for X, y in train_iter:
          X, y = X.to(device), y.to(device)
          l = loss(model(X),y)
          trainer.zero_grad()
          l.mean().backward()
          trainer.step()
          L += l.sum()
          N += l.numel()
          Acc += accuracy(model(X), y)
          #print(L)
      for X, y in test_iter:
          X, y = X.to(device), y.to(device)
          TestN += y.numel()
          TestAcc += accuracy(model(X), y)
      print(f'epoch {epoch + 1}, loss {(L/N):f}\
            , train accuracy  {(Acc/N):f}, test accuracy {(TestAcc/TestN):f}')
```


```python
lr, num_epochs = 0.1, 10
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout = 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, device, lr )

train_FashionMNIST_classifier(model,lr,num_epochs,resize=(img_size, img_size))

```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../data/FashionMNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/26421880 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/raw/train-images-idx3-ubyte.gz to ../data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/29515 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/4422102 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/5148 [00:00<?, ?it/s]


    Extracting ../data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw
    
    epoch 1, loss 0.826274            , train accuracy  0.701333, test accuracy 0.755600
    epoch 2, loss 0.586331            , train accuracy  0.791700, test accuracy 0.776000
    epoch 3, loss 0.545117            , train accuracy  0.805550, test accuracy 0.804200
    epoch 4, loss 0.610295            , train accuracy  0.778200, test accuracy 0.715400
    epoch 5, loss 0.660155            , train accuracy  0.756983, test accuracy 0.745100
    epoch 6, loss 0.674837            , train accuracy  0.755283, test accuracy 0.743400
    epoch 7, loss 0.717554            , train accuracy  0.732617, test accuracy 0.748000
    epoch 8, loss 0.677545            , train accuracy  0.749767, test accuracy 0.755100
    epoch 9, loss 0.603885            , train accuracy  0.777017, test accuracy 0.774900
    epoch 10, loss 0.571257            , train accuracy  0.789967, test accuracy 0.781700


Puede notar que para datasets pequeños como Fashion-MNIST, nuestro transformer de visión implementado no supera a ResNet. Se pueden realizar observaciones similares incluso en el conjunto de datos de ImageNet (1,2 millones de imágenes). Esto se debe a que los transformers carecen de esos principios útiles en la convolución, como la localidad y la invariancia a la traslación. Sin embargo, el panorama cambia cuando se entrenan modelos más grandes en datasets más grandes (por ejemplo, 300 millones de imágenes), donde los transformers de visión superan a las ResNets por un amplio margen en la clasificación de imágenes, lo que demuestra la superioridad intrínseca de los transformers en escalabilidad. 


