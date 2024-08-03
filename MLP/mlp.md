# MLP & MFCC Phoneme Recognition

As mentioned in the readme file, the MLP models usually look like this. 
<img src="MLP/mlp.jpg" width="500" />

And for each node in the picture, it is a linear calculation, which can be represented as "Wx + b", where w is the weight of this node and b is the bias. 

With a given number N, we can create N cells in each layer and connect them with the previous/next layer with weights and bias.

<img src="MLP/cell.jpg" width="400" />
However, linear equation like "Wx + b" can only represent linear relationship. To mimic non-linear relationships in real world, we introduced activation functions.


<img src="MLP/activation.jpg" width="500" />

Activation functions are an indispensable part of neural networks. By introducing non-linearity, they enable the network to learn and represent complex non-linear relationships, 
making it capable of handling complex tasks such as image recognition and natural language processing.

## Model and hyperparameters
For a fully-functioning MLP with loss function and SGD in python, it contains these attributes:
- l: list of model layers
- L: number of model layers
- lr: learning rate
- mu: momentum rate µ, tunable hyperparameter controlling how much the previous updates affect
the direction of current update. µ = 0 means no momentum
- W: list of weight velocity for each layer
- b: list of bias velocity for each layer

```python
import numpy as np

class SGD:

    def __init__(self, model, lr=0.1, momentum=0):
        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        for i in range(self.L):
            if self.mu == 0:
                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW
                self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb
            else:
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb
                self.l[i].W = self.l[i].W - self.lr * self.v_W[i]
                self.l[i].b = self.l[i].b - self.lr * self.v_b[i]
```

## Phoneme Recognition task
After knowing how a MLP works, know we can use this model to predict/classify data. 


