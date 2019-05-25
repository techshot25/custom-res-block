
## Custom Residual Block
### By Ali Shannon

This simple project shows how you can make a simple residual block by passing parameters to two different branches and then concatenating them into one module and running another layer on them.

This is done with pytorch low level API so it might to work in Skorch or Keras


```python
import numpy as np
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
```


```python
X, y = make_classification(n_samples=1000, n_features=100, n_informative=90, n_classes=3)
```

Notice that in the network below the inputs are passed to both **x1** and **x2** for different operations then combined as **x** later to run in the last fully connected layer to softmax.


```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.Dropout(0.2)
        )
        
        
        self.fc = nn.Linear(100, 3)
        
        self.relu = nn.ReLU()
        
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x):
        
        
        # first branch
        x1 = self.branch1(x)
        
        # second branch
        x2 = self.branch2(x)
        
        x = torch.cat((x1, x2), 1)
        
        x = self.relu(x)
        
        x = self.fc(x)
        
        x = self.softmax(x)
        
        return x
        
model = Net().cuda().double() # I use CUDA because I have a gpu
```


```python
print(model)
```

    Net(
      (branch1): Sequential(
        (0): Linear(in_features=100, out_features=75, bias=True)
        (1): ReLU()
        (2): Linear(in_features=75, out_features=50, bias=True)
      )
      (branch2): Sequential(
        (0): Linear(in_features=100, out_features=50, bias=True)
        (1): Dropout(p=0.2)
      )
      (fc): Linear(in_features=100, out_features=3, bias=True)
      (relu): ReLU()
      (softmax): Softmax()
    )



```python
x = torch.tensor(X).cuda()
y = torch.tensor(y).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.02)
```


```python
for epoch in range(1, 1001):
    
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'epoch {epoch} \tLoss: {loss.item():.6f}')
        
```

    epoch 100 	Loss: 0.960346
    epoch 200 	Loss: 0.859943
    epoch 300 	Loss: 0.805659
    epoch 400 	Loss: 0.767474
    epoch 500 	Loss: 0.722628
    epoch 600 	Loss: 0.702597
    epoch 700 	Loss: 0.682851
    epoch 800 	Loss: 0.654043
    epoch 900 	Loss: 0.643702
    epoch 1000 	Loss: 0.632673



```python
model.eval()

y_pred = model(x).detach().cpu().numpy()

from sklearn.metrics import accuracy_score

acc = accuracy_score(np.argmax(y_pred, axis = 1), y.cpu().numpy())

print(f'Accuracy of the residual block net is {acc*100}%')
```

    Accuracy of the residual block net is 96.2%

