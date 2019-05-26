
## Custom Residual Block
### By Ali Shannon

This simple project shows how you can make a simple residual block by passing parameters to two different branches and then concatenating them into one module and running another layer on them.

This is done with pytorch low level API so it might not work with high level API like Skorch or Keras. This is inspired by the work done on ResNet in the past years.


```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
```


```python
X, y = make_classification(n_samples=10000, n_features=100, n_informative=90, n_classes=10)

X = torch.from_numpy(X).cuda()
y = torch.from_numpy(y).cuda()

X_train, y_train = X[:-100], y[:-100]
X_test, y_test = X[-100:], y[-100:]
```

Notice that in the network below the inputs are passed to both **x1** and **x2** for different operations then combined as **x** later to run in the last fully connected layer to softmax.


```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.bn = nn.BatchNorm1d(num_features = 100, momentum = 0.999)
        
        self.branch1 = nn.Sequential(
            nn.Linear(100, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(100, 128),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        
        self.relu = nn.ReLU()
                
        self.fc = nn.Linear(256, 10)
        
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x):
        
        x = self.bn(x)
        
        # first branch
        x1 = self.branch1(x)
        
        # second branch
        x2 = self.branch2(x)
        
        #x = torch.add(x1, x2)
        x = torch.cat((x1, x2), 1)
        
        x = self.relu(x)
        
        x = self.fc(x)
        
        x = self.softmax(x)
        
        return x
        
model = Net().double().cuda() # cuda for for devices with Nvidia GPUs
```


```python
print(model)
```

    Net(
      (bn): BatchNorm1d(100, eps=1e-05, momentum=0.999, affine=True, track_running_stats=True)
      (branch1): Sequential(
        (0): Linear(in_features=100, out_features=512, bias=True)
        (1): Dropout(p=0.2)
        (2): ReLU()
        (3): Linear(in_features=512, out_features=256, bias=True)
        (4): Dropout(p=0.2)
        (5): ReLU()
        (6): Linear(in_features=256, out_features=128, bias=True)
        (7): Dropout(p=0.2)
        (8): ReLU()
      )
      (branch2): Sequential(
        (0): Linear(in_features=100, out_features=128, bias=True)
        (1): Dropout(p=0.2)
        (2): ReLU()
      )
      (relu): ReLU()
      (fc): Linear(in_features=256, out_features=10, bias=True)
      (softmax): Softmax()
    )



```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
```


```python
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

        
for epoch in range(1, 101):
    
    running_loss = 0
    
    # train the network
    
    model.train()
    
    for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size = 1000):
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    running_loss *= len(X_batch)/len(X_train)
    
    # validate with testing data
    
    model.eval()

    y_pred = model(X_test)
    loss = criterion(y_pred, y_test)
        
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} \tTraining Loss: {running_loss:.4f}\tValidation Loss: {loss.item():.4f}')
        
```

    Epoch: 10 	Training Loss: 1.5868	Validation Loss: 1.6570
    Epoch: 20 	Training Loss: 1.5098	Validation Loss: 1.5889
    Epoch: 30 	Training Loss: 1.4951	Validation Loss: 1.5574
    Epoch: 40 	Training Loss: 1.4901	Validation Loss: 1.4972
    Epoch: 50 	Training Loss: 1.4856	Validation Loss: 1.5563
    Epoch: 60 	Training Loss: 1.4821	Validation Loss: 1.5339
    Epoch: 70 	Training Loss: 1.4842	Validation Loss: 1.5560
    Epoch: 80 	Training Loss: 1.4853	Validation Loss: 1.5401
    Epoch: 90 	Training Loss: 1.4831	Validation Loss: 1.5438
    Epoch: 100 	Training Loss: 1.4849	Validation Loss: 1.5475



```python
model.eval()

y_pred = model(X_test).detach().cpu().numpy()

from sklearn.metrics import accuracy_score

acc = accuracy_score(np.argmax(y_pred, axis = 1), y_test.cpu().numpy())

print(f'Accuracy of the residual block net is {acc*100}%')
```

    Accuracy of the residual block net is 91.0%



```python

```
