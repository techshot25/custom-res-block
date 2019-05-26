#!/usr/bin/env python
# coding: utf-8

# ## Custom Residual Block
# ### By Ali Shannon
# 
# This simple project shows how you can make a simple residual block by passing parameters to two different branches and then concatenating them into one module and running another layer on them.
# 
# This is done with pytorch low level API so it might not work with high level API like Skorch or Keras. This is inspired by the work done on ResNet in the past years.

# In[1]:


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import torch
from torch import nn, optim


# In[2]:


X, y = make_classification(n_samples=10000, n_features=100, n_informative=90, n_classes=10)

X = torch.from_numpy(X).cuda()
y = torch.from_numpy(y).cuda()

X_train, y_train = X[:-100], y[:-100]
X_test, y_test = X[-100:], y[-100:]


# Notice that in the network below the inputs are passed to both **x1** and **x2** for different operations then combined as **x** later to run in the last fully connected layer to softmax.

# In[3]:


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
            nn.Dropout(0.2)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(100, 128),
            nn.Dropout(0.2)
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


# In[4]:


print(model)


# In[5]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)


# In[6]:


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
        


# In[7]:


model.eval()

y_pred = model(X_test).detach().cpu().numpy()

from sklearn.metrics import accuracy_score

acc = accuracy_score(np.argmax(y_pred, axis = 1), y_test.cpu().numpy())

print(f'Accuracy of the residual block net is {acc*100}%')

