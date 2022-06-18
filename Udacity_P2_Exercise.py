#!/usr/bin/env python
# coding: utf-8

# In[73]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


# Define transforms
## YOUR CODE HERE ##
train_transform = T.Compose([T.RandomRotation(45),
                             T.RandomVerticalFlip(p=0.3),
                             T.Resize(224),
                             T.ToTensor(),
                             T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])])

test_transform = T.Compose([T.Resize(224),
                            T.ToTensor(),
                            T.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])])

# Create training set and define training dataloader
## YOUR CODE HERE ##
import torchvision.datasets as datasets

trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Create test set and define test dataloader
## YOUR CODE HERE ##
testset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

# The 10 classes in the dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[75]:


def show5(img_loader):
    dataiter = iter(img_loader)
    
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(classes[labels[i]])
    
        image = images[i].numpy()
        std = [0.247, 0.243, 0.261] 
        mean = [0.491, 0.482, 0.447]
        for j in range(3):
            image[j] = std[j] * images[i][j] + mean[j]
        plt.imshow(np.rot90(image.T, k=3))
        plt.show()


# In[76]:


#show5(trainloader)


# In[77]:


# Explore data
## YOUR CODE HERE ##
images, labels = next(iter(trainloader))
print(images.shape)
print(labels.shape)


# In[78]:


## YOUR CODE HERE ##
from torchvision import models

model = models.densenet121(pretrained = True)


# In[79]:


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256,10),
                                nn.LogSoftmax(dim=1))


# In[80]:


## YOUR CODE HERE ##
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)


# In[81]:


## YOUR CODE HERE ##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device);


# In[ ]:


epochs = 5
steps = 0
train_loss = 0
print_every = 10

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model.forward(images)
                    test_loss += criterion(output, labels).item()
                    
                    ps = torch.exp(output)
                    top_p, top_class = ps. topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}") 
            train_loss = 0
            model.train()


# In[ ]:





# In[ ]:




