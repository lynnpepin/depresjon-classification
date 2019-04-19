"""
Build a simple 1-layer classifier using Pytorch.

# https://github.com/andrewliao11/dni.pytorch/blob/master/mlp.py

# TODO: Train loader needs to have tuple (samples, labels), not just samples.
# TODO: And then work on the training part!

"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

TS_LENGTH = 2000
TRAINSPLIT = 652/752
VALSPLIT   = 100/652
RANDOMSTATE= 413

torch.manual_seed(RANDOMSTATE)
np.random.seed(RANDOMSTATE)

condition = np.load("condition_{}_emb.npy".format(TS_LENGTH))
control = np.load("control_{}_emb.npy".format(TS_LENGTH))

X = np.concatenate((condition, control), axis=0)
y = to_categorical(np.array([0]*len(condition) + [1]*len(control)))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 1-TRAINSPLIT,
                                                    random_state = RANDOMSTATE)

X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train,
                                                    test_size=VALSPLIT,
                                                    random_state = RANDOMSTATE)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=X_train, 
                                           batch_size=32, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=X_val, 
                                          batch_size=32, 
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 4, num_classes = 2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters())

for epoch in range(50):
    for i, sample in 
    for i, samples in enumerate(train_loader):  
        # Convert torch tensor to Variable
        # TODO
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(samples)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
