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
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

TS_LENGTH = 2000
TRAINSPLIT = 652/752
VALSPLIT   = 100/652
RANDOMSTATE= 413

torch.manual_seed(RANDOMSTATE)
np.random.seed(RANDOMSTATE)

class DepresjonDataset(Dataset):
    def __init__(self, emb=True, ts_length = TS_LENGTH):
        # Read emb / not emb
        # TODO: Run preprocess, embedding if not done already
        # TODO: Train, test, shuffle state?
        if emb:
            self.condition = np.load("condition_{}_emb.npy".format(ts_length))
            self.control = np.load("control_{}_emb.npy".format(ts_length))
        else:
            self.condition = np.load("condition_{}.npy".format(ts_length))
            self.control = np.load("control_{}.npy".format(ts_length))
    
        self.X = np.concatenate((self.condition, self.control), axis=0)
        self.y = to_categorical(np.array([0]*len(self.condition) + [1]*len(self.control)))
    
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
    def __len__(self):
        assert(len(self.X) == len(self.y))
        return len(self.X)

loader = torch.utils.data.DataLoader(dataset=DepresjonDataset(),
                                     batch_size=32,
                                     shuffle=True)


# TODO - Implement pytorch "dataset" class to use here for loading.
# See: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
# See: https://old.reddit.com/r/MachineLearning/comments/bcfyo2/d_pytorch_implementation_best_practices/


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
    for i, (x, y) in enumerate(loader):
        #x_input = Variable(x.view(-1, 2))
        #y_output = Variable(y)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
