"""
Build a simple 1-layer classifier using Pytorch.
# https://github.com/andrewliao11/dni.pytorch/blob/master/mlp.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from pathlib import Path

TS_LENGTH = 2000
TRAINSPLIT = 652/752
VALSPLIT   = 100/652
RANDOMSTATE= 413

NUM_EPOCHS = 100
BATCH_SIZE = 32

# Set the random seed
torch.manual_seed(RANDOMSTATE)
np.random.seed(RANDOMSTATE)

# Load the condition and control, reshape into input X and target y, then split into train, test sets.
condition = np.load(Path("./condition_{}_emb.npy".format(TS_LENGTH)))
control = np.load(Path("./control_{}_emb.npy".format(TS_LENGTH)))
X = np.concatenate((condition, control), axis=0)
y = np.array([0]*len(condition) + [1]*len(control))
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 1-TRAINSPLIT, random_state = RANDOMSTATE)

# The Dataset class required by PyTorch
class GenericDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
    def __len__(self):
        assert(len(self.X) == len(self.y))
        return len(self.X)

train_dataset = GenericDataset(train_X, train_y)
test_dataset = GenericDataset(test_X, test_y)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)


# Define our NN:
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

# Time to Train
for epoch in range(NUM_EPOCHS):
    for i, sample in enumerate(train_loader):
        x, y = sample
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(Variable(x))
        loss = criterion(outputs, Variable(y))
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
               %(epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')

predict_out = torch.Tensor(test_X)
_, predict_y = torch.max(predict_out, 1)
print( 'prediction accuracy', accuracy_score(test_y.data, predict_y.data))
print( 'macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
print( 'micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
print( 'macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
print( 'micro recall', recall_score(test_y.data, predict_y.data, average='micro'))

