#simple pytorch classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pathlib import Path

TS_LENGTH = 2000
TRAINSPLIT = 652/752
RANDOMSTATE= 413

torch.manual_seed(RANDOMSTATE)
np.random.seed(RANDOMSTATE)

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

#gets full path of data files (assuming they are in same folder as this python file)
#load data from npy files
condition = np.load(Path("./condition_{}_emb.npy".format(TS_LENGTH)))
control = np.load(Path("./control_{}_emb.npy".format(TS_LENGTH)))

X = np.concatenate((condition, control), axis=0)
y = np.array([0]*len(condition) + [1]*len(control))
#train/test split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 1-TRAINSPLIT, random_state = RANDOMSTATE)
#convert into a Variable
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

net = Net()

criterion = nn.CrossEntropyLoss()# cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)#stochastic gradient descent

for epoch in range(1000):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print( 'number of epoch', epoch, 'loss', loss.data.item())

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)

print( 'prediction accuracy', accuracy_score(test_y.data, predict_y.data))

print( 'macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
print( 'micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
print( 'macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
print( 'micro recall', recall_score(test_y.data, predict_y.data, average='micro'))
