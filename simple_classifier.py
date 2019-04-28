"""
Build a simple 1-layer classifier using Keras.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Input, GRU
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from time import time

TS_LENGTH = 2000
TRAINSPLIT = 652 / 752
VALSPLIT = 100 / 652
RANDOMSTATE = np.random.randint(1, 2**16)

root = os.path.curdir

condition = np.load(
    os.path.join(root, "condition_{}_emb.npy".format(TS_LENGTH)))
control = np.load(os.path.join(root, "control_{}_emb.npy".format(TS_LENGTH)))

X = np.concatenate((condition, control), axis=0)
y = to_categorical(np.array([0] * len(condition) + [1] * len(control)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - TRAINSPLIT, random_state=RANDOMSTATE)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=VALSPLIT, random_state=RANDOMSTATE)

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
#model.add(Dense(2, activation = 'softmax'))
model.add(Dense(2, input_dim=TS_LENGTH, activation='softmax'))

model.compile(
    loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=50,
    callbacks=[EarlyStopping(patience=10)],
    validation_split=0.1)

# More in depth analysis could be done here...
yh = model.predict(X_val)
ypred = np.argmax(yh, axis=1)
ytrue = y_val[:, 1]
print("Accuracy:", np.sum(ypred == ytrue) / 100)

