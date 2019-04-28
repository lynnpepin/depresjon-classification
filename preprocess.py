"""preprocess.py

Small script to preprocess the depresjon dataset. See README.

Input files are one-variable time series with values distributed roughly
geometrically.

Because each time series can be a different length, rather than pad,
we truncate and splice the first 2000*k values.
    (E.g. A sequence of lenght 8621 becomes 4 sequences of length 2000,
          and the last 621 values are discarded.)

Values are in the range 0-8000, and are normalized as x := lg(1+x)/lg(8001),
then saved to condition_2000.npy, control_2000.npy.


We should consider a mixture of files with this data:

1. Use time series data,
    a. Normalize [0,1] across all, or 
    b. Normalize per column? (I.e. Per each respective person's max)
2. And/or use meta data or age, gender?
    If time series is fed in to LSTM, the output of LSTM can be concat. with metadata.
3. What kind of model to use?
    a. One layer perceptron, 1-layer MLP, more layers?
    b. GRU, RNN, LSTM, etc.?
    c. And compare to Neural ODE too.
"""

# Truncate and cut to file
import numpy as np
import pandas as pd
import os

# We cut each time-series into parts, length 4000
#   Result: More samples, smaller and uniform size in each sample
TS_LENGTH = 2000

root = os.path.curdir
print("Opening raw CSVs")
condition_path = os.path.join(root, "./depresjon-dataset/condition/condition_{}.csv")
control_path = os.path.join(root, "./depresjon-dataset/control/control_{}.csv")

condition_raw = [
    np.array(pd.read_csv(condition_path.format(x))['activity'])
    for x in range(1, 24)
]
control_raw = [
    np.array(pd.read_csv(control_path.format(x))['activity'])
    for x in range(1, 33)
]

# truncate returns a series, truncated such that axis 0 has an integer multiple of TS_LENGTH
# e.g. a.shape=(51611,32); truncate(a).shape=(50000,32)
print("Processing raw CSVs")
truncate = lambda series: series[:TS_LENGTH * (len(series) // TS_LENGTH)]

# condition, control are both np_arrays shape (number of samples, TS_LENGTH)
condition = np.concatenate(
    [truncate(series).reshape(-1, TS_LENGTH) for series in condition_raw])
control = np.concatenate(
    [truncate(series).reshape(-1, TS_LENGTH) for series in control_raw])

# data is roughly geometrically distributed; let's fix this and then scale down
control = np.log(control + 1)
condition = np.log(condition + 1)
scale = max(control.max(), condition.max())
control = control / scale
condition = condition / scale

print("Saving processed depresjon data to")
print("condition_{}".format(TS_LENGTH),"and","control_{}".format(TS_LENGTH))
np.save("condition_{}".format(TS_LENGTH), condition)
np.save("control_{}".format(TS_LENGTH), control)


# UMAP embeddings
from sklearn import manifold
import umap
print("Loading processed depresjon data.")
condition = np.load("condition_{}.npy".format(TS_LENGTH))
control = np.load("control_{}.npy".format(TS_LENGTH))
print("UMAP embedding condition...")
emb_cond = umap.UMAP().fit_transform(condition)
print("UMAP embedding control...")
emb_cont = umap.UMAP().fit_transform(control)

print("Saving UMAP embeddings to *_emb.npy")
np.save("condition_{}_emb".format(TS_LENGTH), emb_cond)
np.save("control_{}_emb".format(TS_LENGTH), emb_cont)
