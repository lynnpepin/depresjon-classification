# Preprocessing:
# Output classification is "depressed" or "not depressed"
# Input:
#   Per sample, there is a one-variable time series,
#   plus gender and age 'meta' data.
#   Use a non-sequential NN:
#       Time series input --> GRU --------> Rest of NN --> Out
#            Gender/age 'meta' data  --/

# Time series preproc:
#   Data measurements are difficult-to-work-with variable-lengths,
#   ranging from about 19000 samples to about 65000 samples.
#   Let's cut data into 2000-sized samples, discard extraneous data.
#   Then, normalize via 0-1 normalization
#       Consider: max of values across ALL participants, or max per participant?
#                 Different people might have different levels of activity, etc.

# Metadata preproc:
#   0-1 norm days recorded, one-hot encode gender, one-hot encode age

# Models to compare with:
#   (Time series x (Norm all vs Norm per), Meta Data) x (Linear x 1-hidden),
#   and Time series x (Neural ODE, GRU/RNN/LSTM)

# todo: Write actual documentation


import numpy as np
import pandas as pd

# We cut each time-series into parts, length 4000
#   Result: More samples, smaller and uniform size in each sample
TS_LENGTH = 2000

condition_path = "./depresjon-dataset/condition/condition_{}.csv"
control_path = "./depresjon-dataset/control/control_{}.csv"

condition_raw = [np.array(pd.read_csv(condition_path.format(x))['activity']) for x in range(1,24)]
control_raw = [np.array(pd.read_csv(control_path.format(x))['activity']) for x in range(1,33)]

# truncate returns a series, truncated such that axis 0 has an integer multiple of TS_LENGTH
# e.g. a.shape=(51611,32); truncate(a).shape=(50000,32)
truncate = lambda series : series[:TS_LENGTH * (len(series) // TS_LENGTH)]

# condition, control are both np_arrays shape (number of samples, TS_LENGTH)
condition = np.concatenate([truncate(series).reshape(-1, TS_LENGTH) for series in condition_raw])
control = np.concatenate([truncate(series).reshape(-1, TS_LENGTH) for series in control_raw])

# TODO - Write normalization functions!

np.save("condition_{}".format(TS_LENGTH), condition)
np.save("control_{}".format(TS_LENGTH), control)

