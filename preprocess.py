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
#   Let's cut data into 3000-sized samples, discard extraneous data.
#   Then, normalize via 0-1 normalization
#       Consider: max of values across ALL participants, or max per participant?
#                 Different people might have different levels of activity, etc.

# Metadata preproc:
#   0-1 norm days recorded, one-hot encode gender, one-hot encode age

# Models to compare with:
#   (Time series x (Norm all vs Norm per), Meta Data) x (Linear x 1-hidden),
#   and Time series x (Neural ODE, GRU/RNN/LSTM)


