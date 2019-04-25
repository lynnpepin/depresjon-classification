"""
A small visualization of the depresjon dataset, using UMAP.

Cyan is depressed, magenta is not.
"""

import matplotlib.pyplot as plt
from sklearn import manifold
import umap

import numpy as np
TS_LENGTH = 2000
condition = np.load("condition_{}.npy".format(TS_LENGTH))
control = np.load("control_{}.npy".format(TS_LENGTH))
SHOWFIG = False
# Size of each point
SIZE    = .4
ALPHA   = .9

"""
emb_cond = umap.UMAP().fit_transform(condition)
emb_cont = umap.UMAP().fit_transform(control)

np.save("condition_{}_emb".format(TS_LENGTH), emb_cond)
np.save("control_{}_emb".format(TS_LENGTH), emb_cont)
"""

emb_cond = np.load("condition_{}_emb.npy".format(TS_LENGTH))
emb_cont = np.load("control_{}_emb.npy".format(TS_LENGTH))


Y = np.array([0]*len(emb_cond) + [1]*len(emb_cont))

embedded = np.concatenate((emb_cond, emb_cont), axis=0)

plt.figure(figsize=(6,5))
# 0 == cyan == depressed
# 1 == magenta == not depressed.
for target, color in zip([0,1], ['c', 'm']):
    plt.scatter(embedded[Y == target, 0], embedded[Y == target, 1],
                c=color, s=SIZE, alpha=ALPHA)

plt.show()
plt.close()
