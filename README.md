Dataset: https://datasets.simula.no/depresjon/

Download and extract the data into ./depresjon-dataset

The structure should be as such:

    depresjon-dataset/
    ├── condition/
    │   ├── condition_1.csv
    │   │   ...
    │   └── condition_23.csv
    ├── control/
    │   ├── control_1.csv
    │   │   ...
    │   └── control_32.csv
    └── scores.csv 

## Instructions:

1. **Install** torchdiffeq from <https://github.com/rtqichen/torchdiffeq>.

2. **Preprocess** by running preprocess.py to generate conditon\_2000.npy, control\_2000.npy, conditon\_2000\_emb.npy, control\_2000\_emb.npy.

3. **Note:** Each run of preprocess.py will generate a different UMAP embedding! Run **visualize.py** to see what it looks like.

The other Python scripts are example scripts that train a classifier.
