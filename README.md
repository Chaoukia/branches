# branches

Source code for the Algorithm Branches.

## Dependencies

We recommend creating a conda virtual environment from our .yml file as follows:
```
conda env create -f dependencies.yml
conda activate branches
```
To visualize Decision Trees, we need the svgling package, which is not currenlty supported by conda. Thus we install it with pip:
```
pip install svgling
```

## Repository Structure
    .
    ├── data                         # Data used for benchmarking
    ├── src                          # Source files
    │   ├── branch_ordinal.py        # Source file for classification problems with ordinally encoded data
    │   ├── branch_binary.py         # Source file for binary classification problems with binary data
    │   ├── branch_binary_multi.py   # Source file for classification problems with binary data
    │   ├── branches.py              # Source file for the Branches algorithm
    │   └── tutorial.ipynb           # Tutorial .ipynb notebook
    ├── LICENSE
    └── README.md
