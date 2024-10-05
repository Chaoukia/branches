# Branches: A Fast Dynamic Programming and Branch & Bound Algorithm for Optimal Decision Trees

Source code for the Algorithm Branches described in [Branches: A Fast Dynamic Programming and Branch & Bound Algorithm for Optimal Decision Trees](https://arxiv.org/abs/2406.02175) .

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
    ├── trees                        # SVG files of optimal decision trees
    ├── Tables.png                   # PNG file containing a summary of empricial comparisons.
    ├── LICENSE
    └── README.md
File ```src/tutorial.ipynb``` contains a tutorial on how to use Branches with illustrative examples.

## Example of usage

The [MONK's Problems](https://archive.ics.uci.edu/dataset/70/monk+s+problems) are standard datasets for benchmarking Optimal Decision Trees algorithms. We use the first of these problems to illustrate how to use Branches.

```python
from branches import *
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# Reading the data
data = np.genfromtxt('data/monks-1.train', delimiter=' ', dtype=int)
data = data[:, :-1] # Getting rid of the last column, it contains only ids.
data = data[:, ::-1] # Reorder the columns to put the predicted variable Y at the end.

# Ordinal Encoding of the data
encoder = OrdinalEncoder()
encoder.fit(data)
data = encoder.transform(data).astype(int)

# Running Branches
alg = Branches(data)
alg.solve(lambd=0.01)

# Printing the accuracy, number of branches and number of splits
branches, splits = alg.lattice.infer()
print('Number of branches :', len(branches))
print('Number of splits :', splits)
print('Accuracy :', ((alg.predict(data[:, :-1]) == data[:, -1]).sum())/alg.n_total)
```

Using the nltk and svgling packages, we can plot the optimal Decision Tree via the code below. $\color{red}{\textsf{Please note that if you do not see the figures, it is probably due to a contrast issue and you should set a light theme for Github.}}$

```python
tree = alg.plot_tree()
svgling.draw_tree(tree)
```

<img src="trees/monk1-o.svg">

If we are only interested in the tree structure regardless of the predicted classes at the leaves, we can set ```show_classes=False``` in the plot_tree method.

```python
tree = alg.plot_tree(show_classes=False)
svgling.draw_tree(tree)
```

<img src="trees/monk1-o-no_classes.svg">

Here are some more examples of optimal Decision Trees we find for different problems.

<img src="trees/monk1-l.svg">
<img src="trees/car-eval-f.svg">

The tutorial file ```src/tutorial.ipynb``` contains more examples on how to use Branches, especially with its micro-optimisation techniques that allow for significant computational gains.

## Empirical Evaluation

Branches optimises the regularised accuracy $\mathcal{H}_{\lambda}\left( T\right) = \textrm{Accuracy}\left( T\right) - \lambda \mathcal{S}\left( T\right)$, where $\mathcal{S}\left( T\right)$ is the number of splits (internal nodes) of Decision Tree $T$ and $\lambda \in \left[ 0, 1 \right]$ is a penalty parameter. The tables below summarise the empirical comparison between Branches and the state of the art. For more information about the experimental setup, please refer to Section 6 and Appendix F in our paper [Branches: A Fast Dynamic Programming and Branch & Bound Algorithm for Optimal Decision Trees](https://arxiv.org/abs/2406.02175) .

<p align="center">
<img align="center" src="Tables.png", width=1000 height=auto/>
</p>




