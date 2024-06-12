# Branches: A Fast Dynamic Programming and Branch & Bound Algorithm for Optimal Decision Trees

Source code for the Algorithm Branches described in the paper [Branches: A Fast Dynamic Programming and Branch & Bound Algorithm for Optimal Decision Trees](https://arxiv.org/abs/2406.02175) that is currecntly under review.

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

Branches solves for sparsity, which means that it not only optimises for accuracy but also for the complexity of the Decision Tree. A good metric to evaluate these Decision Tree solution is through the regularised objective $\mathcal{H}_{\lambda}\left( T\right) = \textrm{Accuracy}\left( T\right) - \lambda \mathcal{S}\left( T\right)$, where $\mathcal{S}\left( T\right)$ is the number of splits (internal nodes) of $T$ and $\lambda \in \left[ 0, 1 \right]$ is a penalty parameter. Branches achieves and proves convergence in record times on many datasets when compared with the state of the art. The table below summarises the empirical comparison of the different algorithms. $\mathcal{T}$ is the execution time in seconds (TO indicates time out after 5 minutes), and $\mathcal{I}$ the number of iterations. Branches clearly outperforms the Python implementations OSDT and PyGOSDT in terms of optimal convergence and speed. Branches also outperforms the C++ implementation GOSDT in many cases, and even when it is slower, Branches always converges in significantly fewer iterations. Branches is a practical and very promising algorithm, moreover, a future C++ implementation of Branches will likely lead to a significant improvement of Branches' computational performance, just as we notice when comparing PyGOSDT and GOSDT.

<table>
  <tr>
    <td> </td>
    <td colspan="3">OSDT</td>
    <td colspan="3">PyGOSDT</td>
    <td colspan="3">GOSDT</td>
    <td colspan="3">Branches</td>
  </tr>
  <tr>
    <td>Dataset</td>
    <td>$\mathcal{H}_\lambda$</td>
    <td>$\mathcal{T}$</td>
    <td>$\mathcal{I}$</td>
    <td>$\mathcal{H}_\lambda$</td>
    <td>$\mathcal{T}$</td>
    <td>$\mathcal{I}$</td>
    <td>$\mathcal{H}_\lambda$</td>
    <td>$\mathcal{T}$</td>
    <td>$\mathcal{I}$</td>
    <td>$\mathcal{H}_\lambda$</td>
    <td>$\mathcal{T}$</td>
    <td>$\mathcal{I}$</td>
  </tr>
  <tr>
    <td>monk1-l</td>
    <td>0.93</td>
    <td>71</td>
    <td>2e6</td>
    <td>0.93</td>
    <td>181</td>
    <td>3e6</td>
    <td>0.93</td>
    <td>0.71</td>
    <td>3e4</td>
    <td>0.93</td>
    <td>0.11</td>
    <td>617</td>
  </tr>
  <tr>
    <td>monk1-f</td>
    <td>0.97</td>
    <td>TO</td>
    <td>2e4</td>
    <td>0.97</td>
    <td>TO</td>
    <td>2e3</td>
    <td>0.983</td>
    <td>4.02</td>
    <td>9e4</td>
    <td>0.983</td>
    <td>1.31</td>
    <td>1e4</td>
  </tr>
  <tr>
    <td>monk1-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.9</td>
    <td>0.02</td>
    <td>64</td>
  </tr>
  <tr>
    <td>monk2-l</td>
    <td>0.95</td>
    <td>TO</td>
    <td>7e4</td>
    <td>0.95</td>
    <td>TO</td>
    <td>400</td>
    <td>0.97</td>
    <td>10</td>
    <td>1e5</td>
    <td>0.97</td>
    <td>2.8</td>
    <td>3e4</td>
  </tr>
  <tr>
    <td>monk2-f</td>
    <td>0.90</td>
    <td>TO</td>
    <td>4e4</td>
    <td>0.90</td>
    <td>TO</td>
    <td>3e4</td>
    <td>0.93</td>
    <td>11.1</td>
    <td>1e5</td>
    <td>0.93</td>
    <td>5.9</td>
    <td>7e4</td>
  </tr>
  <tr>
    <td>monk2-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.95</td>
    <td>0.14</td>
    <td>1e3</td>
  </tr>
  <tr>
    <td>monk3-l</td>
    <td>0.979</td>
    <td>TO</td>
    <td>593</td>
    <td>0.979</td>
    <td>TO</td>
    <td>123</td>
    <td>0.981</td>
    <td>7.38</td>
    <td>8e4</td>
    <td>0.981</td>
    <td>1.20</td>
    <td>9e3</td>
  </tr>
  <tr>
    <td>monk3-f</td>
    <td>0.975</td>
    <td>TO</td>
    <td>1e4</td>
    <td>0.973</td>
    <td>TO</td>
    <td>9e3</td>
    <td>0.983</td>
    <td>2.13</td>
    <td>5e4</td>
    <td>0.983</td>
    <td>1.14</td>
    <td>9e3</td>
  </tr>
  <tr>
    <td>monk3-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.987</td>
    <td>0.04</td>
    <td>156</td>
  </tr>
  <tr>
    <td>tic-tac-toe</td>
    <td>0.765</td>
    <td>TO</td>
    <td>40</td>
    <td>0.808</td>
    <td>TO</td>
    <td>37</td>
    <td>0.850</td>
    <td>41</td>
    <td>1.6e6</td>
    <td>0.850</td>
    <td>68</td>
    <td>2.6e5</td>
  </tr>
  <tr>
    <td>tic-tac-toe-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.832</td>
    <td>0.95</td>
    <td>3479</td>
  </tr>
  <tr>
    <td>car-eval</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.799</td>
    <td>18</td>
    <td>9e5</td>
    <td>0.799</td>
    <td>62</td>
    <td>3e5</td>
  </tr>
  <tr>
    <td>car-eval-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.812</td>
    <td>0.11</td>
    <td>632</td>
  </tr>
  <tr>
    <td>nursery</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.755</td>
    <td>TO</td>
    <td>9e5</td>
    <td>0.820</td>
    <td>TO</td>
    <td>3e5</td>
  </tr>
  <tr>
    <td>nursery-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.822</td>
    <td>0.34</td>
    <td>244</td>
  </tr>
  <tr>
    <td>mushroom</td>
    <td>0.945</td>
    <td>TO</td>
    <td>4e6</td>
    <td>0.945</td>
    <td>TO</td>
    <td>2e6</td>
    <td>0.925</td>
    <td>TO</td>
    <td>1e6</td>
    <td>0.938</td>
    <td>TO</td>
    <td>2e4</td>
  </tr>
  <tr>
    <td>mushroom-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.975</td>
    <td>0.17</td>
    <td>6</td>
  </tr>
  <tr>
    <td>kr-vs-kp</td>
    <td>0.900</td>
    <td>TO</td>
    <td>6e4</td>
    <td>0.900</td>
    <td>TO</td>
    <td>2e4</td>
    <td>0.815</td>
    <td>TO</td>
    <td>4e5</td>
    <td>0.900</td>
    <td>TO</td>
    <td>8e4</td>
  </tr>
  <tr>
    <td>kr-vskp-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.900</td>
    <td>TO</td>
    <td>8e4</td>
  </tr>
  <tr>
    <td>zoo</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.992</td>
    <td>34</td>
    <td>3e5</td>
    <td>0.992</td>
    <td>15</td>
    <td>3e4</td>
  </tr>
  <tr>
    <td>zoo-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.993</td>
    <td>0.94</td>
    <td>1456</td>
  </tr>
  <tr>
    <td>lymph</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.784</td>
    <td>TO</td>
    <td>1e6</td>
    <td>0.808</td>
    <td>TO</td>
    <td>1e5</td>
  </tr>
  <tr>
    <td>lymph-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.852</td>
    <td>18</td>
    <td>2e4</td>
  </tr>
  <tr>
    <td>balance</td>
    <td>0.693</td>
    <td>TO</td>
    <td>1e5</td>
    <td>0.693</td>
    <td>TO</td>
    <td>3e4</td>
    <td>0.693</td>
    <td>21</td>
    <td>1e6</td>
    <td>0.693</td>
    <td>62</td>
    <td>3e5</td>
  </tr>
  <tr>
    <td>balance-o</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>___</td>
    <td>0.661</td>
    <td>0.02</td>
    <td>130</td>
  </tr>
</table>




