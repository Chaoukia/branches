import svgling
import argparse
from branch_ordinal import *
from branch_binary import *
from branch_binary_multi import *
from time import time

class Branches:
    """
    Description
    -----------------------
    Class describing the Search algorithm.
    """
    
    def __init__(self, data, encoding='ordinal', max_depth=np.inf):
        """
        Description
        -----------------------
        Constructor of class Search.

        Attributes & Parameters
        -----------------------
        data     : 2D np.array, data matrix, the last column is the class vector.
        encoding : String in {'ordinal', 'binary', 'multi'}.
                    - 'ordinal': For multiclass classification of ordinally encoded data.
                    - 'binary' : For binary classification of binary encoded data.
                    - 'multi'  : For multiclass classification of binary encoded data.
        """
        
        self.data = data
        self.n_total = data.shape[0]
        self.encoding = encoding
        self.max_depth = max_depth
        if encoding == 'ordinal':
            categories = [len(set(data[:, j])) for j in range(data.shape[1]-1)]
            attributes_categories = dict(zip(np.arange(len(categories)), categories))
            self.lattice = Lattice(attributes_categories, self.n_total, max_depth)
            
        elif encoding == 'binary':
            attributes = set(range(data.shape[1]-1))
            self.lattice = LatticeBinary(attributes, self.n_total, max_depth)
            
        elif encoding == 'multi':
            K = len(set(data[:, -1]))
            attributes = set(range(data.shape[1]-1))
            self.lattice = LatticeMulti(attributes, self.n_total, K, max_depth)
            
        else:
            raise ValueError("encoding must be 'ordinal' or 'binary' or 'multi'.")

        self.dict_branches = self.lattice.dict_branches

    def solve_ordinal(self, lambd, n=1000, print_iter=100, time_limit=600):
        """
        Description
        -----------------------
        Search for the optimal Decision Tree.
        Parameters
        -----------------------
        lambd      : Float in ]0, 1[, the complexity parameter.
        n          : Int, maximum number of iterations.
        print_iter : Int, number of iterations between two prints.
        time_limit : Int, Time limit in number of seconds.
        """
        
        start_time = time()
        self.lattice.root.evaluate(self.data[:, -1], lambd, self.n_total)
        i=1
        while (i <= n) and (not self.lattice.root.complete):
            branch, sorted_branch, path = self.lattice.select()
            if branch.complete:
                self.lattice.backpropagate(path, lambd)
            else:
                self.lattice.expand(branch, self.data, lambd, sorted_branch)
                self.lattice.backpropagate(path, lambd)
                
            if i%print_iter == 0:
                print('Iteration %d' %i)
                if time() - start_time > time_limit:
                    print('Time Out.')
                    return i
                
            i += 1
            
        if i < n:
            print('The search finished after %d iterations.' %i)
            return i
            
        else:
            print('The search is still incomplete.')
            return i
        
    def solve_binary(self, lambd, n=1000, print_iter=100, time_limit=600):
        """
        Description
        -----------------------
        Search for the optimal Decision Tree.

        Parameters
        -----------------------
        lambd      : Float in [0, 1], the complexity parameter.
        n          : Int, maximum number of iterations.
        print_iter : Int, number of iterations between two prints.
        time_limit : Int, Time limit in number of seconds.
        """
        
        start_time = time()
        self.lattice.initialise(self.data, lambd)
        i=1
        while (i <= n) and (not self.lattice.root.complete):
            branch, sorted_branch, path = self.lattice.select()
            branch_parent = path[-2]
            _, value_complete, attribute, children, key = branch_parent.queue[0]   # The tuple in the queue that includes branch.
            branch_sibling = branch_parent.children[attribute][1 - key]
            if branch.complete:
                # If the sibling is not complete and has not been expanded yet.
                if (not branch_sibling.complete) and (not branch_sibling.children):
                    sorted_branch_sibling = sorted_branch[:-1] + [str((attribute, 1 - key))]
                    self.lattice.expand(branch_sibling, path[-2], self.data, lambd, sorted_branch_sibling)
                    
                self.lattice.backpropagate(path, lambd)

            else:
                self.lattice.expand(branch, path[-2], self.data, lambd, sorted_branch)
                # If the sibling is not complete and has not been expanded yet.
                if (not branch_sibling.complete) and (not branch_sibling.children):
                    sorted_branch_sibling = sorted_branch[:-1] + [str((attribute, 1 - key))]
                    self.lattice.expand(branch_sibling, path[-2], self.data, lambd, sorted_branch_sibling)
                    
                self.lattice.backpropagate(path, lambd)
                
            if i%print_iter == 0:
                print('Iteration %d' %i)
                if time() - start_time > time_limit:
                    print('Time Out.')
                    return i
                
            i += 1
            
        if i < n:
            print('The search finished after %d iterations.' %i)
            return i
            
        else:
            print('The search is still incomplete.')
            return i

    def solve(self, lambd, n=1000, print_iter=100, time_limit=600):
        """
        Description
        -----------------------
        Search for the optimal Decision Tree.

        Parameters
        -----------------------
        lambd      : Float in [0, 1], the complexity parameter.
        n          : Int, maximum number of iterations.
        print_iter : Int, number of iterations between two prints.
        time_limit : Int, Time limit in number of seconds.
        """

        if self.encoding == 'ordinal':
            return self.solve_ordinal(lambd, n, print_iter, time_limit)

        elif (self.encoding == 'binary') or (self.encoding == 'multi'):
            return self.solve_binary(lambd, n, print_iter, time_limit)
            
    def reinitialise(self):
        """
        Description
        --------------
        Clear the memo.
        
        Parameters
        --------------
        
        Returns
        --------------
        """
        
        self.dict_branches.clear()
            
    def infer(self):
        """
        Description
        --------------
        Retrieve the terminal branches of the optimal DT.
        
        Parameters
        --------------
        
        Returns
        --------------
        """
        
        return self.lattice.infer()

    def predict(self, X):
        """
        Description
        --------------
        Predict the classes of the examples in the data matrix X.
        
        Parameters
        --------------
        X : 2D np.array, its rows are the examples we want to classify.
        
        Returns
        --------------
        1D np.array, the predicted classes of the rows in data matrix X.
        """

        preds = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = self.lattice.predict(X[i, :])

        return preds
    
    def plot_tree(self, show_classes=True, compact=False):
        """
        Description
        --------------
        Plot the optimal Decision Tree.
        
        Parameters
        --------------
        show_classes : Boolean, whether to show the predicted classes or not.
        compact      : Boolean, whether to collapse the DT or not.
        
        Returns
        --------------
        nltk tree object, visualize the optimal Decision Tree.
        """

        if self.encoding == 'ordinal':
            return self.lattice.plot_tree(show_classes, compact)

        else:
            return self.lattice.plot_tree(show_classes)

    def save_tree(self, show_classes=True, file_name='tree.svg'):
        """
        Description
        --------------
        Plot the decision tree.
        
        Parameters
        --------------
        show_classes : Boolean, whether to show the predicted classes or not.
        file_name    : String, name of the file where to store the tree.
        
        Returns
        --------------
        """

        tree = self.lattice.plot_tree(show_classes)
        svgling.draw_tree(tree).saveas("../trees/"+file_name, pretty=True) 

def retrieve_depth(branches):
    d = 0
    for branch in branches:
        if branch.depth > d:
            d = branch.depth

    return d

def summary(data, lambd=0.01, n=5000000, print_iter=10000, time_limit=300, encoding='ordinal', max_depth=np.inf, show_classes=True, compact=True):
    alg = Branches(data, encoding, max_depth)
    start_time = time()
    iterations = alg.solve(lambd, n=n, print_iter=print_iter, time_limit=time_limit)
    print('Execution time        : %.4f' %(time() - start_time))
    print('Number of iterations  : %d' %iterations)
    branches, splits = alg.lattice.infer()
    print('Number of branches    :', len(branches))
    print('Number of splits      :', splits)
    d = retrieve_depth(branches)
    print('Depth                 :', d)
    acc = ((alg.predict(data[:, :-1]) == data[:, -1]).sum())/alg.n_total
    print('Accuracy :', acc)
    print('Regularised objective :', acc - lambd*splits)
    tree = alg.plot_tree(show_classes, compact)
    alg.reinitialise()
    return tree

