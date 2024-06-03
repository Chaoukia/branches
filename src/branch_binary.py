import numpy as np
import heapq
from collections import Counter
from nltk import Tree
from queue import Queue
from time import time

dict_branches = {}
"""
dict_branches : Dict, memo the Dynamic Programming.
                - key   : String, unique representation of a branch.
                - value : BranchBinary, stored branch.
"""

class BranchBinary:
    """
    Description
    -----------------------
    Class describing a branch.
    
    Class Attributes:
    -----------------------
    """
    
    def __init__(self, id_branch, attributes, bit_vector):
        """
        Description
        -----------------------
        Constructor of class BranchBinary.

        Attributes & Parameters
        -----------------------
        id_branch        : String representing the branch.
        attributes       : Set of the remaining attributes for potential splits.
                                      - values : Number of categories of each attribute.
        bit_vector       : 1D np.array, indicator vector of the data that the branch contains.
        children         : Dict: - keys   : Splitting attribute.
                                 - values : Dict: - keys   : Value of the splitting attribute.
                                                  - values : The corresponding branch.
        queue            : List in heap queue form. Its elements are tuples of the form (-value, value_complete, attribute, children):
                                - value          : Float, sum of values of the children nodes.
                                                   The reason we store it negative in the tuple is because heapq in python is not a max heapq.
                                - value_complete : Float, sum of the values of the complete children.
                                                   Initialised to 0, once a child is complete, it is popped from the children dictionary and its value is incremented to value_complete.
                                - attribute      : Int, splitting attribute.
                                - children       : Dictionary of the same form as the class attribute children. 
                                                   The difference is that this one only keeps incomplete children (children for which search is still incomplete).
                                                   As soon as a child is complete, it is popped out of the dictionary.
        attribute_opt    : Int, the optimal attribute.
        terminal         : Boolean, whether the branch is terminal or not. A branch is terminal if the terminal action at the branch is optimal.
        complete         : Boolean, whether the search down the branch is complete or not. The search is complete if one of these is true:
                                - The branch is terminal.
                                - The optimal split has been found.
        value            : Float in [0, 1], upper bound on the optimal value of the branch.
        value_terminal   : Float in [0, 1], value of taking the terminal action at the branch.
        value_greedy     : Float in [0, 1], the best calculated value so far.
        freq             : Float in [0, 1], the proportion of samples in the branch.
        pred             : Int, the predicted class.
        n_samples        : Int, the number of observed examples in the branch. None if the branch has not been visited yet.
        n_ones           : Int, the number of observed examples in the branch of class 1. None if the branch has not been visited yet.
        """
        
        self.id_branch = id_branch
        self.attributes = attributes
        self.bit_vector = bit_vector
        self.attribute_opt = None
        if attributes:
            self.terminal = False
            self.complete = False

        else:
            self.terminal = True
            self.complete = True

        self.value = None
        self.value_terminal = None
        self.value_greedy = None
        self.freq = None
        self.pred = 0
        self.n_samples = None
        self.n_ones = None
        self.children = {}
        self.queue = []
        dict_branches[self.id_branch] = self    # Add the branch to the memo.
        
    def set_value(self, value):
        """
        Description
        -----------------------
        Set the value of the branch.
        
        Parameters
        -----------------------
        value : Float in [0, 1], estimated value of the branch.
        
        Returns
        -----------------------
        """
        
        self.value = value
        
    def evaluate_terminal(self, y, n_total, n_samples=None, n_ones=None):
        """
        Description
        -----------------------
        Evaluate the terminal action.

        Parameters
        -----------------------
        y         : 1D np.array, predictor column at the branch.
        n_total   : Int, the total number of samples in the training set.
        n_samples : Int, the number of observed examples in the branch. None if the branch has not been visited yet.
        n_ones    : Int, the number of observed examples in the branch of class 1. None if the branch has not been visited yet.

        Returns
        -----------------------
        """

        if n_samples is not None:
            self.n_samples = n_samples
            self.n_ones = n_ones
            if n_ones > n_samples/2:
                self.pred, pred_n = 1, n_ones

            else:
                self.pred, pred_n = 0, n_samples - n_ones

            self.freq = n_samples/n_total
            value = pred_n/n_total
            self.value_terminal = value
            self.value_greedy = value

            return

        n_samples = y.shape[0]
        self.n_samples = n_samples
        if n_samples == 0:          # If there are no samples in the branch, cut it.
            self.complete = True
            self.value = 0
            self.value_terminal = 0
            self.value_greedy = 0
            self.freq = 0
            self.n_ones = 0

        else:
            s = np.count_nonzero(y)
            self.n_ones = s
            if s > n_samples/2:
                self.pred, pred_n = 1, s

            else:
                self.pred, pred_n = 0, n_samples - s

            self.freq = n_samples/n_total
            value = pred_n/n_total
            self.value_terminal = value
            self.value_greedy = value

    def evaluate(self, y, lambd, n_total, n_samples=None, n_ones=None):
        """
        Description
        -----------------------
        Estimate the upper bound of the value of the branch.

        Parameters
        -----------------------
        y         : 1D np.array, predictor column at the branch.
        lambd     : Float in [0, 1], the complexity parameter.
        n_total   : Int, the total number of samples in the training set.
        n_samples : Int, the number of observed examples in the branch. None if the branch has not been visited yet.
        n_ones    : Int, the number of observed examples in the branch of class 1. None if the branch has not been visited yet.

        Returns
        -----------------------
        """

        # Only evaluate the branch if it has not been evaluated before.
        if self.value is None:
            self.evaluate_terminal(y, n_total, n_samples, n_ones)   # When the branch is explored for the first time, evaluate it.
            if not self.terminal:   # If the branch can still be split because there are still available attributes.
                value_split = -lambd + self.freq
                self.set_value(max(value_split, self.value_terminal))
                if self.value == self.value_terminal:
                    self.complete, self.terminal = True, True   # The terminal action has been proven to be the best at this branch.

            else:    # If the branch can no longer be split because there are no more available attributes.
                self.set_value(self.value_terminal)
            
    def split(self, X_branch, y_branch, lambd, sorted_branch, attribute, n_total):
        """
        Description
        -----------------------
        Split a branch.

        Parameters
        -----------------------
        X_branch      : 1D np.array, attribute vector at the branch we want to split.
        y_branch      : 1D np.array, class vector at the branch we want to split.
        lambd         : Float in [0, 1], the complexity parameter.
        sorted_branch : List of tuples (attribute, value) representing the branch. Sorted by the attribute.
        attribute     : Int, the attribute to use for the split. Note that the attribute has to be among the keys of dictionary attributes_categories.
        n_total       : Int, the total number of samples in the training set.

        Returns
        -----------------------
        """

        self.children[attribute] = {}
        children_queue = {}      # The dictionary of incomplete children that we store in the queue.
        attributes_children = self.attributes.copy()
        attributes_children.remove(attribute)
        value_children_complete = 0
        key = None                    # The key, in children_queue, of the most promising branch to explore.
        value_greedy_min = np.inf     # Initialising the minimum value_greedy among the children branches to consider.
        bit_vector = self.bit_vector

        attribute_value_child = str((attribute, 0))
        sorted_branch_child = sorted(sorted_branch + [attribute_value_child])
        id_branch_child = ''.join(sorted_branch_child)

        id_branch_child_1 = id_branch_child.replace(attribute_value_child, str((attribute, 1)))

        try:
            branch_child = dict_branches[id_branch_child]
            bit_vector_child = branch_child.bit_vector

            try:
                branch_child_1 = dict_branches[id_branch_child_1]

            except KeyError:
                bit_vector_child_1 = np.logical_and(bit_vector, ~bit_vector_child)
                branch_child_1 = BranchBinary(id_branch_child_1, attributes_children, bit_vector_child_1)
                branch_child_1.evaluate(None, lambd, n_total, self.n_samples - branch_child.n_samples, self.n_ones - branch_child.n_ones)

        except KeyError:
            try:
                branch_child_1 = dict_branches[id_branch_child_1]
                bit_vector_child_1 = branch_child_1.bit_vector

            except KeyError:
                indices = (X_branch == 1)
                bit_vector_child_1 = bit_vector.copy()     # Create the child's bit vecotr as a copy of its parent's bit vector.
                bit_vector_child_1[bit_vector] = indices   # Update the ones in the child's bit vector with the indices corresponding to the data that the child contains.
                y_branch_child_1 = y_branch[indices]    # Data corresponding to the current child node.
                branch_child_1 = BranchBinary(id_branch_child_1, attributes_children, bit_vector_child_1)
                branch_child_1.evaluate(y_branch_child_1, lambd, n_total)

            bit_vector_child = np.logical_and(bit_vector, ~bit_vector_child_1)
            branch_child = BranchBinary(id_branch_child, attributes_children, bit_vector_child)
            branch_child.evaluate(None, lambd, n_total, self.n_samples - branch_child_1.n_samples, self.n_ones - branch_child_1.n_ones)

        self.children[attribute][0] = branch_child
        self.children[attribute][1] = branch_child_1

        value_children = branch_child.value + branch_child_1.value
        value_children_greedy = branch_child.value_greedy + branch_child_1.value_greedy
        if branch_child.complete:
            value_children_complete += branch_child.value

        else:
            children_queue[0] = branch_child
            key = 0
            value_greedy_min = branch_child.value_greedy

        if branch_child_1.complete:
            value_children_complete += branch_child_1.value

        else:
            children_queue[1] = branch_child_1
            if branch_child_1.value_greedy < value_greedy_min:
                key = 1

        heapq.heappush(self.queue, (-value_children, value_children_complete, attribute, children_queue, key))
        value_children_greedy = -lambd + value_children_greedy
        if value_children_greedy > self.value_greedy:
            self.attribute_opt = attribute
            self.value_greedy = value_children_greedy
        
    
class LatticeBinary:
    """
    Description
    -----------------------
    Class describing a lattice for binary classification problems with binary predictors.
    
    Class Attributes:
    -----------------------    
    """
    
    def __init__(self, attributes, n_total):
        """
        Description
        -----------------------
        Constructor of class Lattice.

        Attributes & Parameters
        -----------------------
        attributes : Set of the attributes indices.
        n_total    : Int, the total number of data examples.
        """
        
        self.n_total = n_total
        self.root = BranchBinary('', attributes, np.full(n_total, True))
        self.dict_branches = dict_branches
        
    def select(self):
        """
        Description
        -----------------------
        Select the most promising branch.

        Parameters
        -----------------------
        
        Returns
        -----------------------
        branch        : Branch, the selected branch.
        sorted_branch : List of tuples (attribute, value) representing the branch. Sorted by the attribute.
        path          : List of the branches on the selection path.
        """
        
        branch = self.root
        sorted_branch, path = [], [branch]
        while (branch.children) and (not branch.complete):
            attribute, children, key = branch.queue[0][2], branch.queue[0][3], branch.queue[0][4]
            branch = children[key]
            sorted_branch.append(str((attribute, key)))
            path.append(branch)
            
        return branch, sorted_branch, path
    
    def expand(self, branch, data, lambd, sorted_branch):
        """
        Description
        -----------------------
        Expand a branch.
        
        Parameters
        -----------------------
        branch        : Branch, the branch to expand.
        data          : 2D np.array, data matrix, the last column is the class vector.
        lambd         : Float in [0, 1], the complexity parameter.
        sorted_branch : List of tuples (attribute, value) representing the branch. Sorted by the attribute.
        
        Returns
        -----------------------
        """
        
        if branch.children:
            raise ValueError('The branch has already been expanded.')
            
        n_total = data.shape[0]
        data_branch = data[branch.bit_vector, :]
        y_branch = data_branch[:, -1]
        
        for attribute in branch.attributes:
            X_branch  = data_branch[:, attribute]
            branch.split(X_branch, y_branch, lambd, sorted_branch, attribute, n_total)
            
        value_max = -branch.queue[0][0] - lambd
        branch.set_value(value_max)
        if not branch.queue[0][-2]:
            branch.complete = True   # The branch is complete if its best set of children is complete (children_queue is empty).
            
        if branch.value == branch.value_terminal:
            branch.complete, branch.terminal = True, True
                        
    def backpropagate(self, path, lambd):
        """
        Description
        -----------------------
        Backpropagate the value of a branch.
        
        Parameters
        -----------------------
        path  : List of the branches on the selection path.
        lambd : Float in [0, 1], the complexity parameter.
        
        Returns
        -----------------------
        """
        
        length = len(path)
        branch = path[-1]
        index_parent = -2
        while index_parent >= -length:
            branch_parent = path[index_parent]
            _, value_complete, attribute, children, key = heapq.heappop(branch_parent.queue)   # The tuple in the queue that includes branch.
            if branch.complete:
                value_complete += branch.value
                children.pop(key)

            value = value_complete
            value_greedy = value_complete
            value_greedy_min = np.inf
            for category, branch_child in children.items():
                value += branch_child.value
                value_greedy += branch_child.value_greedy
                if branch_child.value_greedy < value_greedy_min:
                    value_greedy_min = branch_child.value_greedy
                    key = category
                
            value_greedy = -lambd + value_greedy
            if value_greedy > branch_parent.value_greedy:
                branch_parent.attribute_opt = attribute
                branch_parent.value_greedy = value_greedy
            
            tuple_new = (-value, value_complete, attribute, children, key)
            heapq.heappush(branch_parent.queue, tuple_new)
            value_parent = - lambd - branch_parent.queue[0][0]
            branch_parent.set_value(max(value_parent, branch_parent.value_terminal))
            if branch_parent.value == branch_parent.value_terminal:
                branch_parent.complete, branch_parent.terminal = True, True
                
            if not branch_parent.queue[0][-2]:
                branch_parent.complete = True   # The parent branch is complete if its best set of children is composed of complete branches.
                
            branch = branch_parent
            index_parent -= 1
            
    def infer(self):
        """
        Description
        --------------
        Retrieve the terminal branches of the optimal DT.
        
        Parameters
        --------------
        
        Returns
        --------------
        branches : List, the branches constituting the optimal DT.
        splits   : Int, the number of splits in the optimal DT.
        """
        
        branches = []
        splits = 0
        queue = Queue()
        queue.put(self.root)
        while not queue.empty():
            branch = queue.get()
            if (branch.terminal) or (not branch.children) or (branch.attribute_opt is None):
                branches.append(branch)
                
            else:
                splits += 1
                children_opt = branch.children[branch.attribute_opt]
                for child in children_opt.values():
                    queue.put(child)
                    
        return branches, splits

    def predict(self, X):
        """
        Description
        --------------
        Predict the class of example X.
        
        Parameters
        --------------
        X : 1D np.array, the example we want to classify.
        
        Returns
        --------------
        Int, the predicted class of X.
        """

        branch = self.root
        while (not branch.terminal) and (branch.children) and (branch.attribute_opt is not None):
            branch = branch.children[branch.attribute_opt][X[branch.attribute_opt]]

        return branch.pred
            
    def build_string_node(self, branch, show_classes):
        """
        Description
        --------------
        Build string representation (with parentheses) of the optimal subtree rooted at branch.
        
        Parameters
        --------------
        branch : Branch, the branch from which we want to build the optimal subtree.
        
        Returns
        --------------
        String representation of the optimal subtree rooted at branch.
        """
        
        string = ''
        if (branch.terminal) or (not branch.children) or (branch.attribute_opt is None):
            if show_classes:
                return 'Y=' + str(branch.pred) + ' '

            else:
                return ''
        
        children_opt = branch.children[branch.attribute_opt]
        for category in range(2):
            child = branch.children[branch.attribute_opt][category]
            string += '(X_' + str(branch.attribute_opt) + '=' + str(category) + ' ' + self.build_string_node(child, show_classes) + ') '

        return string
    
    def build_string(self, show_classes=True):
        """
        Description
        --------------
        Build string representation of the optimal Decision Tree.
        
        Parameters
        --------------
        
        Returns
        --------------
        String representation of the optimal Decision Tree.
        """
        
        return '( ' + self.build_string_node(self.root, show_classes) + ')'
    
    def plot_tree(self, show_classes=True):
        """
        Description
        --------------
        Plot the decision tree.
        
        Parameters
        --------------
        
        Returns
        --------------
        nltk tree object, visualize the optimal Decision Tree.
        """

        return Tree.fromstring(self.build_string(show_classes))

