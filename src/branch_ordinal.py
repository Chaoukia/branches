import numpy as np
import heapq
import warnings
from collections import Counter
from nltk import Tree
from queue import Queue
from time import time

dict_branches = {}
"""
dict_branches : Dict, memo the Dynamic Programming.
                - key   : String, unique representation of a branch.
                - value : Branch, stored branch.
"""

class Branch:
    """
    Description
    -----------------------
    Class describing a branch.
    
    Class Attributes:
    -----------------------    
    """
    
    def __init__(self, id_branch, attributes_categories, bit_vector, depth, max_depth):
        """
        Description
        -----------------------
        Constructor of class Branch.

        Attributes & Parameters
        -----------------------
        id_branch             : String representing the branch.
        attributes_categories : Dict: - keys   : Indices of the attributes to be considered for potential splits within the node.
                                      - values : Number of categories of each attribute.
        bit_vector            : 1D np.array, indicator vector of the data that the branch contains.
        depth                 : Int, the depth of the branch.
        children              : Dict: - keys   : Splitting attribute.
                                      - values : Dict: - keys   : Value of the splitting attribute.
                                                       - values : The corresponding branch.
        queue                 : List in heap queue form. Its elements are tuples of the form (-value, value_complete, attribute, children):
                                - value          : Float, sum of values of the children nodes.
                                                   The reason we store it negative in the tuple is because heapq in python is in min heapq form.
                                - value_complete : Float, sum of the values of the complete children.
                                                   Initialised to 0, once a child is complete, it is popped from the children dictionary and its value is incremented to value_complete.
                                - attribute      : Int, splitting attribute.
                                - children       : Dictionary of the same form as the class attribute children. 
                                                   The difference is that this one only keeps incomplete children (children for which search is still incomplete).
                                                   As soon as a child is complete, it is popped out of the dictionary.
        attribute_opt         : Int, the optimal attribute.
        terminal              : Boolean, whether the branch is terminal or not. A branch is terminal if the terminal action at the branch is optimal.
        complete              : Boolean, whether the search down the branch is complete or not. The search is complete if one of these is true:
                                - The branch is terminal.
                                - The optimal split has been found.
        value                 : Float in [0, 1], upper bound on the optimal value of the branch.
        value_terminal        : Float in [0, 1], value of taking the terminal action at the branch.
        value_greedy          : Float in [0, 1], the best calculated value so far.
        freq                  : Float in [0, 1], the proportion of samples in the branch.
        pred                  : Int, the predicted class.
        """
        
        self.id_branch = id_branch
        self.attributes_categories = attributes_categories
        self.bit_vector = bit_vector
        self.depth = depth
        self.children = {}
        self.queue = []
        self.attribute_opt = None
        if self.attributes_categories and (self.depth < max_depth):
            self.terminal = False
            self.complete = False
            
        else:
            self.terminal = True
            self.complete = True
        
        self.value, self.value_terminal, self.value_greedy = None, None, None
        self.freq, self.pred = None, 0
        dict_branches[self.id_branch] = self    # Add the branch to the memo.
        
    def set_value(self, value):
        """
        Description
        -----------------------
        Set the value of the branch.
        
        Parameters
        -----------------------
        value : Float in [0, 1], estimated value (upper bound) of the branch.
        
        Returns
        -----------------------
        """
        
        self.value = value
        
    def set_value_terminal(self, value_terminal):
        """
        Description
        -----------------------
        Set the value of the terminal action at the branch.
        
        Parameters
        -----------------------
        value_terminal : Float in [0, 1], value of taking the terminal action at the branch.
        
        Returns
        -----------------------
        """
        
        self.value_terminal = value_terminal
        
    def set_value_greedy(self, value_greedy):
        """
        Description
        -----------------------
        Set the value of following the currently best estimated policy from the branch.
        
        Parameters
        -----------------------
        value_greedy : Float in [0, 1], best calculated value so far at the branch.
        
        Returns
        -----------------------
        """
        
        self.value_greedy = value_greedy
        
    def set_freq(self, freq):
        """
        Description
        -----------------------
        Set the freq attribute.
        
        Parameters
        -----------------------
        freq : Float in [0, 1], the proportion of samples in the branch.
        
        Returns
        -----------------------
        """
        
        self.freq = freq
        
    def set_pred(self, pred):
        """
        Description
        -----------------------
        Set the pred attribute.
        
        Parameters
        -----------------------
        pred : Int, the predicted class.
        
        Returns
        -----------------------
        """
        
        self.pred = pred

    def evaluate_terminal(self, y, n_total):
        """
        Description
        -----------------------
        Evaluate the terminal action.
        
        Parameters
        -----------------------
        y        : 1D np.array, predictor column at the branch.
        n_total  : Int, the total number of samples in the training set.
        
        Returns
        -----------------------
        """
        
        n_samples = y.shape[0]
        if n_samples == 0:
            self.complete = True     # If there are no samples in the branch, cut it.
            self.value, self.value_terminal, self.value_greedy, self.freq = 0, 0, 0, 0
            
        else:
            self.pred, pred_n = Counter(y).most_common(1)[0]   # pred_n is the number of correctly classified samples.
            self.set_freq(n_samples/n_total)
            self.set_value_terminal(pred_n/n_total)
            self.set_value_greedy(self.value_terminal)
    
    def evaluate(self, y, lambd, n_total):
        """
        Description
        -----------------------
        Estimate the upper bound of the value of the branch.
        
        Parameters
        -----------------------
        y        : 1D np.array, predictor column at the branch.
        lambd    : Float in [0, 1], the complexity parameter.
        n_total  : Int, the total number of samples in the training set.
        
        Returns
        -----------------------
        """
        
        # Only evaluate the branch if it has not been evaluated before.
        if self.value is None:
            self.evaluate_terminal(y, n_total)   # When the branch is explored for the first time, evaluate it.
            if not self.terminal:   # If the branch can still be split because there are still available attributes or if the branch's depth is smaller than max_depth.
                value_split = -lambd + self.freq
                self.set_value(max(value_split, self.value_terminal))
                if self.value == self.value_terminal:
                    self.complete, self.terminal = True, True   # The terminal action has been proven to be the best at this branch.
                    
            else:    # If the branch can no longer be split because there are no more available attributes.
                self.set_value(self.value_terminal)
            
    def split(self, X_branch, y_branch, lambd, sorted_branch, attribute, n_total, max_depth):
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
        
        try:
            self.attributes_categories[attribute]
            
        except KeyError:
            raise ValueError('Attribute %d has already been used when constructing the branch %s .' %(attribute, self.id_branch))
            
        try:
            self.children[attribute]
            raise ValueError('The branch has already been split with respect to attribute %d .' %attribute)
            
        except KeyError:
            self.children[attribute] = {}
            children_queue = {}      # The dictionary of incomplete children that we store in the queue.
            attributes_categories_children = self.attributes_categories.copy()
            attributes_categories_children.pop(attribute)
            value_children, value_children_complete, value_children_greedy = 0, 0, 0
            key = None                    # The key, in children_queue, of the most promising branch to explore.
            value_greedy_min = np.inf     # Initialising the minimum value_greedy among the children branches to consider.
            bit_vector = self.bit_vector.copy()     # Create the child's bit vecotr as a copy of its parent's bit vector.
            
            for category in range(self.attributes_categories[attribute]):
                attribute_value_child = str((attribute, category))
                sorted_branch_child = sorted(sorted_branch + [attribute_value_child])
                id_branch_child = ''.join(sorted_branch_child)

                try:
                    branch_child = dict_branches[id_branch_child]

                except KeyError:
                    indices = (X_branch == category)
                    bit_vector_child = bit_vector.copy()     # Create the child's bit vecotr as a copy of its parent's bit vector.
                    bit_vector_child[bit_vector] = indices   # Update the ones in the child's bit vector with the indices corresponding to the data that the child contains.
                    bit_vector[bit_vector_child] = 0
                    y_branch_child = y_branch[indices]    # Data corresponding to the current child node.
                    np.invert(indices, out=indices)
                    X_branch = X_branch[indices]            # Data corresponding to the remaining children nodes.
                    y_branch = y_branch[indices]            # Data corresponding to the remaining children nodes.

                    branch_child = Branch(id_branch_child, attributes_categories_children, bit_vector_child, self.depth + 1, max_depth)
                    branch_child.evaluate(y_branch_child, lambd, n_total)   # When the branch is explored for the first time, we have to evaluate it.
                                                                            # Otherwise, its value has already been estimated and maybe updated through other routes.

                value_children += branch_child.value
                value_children_greedy += branch_child.value_greedy
                if branch_child.complete:
                    value_children_complete += branch_child.value

                else:
                    children_queue[category] = branch_child
                    if branch_child.value_greedy < value_greedy_min:
                        key = category
                        value_greedy_min = branch_child.value_greedy

                self.children[attribute][category] = branch_child

            heapq.heappush(self.queue, (-value_children, value_children_complete, attribute, children_queue, key))
            value_children_greedy = -lambd + value_children_greedy
            if value_children_greedy > self.value_greedy:
                self.attribute_opt = attribute
                self.set_value_greedy(value_children_greedy)

class Lattice:
    """
    Description
    -----------------------
    Class describing a lattice.
    
    Class Attributes:
    -----------------------    
    """
    
    def __init__(self, attributes_categories, n_total, max_depth):
        """
        Description
        -----------------------
        Constructor of class Lattice.

        Attributes & Parameters
        -----------------------
        attributes_categories : Dict: - keys   : Indices of the attributes to be considered for potential splits within the node.
                                      - values : Number of categories of each attribute.
        n_total               : Int, the total number of data samples.
        """
        
        self.attributes_categories = attributes_categories
        self.n_total = n_total
        self.max_depth = max_depth
        self.root = Branch('', attributes_categories, np.full(n_total, True), 0, max_depth)
        self.dict_branches = dict_branches

#    def initialise(self, data, lambd):
#        """
#        Description
#        -----------------------
#        Initialise the lattice by running a first iteration expanding the root.
#
#        Attributes & Parameters
#        -----------------------
#        """
#        
#        self.root.evaluate(data[:, -1], lambd, self.n_total)
#        self.expand(self.root, data, lambd, [])
        
    def select(self):
        """
        Description
        -----------------------
        Select the most promising branch according to the current value estimates.

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
            
        sorted_branch.sort()
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
        
        for attribute in branch.attributes_categories:
            X_branch, y_branch = data_branch[:, attribute], data_branch[:, -1]
            branch.split(X_branch, y_branch, lambd, sorted_branch, attribute, n_total, self.max_depth)

        value_max = -branch.queue[0][0] - lambd
        branch.set_value(value_max)
        if not branch.queue[0][-2]:
            branch.complete = True   # The branch is complete if its best set of children is complete (children_queue is empty).
            
        if branch.value <= branch.value_terminal:
            branch.complete, branch.terminal = True, True
            branch.set_value(branch.value_terminal)
                        
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
        
        start_time = time()
        length = len(path)
        branch = path[-1]
        index_parent = -2
        while index_parent >= -length:
            parent = path[index_parent]
            _, value_complete, attribute, children, key = heapq.heappop(parent.queue)   # The tuple in the queue that includes branch.
            if branch.complete:
                value_complete += branch.value
                children.pop(key)

            value = value_complete
            value_greedy = value_complete
            value_greedy_min = np.inf
            for category, child in children.items():
                value += child.value
                value_greedy += child.value_greedy
                if child.value_greedy < value_greedy_min:
                    value_greedy_min = child.value_greedy
                    key = category
                
            value_greedy = -lambd + value_greedy
            if value_greedy > parent.value_greedy:
                parent.attribute_opt = attribute
                parent.set_value_greedy(value_greedy)
            
            tuple_new = (-value, value_complete, attribute, children, key)
            heapq.heappush(parent.queue, tuple_new)
            value_parent = - lambd - parent.queue[0][0]
            parent.set_value(max(value_parent, parent.value_terminal))
            if parent.value == parent.value_terminal:
                parent.complete, parent.terminal = True, True

            if not parent.queue[0][-2]:
                parent.complete = True   # The parent branch is complete if its best set of children is composed of complete branches.
                
            branch = parent
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
#            branch = branch.children[branch.attribute_opt][X[branch.attribute_opt]]
            # When a feature has a value that was not observed during training, predict the class of the current branch (node).
            try:
                branch = branch.children[branch.attribute_opt][X[branch.attribute_opt]]

            except KeyError:
#                warnings.warn("The value %d of feature %d was not obsevred during training" %(X[branch.attribute_opt], branch.attribute_opt))
                warnings.warn("The input " + str(X) + " could not be properly sorted into a leaf because the value " + str(X[branch.attribute_opt]) + " of feature " + str(branch.attribute_opt)
                             + " was not observed during training.")
                return branch.pred
                
        return branch.pred
            
    def build_string_node(self, branch, show_classes):
        """
        Description
        --------------
        Build string representation (with parentheses) of the optimal subtree rooted at branch.
        
        Parameters
        --------------
        branch       : Branch, the branch from which we want to build the optimal subtree.
        show_classes : Boolean, whether to show the predicted classes or not.
        
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
        
#        children_opt = branch.children[branch.attribute_opt]
        for category in range(self.attributes_categories[branch.attribute_opt]):
            child = branch.children[branch.attribute_opt][category]
            string += '(X_' + str(branch.attribute_opt) + '=' + str(category) + ' ' + self.build_string_node(child, show_classes) + ') '

        return string

    def build_string_node_compact(self, branch, show_classes):
        """
        Description
        --------------
        Build a compact string representation (with parentheses) of the optimal subtree rooted at branch.
        
        Parameters
        --------------
        branch       : Branch, the branch from which we want to build the optimal subtree.
        show_classes : Boolean, whether to show the predicted classes or not.
        
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

        dict_subtrees = {}
        for category in range(self.attributes_categories[branch.attribute_opt]):
            child = branch.children[branch.attribute_opt][category]
            dict_subtrees[category] = self.build_string_node_compact(child, show_classes)

        while dict_subtrees:
            categories_string = ''
            category = list(dict_subtrees.keys())[0]
            categories_string += str(category)
            subtree_category = dict_subtrees[category]
            dict_subtrees.pop(category)
            for category_ in list(dict_subtrees.keys()):
                if dict_subtrees[category_] == subtree_category:
                    categories_string += ',' + str(category_)
                    dict_subtrees.pop(category_)

            string += '(X_' + str(branch.attribute_opt) + '=' + '{' + categories_string + '} ' + subtree_category + ') '

        return string
    
    def build_string(self, show_classes=True, compact=False):
        """
        Description
        --------------
        Build string representation of the optimal Decision Tree.
        
        Parameters
        --------------
        show_classes : Boolean, whether to show the predicted classes or not.
        
        Returns
        --------------
        String representation of the optimal Decision Tree.
        """

        if compact:
            return '( ' + self.build_string_node_compact(self.root, show_classes) + ')'

        else:
            return '( ' + self.build_string_node(self.root, show_classes) + ')'
    
    def plot_tree(self, show_classes=True, compact=False):
        """
        Description
        --------------
        Plot the decision tree.
        
        Parameters
        --------------
        show_classes : Boolean, whether to show the predicted classes or not.
        
        Returns
        --------------
        nltk tree object, visualize the optimal Decision Tree.
        """

        return Tree.fromstring(self.build_string(show_classes, compact))

        