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
                - value : BranchMulti, stored branch.
"""

class BranchMulti:
    """
    Description
    -----------------------
    Class describing a branch.
    
    Class Attributes:
    -----------------------
    """
    
    def __init__(self, id_branch, attributes, bit_vector, K, depth, max_depth):
        """
        Description
        -----------------------
        Constructor of class BranchMulti.

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
        """
        
        self.id_branch = id_branch
        self.attributes = attributes
        self.bit_vector = bit_vector
        self.depth = depth
        self.attribute_opt = None
        if attributes and (self.depth < max_depth):
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
        self.n_classes = np.zeros(K)
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
        
    def evaluate_terminal(self, y, n_total, n_samples=None, n_classes=None):
        """
        Description
        -----------------------
        Evaluate the terminal action.

        Parameters
        -----------------------
        y         : 1D np.array, predictor column at the branch.
        n_total   : Int, the total number of samples in the training set.
        n_samples : Int, the number of observed examples in the branch. None if the branch has not been visited yet.
        n_classes : 1D np.array of shape (K,), the kth entry of the array stores the number of examples of class k.

        Returns
        -----------------------
        """
        
        if n_samples is not None:
            self.n_samples = n_samples
            self.n_classes = n_classes
            pred = np.argmax(n_classes)
            pred_n = n_classes[pred]
            self.pred = pred
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

        else:
            count = Counter(y).most_common()
            pred, pred_n = count[0]
            self.pred = pred
            for tup in count:
                self.n_classes[tup[0]] = tup[1]

            self.freq = n_samples/n_total
            value = pred_n/n_total
            self.value_terminal = value
            self.value_greedy = value
            
    def evaluate(self, y, lambd, n_total, n_samples=None, n_classes=None):
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
        n_classes : 1D np.array of shape (K,), the kth entry of the array stores the number of examples of class k.

        Returns
        -----------------------
        """
        
        # Only evaluate the branch if it has not been evaluated before.
        if self.value is None:
            self.evaluate_terminal(y, n_total, n_samples, n_classes)   # When the branch is explored for the first time, evaluate it.
            if not self.terminal:   # If the branch can still be split because there are still available attributes.
                value_split = -lambd + self.freq
                self.set_value(max(value_split, self.value_terminal))
                if self.value == self.value_terminal:
                    self.complete, self.terminal = True, True   # The terminal action has been proven to be the best at this branch.

            else:    # If the branch can no longer be split because there are no more available attributes.
                self.set_value(self.value_terminal)
            
    def split_(self, data, lambd, attribute, n_total, K, max_depth):
        """
        Description
        -----------------------
        Split a branch.

        Parameters
        -----------------------
        X_branch      : 1D np.array, attribute vector at the branch we want to split.
        y_branch      : 1D np.array, class vector at the branch we want to split.
        lambd         : Float in [0, 1], the complexity parameter.
        attribute     : Int, the attribute to use for the split. Note that the attribute has to be among the keys of dictionary attributes_categories.
        n_total       : Int, the total number of samples in the training set.

        Returns
        -----------------------
        """

        id_branch_left = str((attribute, 0))
        id_branch_right = str((attribute, 1))
        self.children[attribute] = {}
        children_queue = {}      # The dictionary of incomplete children that we store in the queue.
        X_branch, y_branch = data[:, attribute], data[:, -1]

        # Constructing the right child.
        bit_vector_right = (X_branch == 1)
        n_samples_right = np.count_nonzero(bit_vector_right)
        if (n_samples_right == self.n_samples) or (n_samples_right == 0):
            return
        
        y_branch_right = y_branch[bit_vector_right]    # Data corresponding to the current child node.
        count = Counter(y_branch_right).most_common()
        n_classes_right = np.zeros(K)
        for tup in count:
            n_classes_right[tup[0]] = tup[1]
            
        attributes_children = self.attributes.copy()
        attributes_children.remove(attribute)
        branch_right = BranchMulti(id_branch_right, attributes_children, bit_vector_right, K, self.depth + 1, max_depth)
        branch_right.evaluate(None, lambd, n_total, n_samples_right, n_classes_right)

        # Constructing the left child.
        bit_vector_left = ~bit_vector_right
        branch_left = BranchMulti(id_branch_left, attributes_children, bit_vector_left, K, self.depth + 1, max_depth)
        branch_left.evaluate(None, lambd, n_total, self.n_samples - n_samples_right, self.n_classes - n_classes_right)

        self.children[attribute][0] = branch_left
        self.children[attribute][1] = branch_right

        value_children_complete = 0
        key = None                    # The key, in children_queue, of the most promising branch to explore.
        value_greedy_min = np.inf     # Initialising the minimum value_greedy among the children branches to consider.
        value_children = branch_left.value + branch_right.value
        value_children_greedy = branch_left.value_greedy + branch_right.value_greedy
        if branch_left.complete:
            value_children_complete += branch_left.value

        else:
            children_queue[0] = branch_left
            key = 0
            value_greedy_min = branch_left.value_greedy

        if branch_right.complete:
            value_children_complete += branch_right.value

        else:
            children_queue[1] = branch_right
            if branch_right.value_greedy < value_greedy_min:
                key = 1

#        heapq.heappush(self.queue, (-value_children, value_children_complete, attribute, children_queue, key))
        heapq.heappush(self.queue, (-value_children, -value_children_complete, attribute, children_queue, key))
        value_children_greedy = -lambd + value_children_greedy
        if value_children_greedy > self.value_greedy:
            self.attribute_opt = attribute
            self.value_greedy = value_children_greedy

    def split(self, y, attribute, attribute_parent, parent, lambd, sorted_branch, n_total, K, max_depth):
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

        attribute_category_left = str((attribute, 0))
        sorted_branch_left = sorted(sorted_branch + [attribute_category_left])
        id_branch_left = ''.join(sorted_branch_left)
        id_branch_right = id_branch_left.replace(attribute_category_left, str((attribute, 1)))
        self.children[attribute] = {}
        children_queue = {}      # The dictionary of incomplete children that we store in the queue.
        
        try:
            branch_left = dict_branches[id_branch_left]
            if (branch_left.n_samples == self.n_samples) or (branch_left.n_samples == 0):
                return
                
            try:
                branch_right = dict_branches[id_branch_right]

            except KeyError:
                bit_vector_left = branch_left.bit_vector
                bit_vector_right = np.logical_and(self.bit_vector, ~bit_vector_left)
                branch_right = BranchMulti(id_branch_right, branch_left.attributes, bit_vector_right, K, self.depth + 1, max_depth)
                branch_right.evaluate(None, lambd, n_total, self.n_samples - branch_left.n_samples, self.n_classes - branch_left.n_classes)

        except KeyError:
            try:
                branch_right = dict_branches[id_branch_right]
                if (branch_right.n_samples == self.n_samples) or (branch_right.n_samples == 0):
                    return
                
            except KeyError:
                # Calculating the bit vector of the right child.
                try:
                    bit_vector_right = np.logical_and(self.bit_vector, parent.children[attribute][1].bit_vector)

                except KeyError: # If the parent has no child with respect to attribute, then this attribute should not be considered further.
                    return
                    
                n_samples_right = np.count_nonzero(bit_vector_right)
                if (n_samples_right == self.n_samples) or (n_samples_right == 0):
                    return

                attributes_children = self.attributes.copy()
                attributes_children.remove(attribute)
                branch_right = BranchMulti(id_branch_right, attributes_children, bit_vector_right, K, self.depth + 1, max_depth)
                
                sorted_branch_sibling = sorted_branch[:-1] + [str((attribute_parent, 1 - int(sorted_branch[-1][-2])))]  # Sorted branch list of the sibling of the branch being split.
                sorted_branch_sibling_left = sorted(sorted_branch_sibling + [attribute_category_left])
                id_branch_sibling_left = ''.join(sorted_branch_sibling_left)  # Id of the left child of the sibling when splitting with respect to attribute.
                id_branch_sibling_right = id_branch_sibling_left.replace(attribute_category_left, str((attribute, 1)))  # Id of the right child.
                try:
                    branch_sibling_right = dict_branches[id_branch_sibling_right]
                    n_classes_right = parent.children[attribute][1].n_classes - branch_sibling_right.n_classes

                except KeyError:
                    try:
                        branch_sibling_left = dict_branches[id_branch_sibling_left]
                        n_classes_right = parent.children[attribute][1].n_classes - parent.children[attribute_parent][1 - int(sorted_branch[-1][-2])].n_classes + branch_sibling_left.n_classes

                    except KeyError:
                        count = Counter(y[bit_vector_right]).most_common()
                        n_classes_right = np.zeros(K)
                        for tup in count:
                            n_classes_right[tup[0]] = tup[1]

                branch_right.evaluate(None, lambd, n_total, n_samples_right, n_classes_right)

            bit_vector_left = np.logical_and(self.bit_vector, parent.children[attribute][0].bit_vector)
            branch_left = BranchMulti(id_branch_left, branch_right.attributes, bit_vector_left, K, self.depth + 1, max_depth)
            branch_left.evaluate(None, lambd, n_total, self.n_samples - branch_right.n_samples, self.n_classes - branch_right.n_classes)

        self.children[attribute][0] = branch_left
        self.children[attribute][1] = branch_right

        value_children_complete = 0
        key = None                    # The key, in children_queue, of the most promising branch to explore.
        value_greedy_min = np.inf     # Initialising the minimum value_greedy among the children branches to consider.
        value_children = branch_left.value + branch_right.value
        value_children_greedy = branch_left.value_greedy + branch_right.value_greedy
        if branch_left.complete:
            value_children_complete += branch_left.value

        else:
            children_queue[0] = branch_left
            key = 0
            value_greedy_min = branch_left.value_greedy

        if branch_right.complete:
            value_children_complete += branch_right.value

        else:
            children_queue[1] = branch_right
            if branch_right.value_greedy < value_greedy_min:
                key = 1

#        heapq.heappush(self.queue, (-value_children, value_children_complete, attribute, children_queue, key))
        heapq.heappush(self.queue, (-value_children, -value_children_complete, attribute, children_queue, key))
        value_children_greedy = -lambd + value_children_greedy
        if value_children_greedy > self.value_greedy:
            self.attribute_opt = attribute
            self.value_greedy = value_children_greedy

        
class LatticeMulti:
    """
    Description
    -----------------------
    Class describing a lattice for binary classification problems with binary predictors.
    
    Class Attributes:
    -----------------------    
    """
    
    def __init__(self, attributes, n_total, K, max_depth):
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
        self.K = K
        self.max_depth = max_depth
        self.root = BranchMulti('', attributes, np.full(n_total, True), K, 0, max_depth)
        self.dict_branches = dict_branches

    def initialise(self, data, lambd):
        """
        Description
        -----------------------
        Initialise the lattice by running a first iteration expanding the root.

        Attributes & Parameters
        -----------------------
        """
        
        self.root.evaluate(data[:, -1], lambd, self.n_total)
        for attribute in self.root.attributes:
            self.root.split_(data, lambd, attribute, self.n_total, self.K, self.max_depth)

        if self.root.queue:
            value_max = -self.root.queue[0][0] - lambd
            self.root.set_value(value_max)
            if not self.root.queue[0][-2]:
                self.root.complete = True   # The root is complete if its best set of children is complete (children_queue is empty).
                
            if self.root.value <= self.root.value_terminal:
                self.root.complete, self.root.terminal = True, True
                self.root.set_value(self.root.value_terminal)

        else:
            self.root.complete, self.root.terminal = True, True
            self.root.set_value(self.root.value_terminal)
        
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

    def expand(self, branch, parent, data, lambd, sorted_branch):
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
            
        y = data[:, -1]
        attribute_parent = parent.queue[0][2]
        for attribute in branch.attributes:
            branch.split(y, attribute, attribute_parent, parent, lambd, sorted_branch, self.n_total, self.K, self.max_depth)

        if branch.queue:
            value_max = -branch.queue[0][0] - lambd
            branch.set_value(value_max)
            if not branch.queue[0][-2]:
                branch.complete = True   # The branch is complete if its best set of children is complete (children_queue is empty).
                
            if branch.value <= branch.value_terminal:
                branch.complete, branch.terminal = True, True
                branch.set_value(branch.value_terminal)

        else:
            branch.complete, branch.terminal = True, True
            branch.set_value(branch.value_terminal)
                        
    def update_parent(self, branch_parent, lambd):
        """
        Description
        -----------------------
        Update a parent branch during Backpropagation.
        
        Parameters
        -----------------------
        lambd : Float in [0, 1], the complexity parameter.
        
        Returns
        -----------------------
        """

        attributes_treated = set()
#        value_neg, value_complete, attribute, children, key = branch_parent.queue[0]
        value_neg, value_complete_neg, attribute, children, key = branch_parent.queue[0]
        while attribute not in attributes_treated:
            value = -value_neg
            value_complete = -value_complete_neg
            if not children:
                branch_parent.complete = True
                branch_parent.attribute_opt = attribute
                branch_parent.value_greedy = max(branch_parent.value_greedy, -lambd + value)
                branch_parent.set_value(max(branch_parent.value_terminal, -lambd + value))
                if branch_parent.value == branch_parent.value_terminal:
                    branch_parent.terminal = True
                    branch_parent.attribute_opt = None
                
                return
    
            value = value_complete
            value_greedy = value_complete
            value_greedy_min = np.inf
            categories_to_discard = set()
            for category, branch_child in children.items():
                value += branch_child.value
                value_greedy += branch_child.value_greedy
                if branch_child.complete:
                    categories_to_discard.add(category)
                    value_complete += branch_child.value
                    
                elif branch_child.value_greedy < value_greedy_min:
                    value_greedy_min = branch_child.value_greedy
                    key = category

            for category in categories_to_discard:
                children.pop(category)
    
            value_greedy = -lambd + value_greedy
            if value_greedy > branch_parent.value_greedy:
                branch_parent.attribute_opt = attribute
                branch_parent.value_greedy = value_greedy
            
#            tuple_new = (-value, value_complete, attribute, children, key)
            tuple_new = (-value, -value_complete, attribute, children, key)
            heapq.heapreplace(branch_parent.queue, tuple_new)
            attributes_treated.add(attribute)
#            value_neg, value_complete, attribute, children, key = branch_parent.queue[0]
            value_neg, value_complete_neg, attribute, children, key = branch_parent.queue[0]

#        value_neg, value_complete, attribute, children, key = branch_parent.queue[0]
        value_neg, value_complete_neg, attribute, children, key = branch_parent.queue[0]
        value = -value_neg
        value_complete = -value_complete_neg
        branch_parent.set_value(max(-lambd + value, branch_parent.value_terminal))
        if branch_parent.value == branch_parent.value_terminal:
            branch_parent.complete, branch_parent.terminal = True, True
            branch_parent.attribute_opt = None
            return
            
        if not children:
            branch_parent.complete = True   # The parent branch is complete if its best set of children is composed of complete branches.

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
        index_parent = -2
        while index_parent >= -length:
            branch_parent = path[index_parent]
            self.update_parent(branch_parent, lambd)
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
        show_classes : Boolean, whether to show the predicted classes or not.
        
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
        show_classes : Boolean, whether to show the predicted classes or not.
        
        Returns
        --------------
        nltk tree object, visualize the optimal Decision Tree.
        """

        return Tree.fromstring(self.build_string(show_classes))

