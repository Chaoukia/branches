import svgling
import pandas as pd
from branches import *
from time import time

def summary(data, lambd=0.01, n=5000000, print_iter=10000, time_limit=300, encoding='ordinal', max_depth=np.inf, show_classes=True, compact=True):
    alg = Branches(data, encoding, max_depth)
    start_time = time()
    iterations = alg.solve(lambd, n=n, print_iter=print_iter, time_limit=time_limit)
    time_execution = time() - start_time
    print('Execution time        : %.4f' %time_execution)
    print('Number of iterations  : %d' %iterations)
    branches, splits = alg.lattice.infer()
    n_branches = len(branches)
    print('Number of branches    :', n_branches)
    print('Number of splits      :', splits)
    d = retrieve_depth(branches)
    print('Depth                 :', d)
    acc = ((alg.predict(data[:, :-1]) == data[:, -1]).sum())/alg.n_total
    print('Accuracy :', acc)
    obj = acc - lambd*splits
    print('Regularised objective :', obj)
    tree = alg.plot_tree(show_classes, compact)
    alg.reinitialise()
    return tree, obj, acc, splits, n_branches, d, iterations, time_execution

arguments = {'monk1-o' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'monk1' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'monk1-l' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'monk1-f' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'monk2-o' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'monk2' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'monk2-f' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'monk3-o' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'monk3' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'monk3-f' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'tictactoe-o' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'tictactoe' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'tictactoe-f' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'binary'}, 
             'careval-o' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'careval' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'careval-f' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'nursery-o' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'nursery' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'nursery-f' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'mushroom-o' : {'lambd' : 0.01, 'print_iter' : 1000, 'encoding' : 'ordinal'}, 
             'mushroom' : {'lambd' : 0.01, 'print_iter' : 1000, 'encoding' : 'binary'}, 
             'mushroom-f' : {'lambd' : 0.01, 'print_iter' : 1000, 'encoding' : 'binary'}, 
             'krvskp' : {'lambd' : 0.01, 'print_iter' : 1000, 'encoding' : 'binary'}, 
             'zoo-o' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'zoo' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'zoo-f' : {'lambd' : 0.001, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'lymph-o' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'lymph' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'lymph-f' : {'lambd' : 0.01, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'balance-o' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'ordinal'}, 
             'balance' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'multi'}, 
             'balance-f' : {'lambd' : 0.005, 'print_iter' : 10000, 'encoding' : 'multi'}, 
            }

if __name__ == '__main__':
    results_array = np.empty((len(arguments), 7))
    for i, file_name in enumerate(arguments):
        print('Experiment on ' + file_name)
        data = np.genfromtxt('../data/preprocessed/' + file_name + '.txt', delimiter=' ', dtype=int)
        dict_arguments = arguments[file_name]
        tree, obj, acc, splits, n_branches, d, iterations, time_execution = summary(data, lambd=dict_arguments['lambd'], print_iter=dict_arguments['print_iter'], 
                                                                                    encoding=dict_arguments['encoding'])
        results_array[i, :] = [obj, acc, splits, n_branches, d, iterations, time_execution]
        path_tree = '../trees/' + file_name + '-tree.svg'
        svgling.draw_tree(tree).saveas(path_tree, pretty=True)
        print('The tree has been saved in ' + path_tree)
        print('\n')

    results_df = pd.DataFrame(results_array, index=arguments.keys(), columns=['Objective', 'Accuracy', 'Splits', 'Leaves', 'Depth', 'Iterations', 'Runtime'])
    results_df.to_csv('../results/experiments.csv', )

