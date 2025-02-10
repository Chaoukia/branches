import svgling
import argparse
from branches import *

def str_to_bool(string):
    assert (string == 'True') or (string == 'False'), "The argument should be either 'True' or 'False'."
    if string == 'False':
        return False

    elif string == 'True':
        return True

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Parse options')
    
    argparser.add_argument('--path', type=str, help="Path to the data file. The data must be preprocessed with some sore of encoding (Ordinal or OneHot for example). The predicted variable is the last column and the data is delimited with ' '.")
    argparser.add_argument('--lambd', type=float, default=0.01, help="Float in ]0, 1[, the complexity parameter.")
    argparser.add_argument('--n', type=int, default=5000000, help="Maximum number of iterations.")
    argparser.add_argument('--print_iter', type=int, default=10000, help="Number of iterations between two prints.")
    argparser.add_argument('--time_limit', type=int, default=300, help="Time limit.")
    argparser.add_argument('--encoding', type=str, default='ordinal', help="Type of encoding of the loaded data, in {'ordinal', 'binary', 'multi'}.")
    argparser.add_argument('--max_depth', type=float, default=np.inf, help="Maximum depth constraint.")
    argparser.add_argument('--show_classes', type=str, default='True', help="Whether to show the classes when drawing the DT solution or not.")
    argparser.add_argument('--compact', type=str, default='True', help="Whether to collapse the DT solution or not.")
    argparser.add_argument('--path_tree', type=str, default="../trees/tree.svg", help="Path to the file where to save the DT solution.")
    
    args = argparser.parse_args()

    data = np.genfromtxt(args.path, delimiter=' ', dtype=int)
    tree = summary(data, args.lambd, args.n, args.print_iter, args.time_limit, args.encoding, args.max_depth, str_to_bool(args.show_classes), str_to_bool(args.compact))
    svgling.draw_tree(tree).saveas(args.path_tree, pretty=True)
    print('The tree has been saved in ' + args.path_tree)
    
    



    
    

















