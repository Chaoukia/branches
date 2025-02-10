import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

if __name__ == '__main__':

    ########### monk1-o
    
    data = np.genfromtxt('../monks-1.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('monk1-o.txt', data, fmt='%s')

    ########### monk1
    
    data = np.genfromtxt('../monks-1.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X, y = encoder_X.transform(X).astype(int), encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('monk1.txt', data, fmt='%s')
    
    ########### monk1-f
    
    data = np.genfromtxt('../monks-1.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X, y = encoder_X.transform(X).astype(int), encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('monk1-f.txt', data, fmt='%s')

    ########### monk1-l
    
    data = np.genfromtxt('../monks-1.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=[3, 3, 2, 3, 4, 2], sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X, y = encoder_X.transform(X).astype(int), encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('monk1-l.txt', data, fmt='%s')

    ########### monk2-o
    
    data = np.genfromtxt('../monks-2.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('monk2-o.txt', data, fmt='%s')

    ########### monk2
    
    data = np.genfromtxt('../monks-2.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X, y = encoder_X.transform(X).astype(int), encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('monk2.txt', data, fmt='%s')

    ########### monk2-f
    
    data = np.genfromtxt('../monks-2.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X, y = encoder_X.transform(X).astype(int), encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('monk2-f.txt', data, fmt='%s')

    ########### monk3-o
    
    data = np.genfromtxt('../monks-3.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('monk3-o.txt', data, fmt='%s')

    ########### monk3
    
    data = np.genfromtxt('../monks-3.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X, y = encoder_X.transform(X).astype(int), encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('monk3.txt', data, fmt='%s')

    ########### monk3-f
    
    data = np.genfromtxt('../monks-3.train', delimiter=' ', dtype=int)
    data = data[:, :-1] # Getting rid of the last column, it contains only ids.
    
    # Place the first column last.
    data_ = np.empty_like(data)
    data_[:, :-1], data_[:, -1] = data[:, 1:], data[:, 0]
    data = data_
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X, y = encoder_X.transform(X).astype(int), encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('monk3-f.txt', data, fmt='%s')

    ########### tictactoe-o
    
    data = np.genfromtxt('../tic-tac-toe.data', delimiter=',', dtype=str)
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('tictactoe-o.txt', data, fmt='%s')

    ########### tictactoe
    
    data = np.genfromtxt('../tic-tac-toe.data', delimiter=',', dtype=str)
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('tictactoe.txt', data, fmt='%s')

    ########### tictactoe-f
    
    data = np.genfromtxt('../tic-tac-toe.data', delimiter=',', dtype=str)
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('tictactoe-f.txt', data, fmt='%s')

    ########### careval-o
    
    data = np.genfromtxt('../car.data', delimiter=',', dtype=str)
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('careval-o.txt', data, fmt='%s')

    ########### careval
    
    data = np.genfromtxt('../car.data', delimiter=',', dtype=str)
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('careval.txt', data, fmt='%s')

    ########### careval-f
    
    data = np.genfromtxt('../car.data', delimiter=',', dtype=str)
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('careval-f.txt', data, fmt='%s')

    ########### nursery-o
    
    data = np.genfromtxt('../nursery.data', delimiter=',', dtype=str)
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('nursery-o.txt', data, fmt='%s')

    ########### nursery
    
    data = np.genfromtxt('../nursery.data', delimiter=',', dtype=str)
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('nursery.txt', data, fmt='%s')

    ########### nursery-f
    
    data = np.genfromtxt('../nursery.data', delimiter=',', dtype=str)
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('nursery-f.txt', data, fmt='%s')

    ########### mushroom-o
    
    data = np.genfromtxt('../agaricus-lepiota.data', delimiter=',', dtype=str)[:, ::-1]
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('mushroom-o.txt', data, fmt='%s')

    ########### mushroom
    
    data = np.genfromtxt('../agaricus-lepiota.data', delimiter=',', dtype=str)[:, ::-1]
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('mushroom.txt', data, fmt='%s')

    ########### mushroom-f
    
    data = np.genfromtxt('../agaricus-lepiota.data', delimiter=',', dtype=str)[:, ::-1]
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('mushroom-f.txt', data, fmt='%s')

    ########### krvskp
    
    data = np.genfromtxt('../kr-vs-kp.data', delimiter=',', dtype=str)

    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('krvskp.txt', data, fmt='%s')

    ########### zoo-o
    
    data = np.genfromtxt('../CP4IM/zoo.data', delimiter=',', dtype=str)[:, 1:]

    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('zoo-o.txt', data, fmt='%s')

    ########### zoo
    
    data = np.genfromtxt('../CP4IM/zoo.data', delimiter=',', dtype=str)[:, 1:]

    X, y = data[:, :-1], data[:, -1]
    
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('zoo.txt', data, fmt='%s')

    ########### zoo-f
    
    data = np.genfromtxt('../CP4IM/zoo.data', delimiter=',', dtype=str)[:, 1:]

    X, y = data[:, :-1], data[:, -1]
    
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('zoo-f.txt', data, fmt='%s')

    ########### lymph-o
    
    data = np.genfromtxt('../CP4IM/lymphography.data', delimiter=',', dtype=str)[:, ::-1]
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('lymph-o.txt', data, fmt='%s')

    ########### lymph
    
    data = np.genfromtxt('../CP4IM/lymphography.data', delimiter=',', dtype=str)[:, ::-1]
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('lymph.txt', data, fmt='%s')

    ########### lymph-f
    
    data = np.genfromtxt('../CP4IM/lymphography.data', delimiter=',', dtype=str)[:, ::-1]
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('lymph-f.txt', data, fmt='%s')

    ########### balance-o
    
    data = np.genfromtxt('../balance-scale.data', delimiter=',', dtype=str)[:, ::-1]
    
    encoder = OrdinalEncoder()
    encoder.fit(data)
    data = encoder.transform(data).astype(int)
    
    np.savetxt('balance-o.txt', data, fmt='%s')

    ########### balance
    
    data = np.genfromtxt('../balance-scale.data', delimiter=',', dtype=str)[:, ::-1]
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop=None, sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('balance.txt', data, fmt='%s')

    ########### balance-f
    
    data = np.genfromtxt('../balance-scale.data', delimiter=',', dtype=str)[:, ::-1]
    
    X, y = data[:, :-1], data[:, -1]
    encoder_X, encoder_y = OneHotEncoder(drop='first', sparse_output=False), LabelEncoder()
    encoder_X.fit(X)
    encoder_y.fit(y)
    X = encoder_X.transform(X).astype(int)
    y = encoder_y.transform(y)
    data = np.hstack((X, y.reshape(-1, 1)))
    
    np.savetxt('balance-f.txt', data, fmt='%s')

