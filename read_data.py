import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split



def read_mat(path=None):
    mat = scipy.io.loadmat(path)
    if('X' in mat):
        X = mat['X']
        y = mat['Y'][:, 0]
    else:
        data=mat['data']
        # data = data[~np.isnan(data).any(axis=1)]
        X=data[:,1:]
        y=data[:,0]
    n_samples, n_features = np.shape(X)
    print(n_samples, n_features)
    n_labels = np.shape(y)
    print(n_labels)
    print(type(X))
    print(type(y))
    return X,y


def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test