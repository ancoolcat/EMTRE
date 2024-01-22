import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

def error_rate(xtrain, ytrain, x, opts):#召回率
    # parameters
    k = opts['k']
    fold = opts['fold']
    xt = fold['xt']
    yt = fold['yt']
    xv = fold['xv']
    yv = fold['yv']

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    if 'task' in opts:
        task=opts['task'][opts['i']]
        xtrain=xt[:,task]
        xtrain = xtrain[:, x == 1]
        ytrain = yt.reshape(num_train)
        xvalid=xv[:,task]
        xvalid = xvalid[:, x == 1]
        yvalid = yv.reshape(num_valid)
    else:
        xtrain = xt[:, x == 1]
        ytrain = yt.reshape(num_train)
        xvalid = xv[:, x == 1]
        yvalid = yv.reshape(num_valid)

    # Training
    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred = mdl.predict(xvalid)
    # acc = np.sum(yvalid == ypred) / num_valid
    labels = list(set(ytrain.tolist()))
    r = recall_score(yvalid, ypred, labels=labels, average="macro")
    error = 1 - r

    return error



# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha    = 0.9
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost  = alpha * error + beta * (num_feat / max_feat)
        
    return cost

