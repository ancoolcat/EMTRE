import numpy as np
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier

def result_evaluate(fmdl,xtrain,xtest,ytrain,ytest):
    sf = fmdl['sf']
    # model with selected features
    num_train = np.size(xtrain, 0)
    num_valid = np.size(xtest, 0)
    x_train = xtrain[:, sf]
    y_train = ytrain.reshape(num_train)  # Solve bug
    x_valid = xtest[:, sf]
    y_valid = ytest.reshape(num_valid)  # Solve bug

    mdl = KNeighborsClassifier(n_neighbors=5)
    mdl.fit(x_train, y_train)

    # accuracy
    y_pred = mdl.predict(x_valid)
    labels = list(set(ytrain.tolist()))
    accu = recall_score(y_pred, y_valid, labels=labels, average="macro")

    print("Accuracy:", 100 * accu)

    # number of selected features
    num_feat = fmdl['nf']
    print("Feature Size:", num_feat)

    return num_feat,accu