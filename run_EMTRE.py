import numpy as np
import warnings
from task_sampling import a_res_based_sampling
from sklearn import preprocessing
warnings.filterwarnings("ignore")
import task_selection
from skfeature.function.similarity_based import reliefF
import EMTRE
from read_data import read_mat, split_data
from result_evaluate import result_evaluate


#Standardization and normalization
def map_data(data,MIN,MAX):
    zscore_scaler = preprocessing.StandardScaler()
    data_scaler = zscore_scaler.fit_transform(np.array(data).reshape(-1,1))
    d_max = np.max(data_scaler)
    d_min = np.min(data_scaler)
    Max_Min_data=MIN +(MAX-MIN)/(d_max-d_min) * (data_scaler - d_min)
    return Max_Min_data.reshape(-1,)

def dij_distance(list1, list2,beta):

    set1 = set(list1)
    set2 = set(list2)

    common_elements = set1.intersection(set2)

    l_fea = len(common_elements)/len(list1)+len(common_elements)/len(list2)

    return abs(l_fea/2-beta)


def task_generation(X_train, y_train, beta, N_task=5):

    theta = reliefF.reliefF(X_train, y_train)
    theta = map_data(theta, 0.2, 0.8).tolist()


    N_sam = 50

    Tasks=a_res_based_sampling(theta,N_sam)
    matrixs=[]
    for task in Tasks:
        matrix=[]
        for task2 in Tasks:
            dis=dij_distance(task,task2,beta)
            matrix.append(dis)
        matrixs.append(matrix)

    preList=task_selection.HSP(matrixs, N_sam, N_task)

    indices = []
    for i in preList:
        indices.append(Tasks[i])
    print("End of task generation.")
    return indices


if __name__ == '__main__':
    paths = "dataset/warpAR10P.mat"
    k = 5  # KNN
    N = 20  # population size
    T = 200  # Maximum number of iterations

    beta=0.25
    rmp = 0.4
    N_task = 8
    X, y = read_mat(paths)
    xtrain, xtest, ytrain, ytest = split_data(X, y)
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

    Tasks = task_generation(xtrain, ytrain, beta, N_task)
    opts = {'k': k, 'fold': fold, 'N': N, 'N_task': T, 'task': Tasks}
    fmdl = EMTRE.jfs(xtrain, ytrain, opts, rmp)

    result_evaluate(fmdl, xtrain, xtest, ytrain, ytest)

