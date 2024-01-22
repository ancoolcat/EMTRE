import math
import random

import numpy as np
from numpy.random import rand
from fitness_function import Fun

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X


def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]

    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()

    return V, Vmax, Vmin


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x



def jfs(xtrain, ytrain, opts,rmp):
    # Parameters
    ub = 1  # upper boundary
    lb = 0  # lower boundary
    thres = 0.5 # threshold

    # population size
    N = opts['N']
    # Maximum number of iterations
    max_iter = opts['N_task']
    tasks = opts['task']
    N_task = len(tasks)

    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']

    task_intersection = {}
    # Intersect the list two by two and store the result in a dictionary
    for i in range(len(tasks)):
        for j in range(i+1,len(tasks)):
            intersection = list(set(tasks[i]).intersection(set(tasks[j])))
            index_list1 = [(i, tasks[i].index(x)) for x in intersection ]
            index_list2 = [(j, tasks[j].index(x)) for x in intersection ]
            task_intersection[(i, j)] = (intersection, index_list1,index_list2)
            task_intersection[(j, i)] = (intersection, index_list2, index_list1)


    # Initialize information for all tasks
    imformation_task = []
    for task in tasks:
        xtrain_task = xtrain[:, task]
        dim = np.size(xtrain_task, 1)
        ub_tem = ub * np.ones([1, dim], dtype='float')  # 1
        lb_tem = lb * np.ones([1, dim], dtype='float')  # 0

        # Initialize position & velocity
        X = init_position(lb_tem, ub_tem, int(N / len(tasks)), dim)
        V, Vmax, Vmin = init_velocity(lb_tem, ub_tem, int(N / len(tasks)), dim)

        # Pre
        fit = np.zeros([int(N / len(tasks)), 1], dtype='float')
        Xgb = np.zeros([1, dim], dtype='float')  # Xgbest
        fitG = float('inf')  # Gbest
        Xpb = np.zeros([int(N / len(tasks)), dim], dtype='float')  # Xpbest
        fitP = float('inf') * np.ones([int(N / len(tasks)), 1], dtype='float')#Pbest

        n = int(N / len(tasks)) #Individual population size
        imformation_task.append([xtrain_task.copy(), dim, ub_tem, lb_tem, X, V, Vmax, Vmin,
                                 fit, Xpb, fitP, n, Xgb, fitG])

    curve = np.zeros([1, max_iter], dtype='float')  # convergence curve
    min_figG = 1
    min_dim = np.size(xtrain, 1) + 1
    min_xgb = []
    min_i = 0

    t = 0  # Iteration begins
    while t < max_iter:

        for task_i in range(len(tasks)):
            xtrain_task, dim, ub_tem, \
            lb_tem, X, V, Vmax, Vmin, \
            fit, Xpb, fitP, n, Xgb, fitG = imformation_task[task_i]

            # Binary conversion
            Xbin = binary_conversion(X, thres, n, dim)
            opts['i'] = task_i
            # Fitness
            for i in range(n):
                fit[i, 0] = Fun(xtrain_task, ytrain, Xbin[i, :], opts)
                if fit[i, 0] < fitP[i, 0]:
                    Xpb[i, :] = X[i, :]
                    fitP[i, 0] = fit[i, 0]
                if fitP[i, 0] < fitG:
                    Xgb[0, :] = Xpb[i, :]
                    fitG = fitP[i, 0]
            imformation_task[task_i][-1] = fitG  # fitG updates the global optimum for the current task

        # Record the maximum value of all current tasks
        for task_i in range(len(tasks)):
            if imformation_task[task_i][-1] < min_figG:
                min_figG = imformation_task[task_i][-1]
                min_xgb = imformation_task[task_i][-2].copy()
                min_dim = imformation_task[task_i][1]
                min_i = task_i


        # Store result
        curve[0, t] = min_figG
        print("Iteration:", t + 1)
        print("Best (EMTRE):", curve[0, t])
        t += 1

        w = 0.9 - (0.5 * (t - 1) / max_iter)  # inertia weight
        c1 = 1.49445  # acceleration factor
        c2 = 1.49445  # acceleration factor
        for task_i in range(N_task):
            xtrain_task, dim, ub_tem, \
            lb_tem, X, V, Vmax, Vmin, \
            fit, Xpb, fitP, n, Xgb, fitG = imformation_task[task_i]
            if rand() <= rmp:
                #knowledge transfer
                for i in range(n):
                    # Tournament Selectionm m
                    li = [k  for k in range(N_task) if k!=task_i]
                    tournament_size = min(len(li), 3)
                    li=random.sample(li, tournament_size)
                    m = li[0]
                    for number in li:
                        if imformation_task[number][-1] < imformation_task[m][-1]:
                            m = number

                    # Starting knowledge transfer
                    Xgb_new=np.zeros([1, dim], dtype='float')
                    # Take the intersection of the two tasks
                    inSection ,index_i,index_j= task_intersection[(task_i,m)]
                    xgb_a = imformation_task[task_i][-2].copy()
                    xgb_b = imformation_task[m][-2].copy()
                    for insec in range(len(inSection)):

                        a = index_i[insec][1]#Take the index of the insec intersection element of task i at task i
                        b = index_j[insec][1]#Take the index of the insec intersection element of task j at task j

                        # Generate guiding vectors
                        D_alpha = abs(2 * rand() * xgb_a[0, a])
                        D_beta = abs(2 * rand() * xgb_b[0, b])
                        A1 = (2 * rand() - 1) * (1 - t / max_iter)
                        A2 = (2 * rand() - 1) * (1 - t / max_iter)
                        G_1 = xgb_a[0, a] - A1 * D_alpha
                        G_2 = xgb_b[0, b] - A2 * D_beta
                        # Generate a new solution
                        Xgb_new[0, a] = (G_1 + G_2)/2
                        Xgb_new[0,a]=boundary(Xgb_new[0,a], lb_tem[0, a], ub_tem[0, a])


                    for d in range(dim):
                        # Update velocity
                        r11 = rand()
                        r22 = rand()
                        V[i, d] = w * V[i, d] + c1 * r11 * (Xpb[i, d] - X[i, d]) + c2 * r22 * (Xgb_new[0, d] - X[i, d])
                        # Boundary
                        V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                        # Update position
                        X[i, d] = X[i, d] + V[i, d]
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb_tem[0, d], ub_tem[0, d])
            else:
                # no knowledge transfer
                for i in range(n):
                    for d in range(dim):
                        # Update velocity
                        r11 = rand()
                        r22 = rand()
                        V[i, d] = w * V[i, d] + c1 * r11 * (Xpb[i, d] - X[i, d]) + c2 * r22 * (Xgb[0, d] - X[i, d])
                        # Boundary
                        V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                        # Update position
                        X[i, d] = X[i, d] + V[i, d]
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb_tem[0, d], ub_tem[0, d])



    # final result
    opts['i'] = min_i

    Gbin = binary_conversion(min_xgb, thres, 1, min_dim)

    Gbin = list(Gbin.reshape(min_dim))
    fea_list = np.asarray([0] * np.size(xtrain, 1))
    for i in range(len(Gbin)):
        if Gbin[i] == 1:
            fea_list[tasks[min_i][i]] = 1
    # print("Selected features：" + str(fea_list))
    pos = np.asarray(range(0, np.size(xtrain, 1)))
    sel_index = pos[fea_list == 1]
    # print("Selected Feature Positions：" + str(sel_index))
    num_feat = len(sel_index)
    print("The number of features is " + str(num_feat))
    # Create dictionary
    pso_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return pso_data
