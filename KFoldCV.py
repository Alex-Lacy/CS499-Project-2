'''X_mat, a matrix of numeric inputs (one row for each observation, one column for each feature).
   y_vec, a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
   ComputePredictions, a function that takes three inputs (X_train,y_train,X_new), trains a model using X_train,y_train, then outputs a vector of predictions (one element for every row of X_new).
   fold_vec, a vector of integer fold ID numbers (from 1 to K).'''
from sklearn.model_selection import KFold


import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
'''
def kfold(X_mat, y_vec, ComputePred, fold_vec, n):
    np.random.seed(2)
    X = X_mat
    y = y_vec
    kf = KFold(n_splits=len(fold_vec), shuffle=False)
    k=0
    error_vec1 = np.zeros(len(fold_vec))
    error_vec2 = np.zeros(len(fold_vec))
    num_of_1 = np.zeros(len(fold_vec))
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pred_new1 = np.zeros(len(y_test))
        pred_new2 = np.zeros(len(y_train))

        num_of_1[k]=len(np.where(y_test==0)[0])
        print(num_of_1[k])
        prob = ComputePred(X_train, y_train, X_test, n)
        if n==1:
            np.savetxt("p1.txt", prob, fmt='%f', delimiter=' ')
            np.savetxt("y1.txt",y_test , fmt='%f', delimiter=' ')
        elif n==3:
            np.savetxt("p2.txt", prob, fmt='%f', delimiter=' ')
            np.savetxt("y2.txt", y_test, fmt='%f', delimiter=' ')
        for i in range(len(y_test)):
            if prob[i][0] > prob[i][1]:
                pred_new1[i] = 0
            else:
                pred_new1[i] = 1
        prob = ComputePred(X_train, y_train, X_train, n)
        for i in range(len(y_train)):
            if prob[i][0] > prob[i][1]:
                pred_new2[i] = 0
            else:
                pred_new2[i] = 1
        count=0
        for i in range(len(y_test)):
            if pred_new1[i] == y_test[i]:
                count += 1
        error_vec1[k] = (1 - count / len(y_test)) * 100
        count=0
        for i in range(len(y_train)):
            if pred_new2[i] == y_train[i]:
                count += 1
        error_vec2[k] = (1 - count / len(y_train)) * 100
        k+=1
    return [error_vec1, error_vec2, num_of_1]
'''
def KFoldCV(X_mat, y_vec, ComputePred, fold_vec, n):
    error_vec1 = np.zeros(len(fold_vec))
    error_vec2 = np.zeros(len(fold_vec))

    np.random.seed(36)
    # fold_vec = np.array(fold_vec)
    indices = np.random.permutation(X_mat.shape[0])
    #print(indices)
    num = X_mat.shape[0]//len(fold_vec)
    for k in fold_vec:
        rg = indices[k * num:(k + 1) * num]
        X_new = X_mat[rg][:]
        y_new = y_vec[rg]
        X_train = np.delete(X_mat, rg, axis=0)
        y_train = np.delete(y_vec, rg, axis=0)
        #print(rg)

        pred_new1 = np.zeros(len(y_new))
        pred_new2 = np.zeros(len(y_train))
        count = 0

        prob = ComputePred(X_train, y_train, X_new, n)
        for i in range(len(y_new)):
            if prob[i][0] > prob[i][1]:
                pred_new1[i] = 0
            else:
                pred_new1[i] = 1
        for i in range(len(y_new)):
            if pred_new1[i] == y_new[i]:
                count+=1
        error_vec1[k] = (1 - count / len(y_new))*100

        count = 0

        prob = ComputePred(X_train, y_train, X_train, n)
        for i in range(len(y_train)):
            if prob[i][0] > prob[i][1]:
                pred_new2[i] = 0
            else:
                pred_new2[i] = 1
        for j in range(len(y_train)):
            if pred_new2[j] == y_train[j]:
                count += 1
        error_vec2[k] = (1 - count / len(y_train))*100


    return [error_vec1, error_vec2]

def NearestNeighborsCV(X_mat, y_vec, num_folds, max_neighbors, computepred):
    # validation_fold_vec=integer values from 1 to num_folds
    error_mat1 = np.zeros((max_neighbors, num_folds))
    error_mat2 = np.zeros((max_neighbors, num_folds))
    computepre=computepred
    fold_vec = list(range(num_folds))
    for n in range(max_neighbors):
        n_neighbors = n+1
        # tmp = KFoldCV(X_mat, y_vec, computepre, fold_vec, n_neighbors)
        tmp =KFoldCV(X_mat, y_vec, computepre, fold_vec, n_neighbors)
        error_mat1[n][:] = tmp[0]
        error_mat2[n][:] = tmp[1]
        num_of_1 = tmp[2]
    error_mat1 = np.array(error_mat1)
    error_mat1 = error_mat1.transpose()
    error_mat2 = np.array(error_mat2)
    error_mat2 = error_mat2.transpose()
    # best_neighbors =
    mean_error_vec = error_mat1
    return [num_of_1, error_mat1, error_mat2]

def knn(X_mat, y_vec, X_new, n):
    n_neighbors = n
    clf = KNeighborsClassifier(n_neighbors, 'uniform', 'ball_tree')
    # clf = KNeighborsClassifier(n_neighbors, 'distance', 'auto')
    # 定义一个knn分类器对象
    clf.fit(X_mat, y_vec)
    # pred_new = clf.predict(X_new)
    pred_new = clf.predict_proba(X_new)
    # print(pred_new)
    return pred_new

file = 'spam.txt'
data = np.loadtxt(file, delimiter=' ', dtype=float) #, skiprows=1)

computepred = knn

X = data[:, :-1]
y = data[:, -1]

num_folds = 7
max_neighbors = 20

Xm = np.zeros(len(y))
Xs = np.zeros(len(y))
for i in range(57):
    Xm = np.mean(X[:, i])
    Xs = np.std(X[:, i], ddof=1)

    X[:, i] = (X[:, i]-Xm)/Xs
# print(X)

[m, r1, r2] = NearestNeighborsCV(X, y, num_folds, max_neighbors, computepred)
fig1=plt.figure(2)
x = range(1, max_neighbors + 1)
plt.subplot(121)
for i in range(num_folds):
    plt.plot(x, r1[i][:], label="validation", color='red')
    plt.plot(x, r2[i][:], label="train", color='blue')
plt.subplot(122)
area = np.pi * 4**2
for i in range(num_folds):
    plt.scatter(100-r1[i][3], 30, s=area, c='red', marker='.')
    plt.scatter(100-r1[i][1], 20, s=area, c='blue', marker='.')
    plt.scatter(100*(m[i]/(X.shape[0]//num_folds)), 10, s=area, c='green', marker='.')

plt.subplot(122)
r1m = np.zeros(max_neighbors)
r2m = np.zeros(max_neighbors)
for i in range(max_neighbors):
    r1m[i] = np.mean(r1[:, i])
    r2m[i] = np.mean(r2[:, i])
plt.plot(x, r1m, label="mean_validation", color='red')
plt.plot(x, r2m, label="mean_train", color='blue')

# axes[0,0].set(title='666')
plt.show()