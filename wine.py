from __future__ import print_function

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import plotSVMBoundaries as psvmb
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

C = 1
numParam = 50
numRuns = 20


numFeat = 2 #13 or 2
numClass = 3
wineTrain = np.genfromtxt('Data/wine_csv/feature_train.csv', delimiter=',')
wineTest = np.genfromtxt('Data/wine_csv/feature_test.csv', delimiter=',')
X = wineTrain[:,0:numFeat]
feat_test = wineTest[:,0:numFeat]
y = np.genfromtxt('Data/wine_csv/label_train.csv', delimiter=',')
label_test = np.genfromtxt('Data/wine_csv/label_test.csv', delimiter=',')

#clf.fit(X,y)

N = np.logspace(-3, 3, numParam)
G = np.logspace(-3,3, numParam)

Acc = np.zeros((numParam, numParam))
Dev = np.zeros((numParam, numParam))

RunAcc = np.zeros((numRuns,1))
RunDev = np.zeros((numRuns,1))

for GammaNdx in reversed(range(len(G))):
    for Cndx in range(len(N)):
        for run in range(numRuns):
        
            score = np.zeros((5,1))
            ndx = 0
            clf = SVC(kernel='rbf', gamma=G[GammaNdx], C=N[Cndx])

            skf = StratifiedKFold(n_splits=5, shuffle=True)
        

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train,y_train)
                #print("Score ", clf.score(X_test,y_test))
                score[ndx] = clf.score(X_test,y_test)
                ndx = ndx + 1
        #print("Mean: ", np.mean(score))
        #print("Dev: ", np.std(score))
            RunAcc[run] = np.mean(score)
            RunDev[run] = np.std(score)
        Acc[Cndx, GammaNdx] = np.mean(RunAcc)        
        Dev[Cndx, GammaNdx] = np.mean(RunDev)

MaxLoc = np.argmax(Acc)
MaxLocTuple = np.unravel_index(MaxLoc, Acc.shape)
print("Max Accuracy Location: ", MaxLoc)
print("Max Val: ", Acc[MaxLocTuple])
print("Standard Dev: ", Dev[MaxLocTuple])
print("GAMMA: ", G[MaxLocTuple[1]])
print("C: ", N[MaxLocTuple[0]])



plt.figure()
plt.imshow(Acc)
plt.colorbar()
plt.xlabel('Gamma Index')
plt.ylabel('C Index')
plt.savefig('Acc20.png')
plt.figure()
plt.imshow(Dev)
plt.colorbar()
plt.xlabel('Gamma Index')
plt.ylabel('C Index')
plt.savefig('Dev20.png')


#psvmb.plotSVMBoundaries(X,y,clf) #clf.support_vectors_)
#print('done')
#print("Support Vectors: ")
#print(clf.support_vectors_)
