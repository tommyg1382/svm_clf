from __future__ import print_function

import numpy as np

#import matplotlib
#import matplotlib.pyplot as plt
import plotSVMBoundaries as psvmb
from sklearn.svm import SVC

C = 1

X = np.genfromtxt('Data/HW10_1_csv/train_x.csv', delimiter=',')
y = np.genfromtxt('Data/HW10_1_csv/train_y.csv', delimiter=',')
clf = SVC(kernel='linear', C=C)#, gamma=5000)
clf.fit(X,y)
print('Accuracy:')
print(clf.score(X,y))

psvmb.plotSVMBoundaries(X,y,clf) #clf.support_vectors_)
#print('done')
#print("Support Vectors: ")
#print(clf.support_vectors_)

