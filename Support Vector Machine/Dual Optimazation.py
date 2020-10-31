#Task: The program is to predict non-versicolor v.s versicolor, so the target would be #imbalanced.
#Note: For model assessment, we use 75% data as training set and 25% as testing set.
from numpy import *
import pandas as pd
from sklearn.datasets import load_iris

# function for creating a dataset
def dataset():
    # load data from iris package
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    label = iris.target

    # put label into data
    iris_df['label'] = label

    # set dataframe into matrix
    df = array(iris_df)

    # define feature and target
    feature = df[:,:-2]
    target = df[:,-1]

    # insert intercept into feature
    feature = insert(feature,0,1,axis=1)
    target[target == 0] = -1

    return feature[0:100,:], target[0:100]

# create X and y
X,y = dataset()
n = X.shape[0]
a = zeros(n)


# define a function for gradient of L_dual
# i is the ith gradient
def gradient(a,n,i):
    total = 0 # sum of total
    for j in range(0,n):
        total += y[i]*a[j]*y[j]*matmul(X[i].T,X[j])
    return 1-total
gradient(a,n,10)

# define a function to solve a
# c is a maxium range from KKT
# eta is the learning rate
def gradientDescent(c,maxIters):
    for t in range(0,maxIters):
        for i in range(0,n):
            if 0 <= a[i] <= c:
                a[i] += 0.01*gradient(a,n,i)
    return a
a_hat = gradientDescent(1,1000)


w = (X.T@(a_hat*y)).reshape(-1,1)
y_pred = X@w
for i in range(0,len(y_pred)):
    if y_pred[i] > 0:
        y_pred[i] = 1
    else:
        y_pred[i] = -1

print(squeeze(y_pred) == y)