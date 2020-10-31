# usage: SMO_Assignment_Dataset.txt

# libraries
from sklearn.model_selection import train_test_split
import numpy as np
import re

# scan a dataset
def createData(file):
    f = open(file)
    data = []
    for line in f:
        L = ' '.join(line.split())
        L = re.sub(' ',', ',L)
        L = re.sub('}',"",L)
        L = L.strip('\n').split(',')
        L = [re.sub(' ','',l) for l in L]
        L = [int(l) for l in L]
        data.append(L)
    data = np.array(data)
    X,y = data[:,0:2],data[:,2]
    return X,y

X,y = createData("SMO_Assignment_Dataset.txt")

# clip conditions
def clipAlpha(a, L, H):
    # set up KKT constrain
    if a < L:
        return L
    elif a > H:
        return H
    else:
        return a


# this function picks all the alpha_js where j!=i
# m is the total number of alpha
def randomPick(i, n):
    l = list(range(n))
    l_delete = np.delete(l, i, 0)
    return np.random.choice(l)


# we know w = a_1y_1x_1+...+a_ny_nx_n = sum(a_i*y_i*x_i)
def computeWeight(X, y, a):
    n, p = X.shape
    w = np.zeros(p)
    for i in range(0, n):
        w = np.add(w, a[i] * y[i] * X[i])
    return w.reshape(-1, 1)


# create a function of g(x)=w^Tx+b, x is linear kernel phi(x)
# X: feature matrix
# x: a instance of feature matrix
# y: labels
# a: lagrange multiplier
# b: offset
def g(X, x, y, a, b):
    xT = x.reshape(-1, 1)
    K = X @ xT
    ay = (a * y).reshape(1, -1)
    gx = ay @ K + b
    return gx[0, 0]

# X: feature matrix
# y: labels
# c: upper bound from KKT
# error: threshold
# maxIter: max iteration
def optimalSMO(X, y, c, error, maxIter):
    n = len(y)
    a = np.zeros(n)
    b = 0
    epoch = 0

    while epoch < maxIter:
        pairs = 0
        for i in range(0, n):
            # define ai xi yi gxi Ei
            ai = a[i]
            xi = X[i]
            yi = y[i]
            gxi = g(X, xi, y, a, b)
            Ei = gxi - yi

            # define optimal pick
            def optimalPick():
                maxJ = -1
                maxE = 0
                for j in range(0, n):
                    if j == i:
                        continue
                    Ej = g(X, X[j], y, a, b) - y[j]
                    e = abs(Ei - Ej)
                    if e > maxE:
                        maxE = e
                        maxJ = j
                return maxJ

            # check KKT conditions, only consider the points not satisfy KKT
            if (yi * Ei < error and ai < c) or (yi * Ei > error and ai > 0):
                # define aj xj yj gxj Ej
                j = optimalPick()
                # j = randomPick(i,n)
                aj = a[j]
                xj = X[j]
                yj = y[j]
                gxj = g(X, xj, y, a, b)
                Ej = gxj - yj

                # define kernel K, where K is PSD and symmetric
                Kii = np.dot(xi, xi)
                Kjj = np.dot(xj, xj)
                Kij = np.dot(xi, xj)

                # define eta
                eta = Kii + Kjj - 2 * Kij

                if eta == 0:
                    print('eta = 0, pass')
                    continue

                # update aj
                ai_old = ai
                aj_old = aj
                aj_new = aj_old + yj * (Ei - Ej) / eta

                # clip aj
                # two cases:
                # 1. ai-aj=k
                if yi != yj:
                    L = max(0, aj_old - ai_old)
                    H = min(c, c + aj_old - ai_old)
                # 2. ai+aj=k
                else:
                    L = max(0, ai_old + aj_old - c)
                    H = min(c, aj_old + ai_old)
                if L == H:
                    print("L = H, pass")
                    continue
                aj_new = clipAlpha(aj_new, L, H)

                # update j=ai
                ai_new = ai_old + yi * yj * (aj_old - aj_new)

                # check if aj not moving
                if abs(aj_new - aj_old) < 0.00001:
                    print('aj does not move enough, pass')
                    continue
                # repeat
                a[i] = ai_new
                a[j] = aj_new

                # update b
                bi = -Ei - yi * Kii * (ai_new - ai_old) - yj * Kij * (aj_new - aj_old) + b
                bj = -Ej - yi * Kij * (ai_new - ai_old) - yj * Kjj * (aj_new - aj_old) + b

                # check constrains: 0<ai<c,0<aj<c
                if 0 < ai_new < c:
                    b = bi
                elif 0 < aj_new < c:
                    b = bj
                else:
                    b = (bi + bj) / 2
                pairs += 1
                print('Epoch: {}, line: {} pair points: {}'.format(epoch, i, pairs))
            if pairs == 0:
                epoch += 1
            else:
                epoch = 0
            print('Epoch number: {}'.format(epoch))
        else:
            print("Satisfy KKT, pass")
    return a, b

# define decision function
def decisionFunction(y):
    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 1
        elif y[i] < 0:
            y[i] = -1
    return y

# define accuracy
def accuracy(y1,y2):
    return np.mean(y1==y2)


def main():
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1995)
    a, b = optimalSMO(X_train, y_train, 0.8, 0.0001, 1000)

    # get prediction
    w = computeWeight(X_train, y_train, a)
    y_pred = (X_test @ w + b).T

    # print result
    print("\n=================Alphas=======================")
    print(a)
    print("\n=================b=======================")
    print("The offset b: {}".format(b))
    print("\n=================True Outcomes=======================")
    print(y_test, "\n")
    print("=================Prediction Outcomes=======================")
    print(decisionFunction(y_pred[0]))
    print("\nPrediction Accuracy: {}".format(accuracy(y_test, decisionFunction(y_pred[0]))))

if __name__ == "__main__":
        main()