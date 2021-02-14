import numpy as np

class SigmoidRegression:
    def __init__(self,eta,error,maxIter,implicitBias):
        # constructor
        # param eta: learning rate
        # param error: error threshold
        # param maxIter: maximum iteration
        # param implicitBias: vectorization offset
        self.eta = eta
        self.error = error
        self.maxIter = maxIter
        self.implicitBias = implicitBias
        return

    def __addIntercept__(self,X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept,X),axis=1)

    def __sigmoid__(self,z):
        return 1/(1+np.exp(-z))

    def __loss__(self,p,y):
        # loss function is same as cross entropy
        return (-y*np.log(p)-(1 - y)*np.log(1-p)).mean()

    def __train__(self,X,y):
        # check if implicitBias
        if self.implicitBias:
            X = self.__addIntercept__(X)

        # define dimension of attribute
        m, d = X.shape

        # initialize weight vector
        self.w = np.zeros(d)

        # training model
        epoch = 0
        JLast = -1
        while epoch < self.maxIter:
            epoch += 1
            # compute sigmoid
            z = np.dot(X,self.w)
            p = self.__sigmoid__(z)
            J = self.__loss__(p,y)
            print("Epoch: {}, Loss: {}".format(epoch,J))
            gradient = np.dot(X.T,(p-y))/m
            self.w -= self.eta*gradient
            if abs(J-JLast) < self.error:
                break
            JLast = J
        return

    def predict_prob(self,X):
        if self.implicitBias:
            X = self.__addIntercept__(X)
        return self.__sigmoid__(np.dot(X,self.w))

    def predict(self,X,threshold):
        y_pred = self.predict_prob(X)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

if __name__ == '__main__':
    # scan data
    f = open("sample.txt")
    data = []
    for line in f:
        L = line.strip("\n").split()
        L = [int(l) for l in L]
        data.append(L)
    data = np.array(data)
    X,y = data[:, 0:2],data[:, 2]
    y[y==-1] = 0

    lr = SigmoidRegression(0.0001,0.0001,1000,True)
    lr.__train__(X,y)
    pred = lr.predict(X,0.5)

    print("Prediction accuracy is {}".format(np.mean(pred==y)))
