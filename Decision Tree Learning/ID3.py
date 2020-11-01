import numpy as np
import pandas as pd
from math import log

# define dataset
def create_data():
    # given instances of features and target
    instances = np.array([['young', 'no', 'no', 'normal', 'no'],
                          ['young', 'no', 'no', 'good', 'no'],
                          ['young', 'yes', 'no', 'good', 'yes'],
                          ['young', 'yes', 'yes', 'normal', 'yes'],
                          ['young', 'no', 'no', 'normal', 'no'],
                          ['middle', 'no', 'no', 'normal', 'no'],
                          ['middle', 'no', 'no', 'good', 'no'],
                          ['middle', 'yes', 'yes', 'good', 'yes'],
                          ['middle', 'no', 'yes', 'excellent', 'yes'],
                          ['middle', 'no', 'yes', 'excellent', 'yes'],
                          ['old', 'no', 'yes', 'excellent', 'yes'],
                          ['old', 'no', 'yes', 'good', 'yes'],
                          ['old', 'yes', 'no', 'good', 'yes'],
                          ['old', 'yes', 'no', 'excellent', 'yes'],
                          ['old', 'no', 'no', 'normal', 'no']])
    # give feature labels
    labels = np.array(['age', 'work', 'own house', 'credit', 'loan'])
    # set as dataframe
    data = pd.DataFrame(instances, columns=labels)
    return data

X = create_data()


# define a function to compute entropy
def entropy(data, target):
    y = data[target]  # find target variable
    freq = list(y.value_counts())  # count how many values for yes and no
    # compute entropy
    n = np.sum(freq)
    ent = -sum([(f / n) * log(f / n, 2) for f in freq])
    return ent

# define a function to compute information gain
def info_gain(data, feature, target):
    val = data[feature].unique()  # get unique value of feature
    freq = list(data[feature].value_counts())  # count each unique value for that feature
    n = np.sum(freq)  # total instances
    weight = [p / n for p in freq]  # weight for each value in feature
    # compute conditional entropy
    sub_cond_ent = []
    for v in val:
        subdata = data[data[feature] == v]
        sub_cond_ent.append(entropy(subdata, target))
    cond_ent = sum([c * w for c, w in zip(weight, sub_cond_ent)])
    # return information gain
    return entropy(data, target) - cond_ent

# define a function to select best feature split
def bestSplit(data, target):
    attributes = data.columns[data.columns != target]  # get all attributes except target
    infoGain = {}
    # compute information gain
    for a in attributes:
        infoGain[a] = info_gain(data, a, target)
    return max(infoGain, key=infoGain.get)

# define a function to produce a tree
def treeLearning(data, target):
    size = len(data.columns[data.columns != target])
    while size > 0:
        # find the parent node
        parent = bestSplit(data, target)
        leaf = data[parent].unique()
        for v in leaf:
            subdata = data[data[parent] == v]
            if len(subdata[target].unique()) == 1:
                print("Feature:{} AND Value:{} -> {}:{}\n Otherwise:".format(
                    parent, v, target, subdata[target].unique()[0]))
                data = data.drop([parent], axis=1)
        size -= 1
    print("No")

treeLearning(X, 'loan')