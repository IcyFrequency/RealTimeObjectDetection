"""
This code firstly loads images in the form of feature values from a text file, which means that the textfile containg
feature values has to fit the loading of this code perfectly in order to work.
Sci kit learn machine learning methods are used to train a model with the training data.
At last the model is used to make predictions on the test data, which then ouputs the TP TN FP FN

Any changes and modificatations
"""

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import time
import numpy as np
import cPickle
import threading
# np.set_printoptions(threshold=None)
from sklearn.ensemble import RandomForestClassifier


path = '/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/'

#########################Training data#########################
print('Loading training data from txt file...')
fileTrain = open(path + "FeatureTexts/gray_Test_opencv.txt")
dataTrain = np.loadtxt(fileTrain, skiprows=2, usecols=range(2, 502, 1))
print(dataTrain)
print ('Loading Done! \n')

#########################Testing data#########################
print ('Loading test data from txt file...')
fileTest = open(path + "FeatureTexts/gray_Train_opencv.txt")
dataTest = np.loadtxt(fileTest, skiprows=2, usecols=range(2, 502, 1))
print(dataTest)
print ('Loading Done! \n')
# return dataTest


print ('-------------Converting data into train and class array-------------------')
# for the training data
Xtrain = dataTrain[:, 1:-1]  # everything except last column
ytrain = dataTrain[:, -1]  # select column 0 which is the classes decoded as 1 for pos and 0 for negative

# for the test data
Xtest = dataTest[:, 1:-1]  # the features
ytest = dataTest[:, -1]  # the classes
print ('Conversion Done! \n')
"""
names = [
    "MLPClassifier(alpha=1)",
    "MLPClassifier(hidden_layer_sizes=(500,500))",
    "MLPClassifier(hidden_layer_sizes=(500,500), activation='logistic', solver='adam', learning_rate='invscaling')",
    "MLPClassifier(hidden_layer_sizes=(500,500), activation='logistic', solver='sgd', learning_rate='invscaling')",
    "MLPClassifier(hidden_layer_sizes=(1008,500,500,1), solver='sgd', learning_rate='adaptive', momentum=0.4)",
    "MLPClassifier(hidden_layer_sizes=(1008,500,500,1), solver='sgd', learning_rate='adaptive', momentum=0.4)",
    "MLPClassifier(hidden_layer_sizes=(1008,500,500,1), solver='sgd', learning_rate='adaptive', momentum=0.4)",
    "MLPClassifier(hidden_layer_sizes=(1008,500,500,2), activation='logistic', solver='adam', learning_rate='invscaling')",
    "MLPClassifier(hidden_layer_sizes=(1008,500,500,2), activation='logistic', solver='adam', learning_rate='invscaling')",
    "MLPClassifier(hidden_layer_sizes=(1008,500,500,2), activation='logistic', solver='adam', learning_rate='invscaling')",
    "MLPClassifier(hidden_layer_sizes=(1008,500,100,2), solver='sgd', learning_rate='adaptive', momentum=0.4)",
    "MLPClassifier(hidden_layer_sizes=(1008,500,100,2), solver='sgd', learning_rate='adaptive', momentum=0.4)",
    "MLPClassifier(hidden_layer_sizes=(1008,500,100,2), solver='sgd', learning_rate='adaptive', momentum=0.4)"
         ]

classifiers = [
    MLPClassifier(alpha=1),
    MLPClassifier(hidden_layer_sizes=(500,500)),
    MLPClassifier(hidden_layer_sizes=(500,500), activation='logistic', solver='adam', learning_rate='invscaling'),
    MLPClassifier(hidden_layer_sizes=(500,500), activation='logistic', solver='sgd', learning_rate='invscaling'),
    MLPClassifier(hidden_layer_sizes=(1008,500,100,1), solver='sgd', learning_rate='adaptive', momentum=0.4),
    MLPClassifier(hidden_layer_sizes=(1008,500,100,1), solver='sgd', learning_rate='adaptive', momentum=0.4),
    MLPClassifier(hidden_layer_sizes=(1008,500,100,1), solver='sgd', learning_rate='adaptive', momentum=0.4),
    MLPClassifier(hidden_layer_sizes=(1008,500,500,2), activation='logistic', solver='adam', learning_rate='invscaling'),
    MLPClassifier(hidden_layer_sizes=(1008,500,500,2), activation='logistic', solver='adam', learning_rate='invscaling'),
    MLPClassifier(hidden_layer_sizes=(1008,500,500,2), activation='logistic', solver='adam', learning_rate='invscaling'),
    MLPClassifier(hidden_layer_sizes=(1008,500,100,2), solver='sgd', learning_rate='adaptive', momentum=0.4),
    MLPClassifier(hidden_layer_sizes=(1008,500,100,2), solver='sgd', learning_rate='adaptive', momentum=0.4),
    MLPClassifier(hidden_layer_sizes=(1008,500,100,2), solver='sgd', learning_rate='adaptive', momentum=0.4)
]
"""

"""
names = [
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1, leaf_size=10000)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights=distance,algorithm=ball_tree,leaf_size=10000)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights=distance,algorithm=kd_tree,leaf_size=10000)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights=distance,algorithm=brute,leaf_size=10000)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights=distance,algorithm=auto,leaf_size=5000)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights=uniform,algorithm=ball_tree,leaf_size=10000)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights=uniform,algorithm=kd_tree,leaf_size=10000)",
    "KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights=uniform,algorithm=brute,leaf_size=10000)",
    "KNeighborsClassifier(3)"
         ]

classifiers = [
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1, leaf_size=10000),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights="distance",algorithm="ball_tree",leaf_size=10000),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights="distance",algorithm="kd_tree",leaf_size=10000),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights="distance",algorithm="brute",leaf_size=10000),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights="distance",algorithm="auto",leaf_size=5000),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights="uniform",algorithm="ball_tree",leaf_size=10000),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights="uniform",algorithm="kd_tree",leaf_size=10000),
    KNeighborsClassifier(n_neighbors=2,n_jobs=-1,weights="uniform",algorithm="brute",leaf_size=10000),
    KNeighborsClassifier(3)
]
"""

#"""
names = [
    "KNeighborsClassifier",
    #"SVC",
    #"SVC",
    #"GaussianProcessClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    #"MLPClassifier",
    "AdaBoostClassifier",
    "GaussianNB"
    # QuadraticDiscriminantAnalysis()
         ]

classifiers = [
   KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1),
   #SVC(kernel="sigmoid", C=0.025, gamma='auto', coef0=0.0),
   #SVC(gamma=2, C=1),
   #MultiLabelBinarizer(),
   #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
   DecisionTreeClassifier(max_depth=None),
   RandomForestClassifier(n_estimators=1, max_depth=None, random_state=0, max_features='auto', max_leaf_nodes=None,
   n_jobs=-1),
   #MLPClassifier(hidden_layer_sizes=(500,500), activation='logistic', solver='adam'),
   AdaBoostClassifier(base_estimator=None,learning_rate=0.01),
   GaussianNB(),
   #QuadraticDiscriminantAnalysis()
]
#"""
#clf = RandomForestClassifier(n_estimators=200,max_depth=None,random_state=0,max_features='auto',max_leaf_nodes=None,n_jobs=-1).fit(Xtrain,ytrain)


print ('---------------Fit the model to the test data and output the score------------------')

for name, clf in zip(names, classifiers):
    start = time.time()
    print ('Training the classifier...\n %s' % (name))
    clf.fit(Xtrain, ytrain)
    print ('Done! Calculating the accuracy...')
    accuracy = clf.score(Xtest, ytest)
    print ("--Accuracy--")
    print (accuracy)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    pred = np.array([])
    for i in clf.predict(Xtest):
        pred = np.append(pred, i)

    for i in range(len(pred)):
        if ytest[i] == pred[i] == 1:
            TP += 1
        elif pred[i] == 1 and ytest[i] == 0:
            FP += 1
        elif ytest[i] == pred[i] == 0:
            TN += 1
        else:
            FN += 1
    print "TP, TN, FP, FN"
    print (TP, TN, FP, FN)
    print ('\n')


    # Save classifier

    with open(path + 'Models/Classifier_%s.pkl' % (name), 'wb') as clf_GBC:
        cPickle.dump(clf, clf_GBC)

    end = time.time()
    print ("Time: ", end - start)
    print("\n")
print ('Finished! \n')


# print mean_squared_error(ytest, clf.predict(Xtest))
# print clf.predict(Xtest)


"""

# load it again
with open('GradientBoostingClassifier.pkl', 'rb') as clf_GBC:
    gnb_loaded = cPickle.load(clf_GBC)

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(ypredict)):
    if ytest[i] == ypredict[i] == 1:
        TP += 1
for i in range(len(ypredict)):
    if ypredict[i] == 1 and y_test != ypredict[i]:
        FP += 1
for i in range(len(ypredict)):
    if ytest[i] == ypredict[i] == 0:
        TN += 1
for i in range(len(ypredict)):
    if ypredict[i] == 0 and ytest != ypredict[i]:
        FN += 1
print (TP, FP, TN, FN)



iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
print(scores.mean())
"""