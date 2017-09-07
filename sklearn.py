from __future__ import division
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

def split_label(data, num_class):
    np.random.shuffle(data)
    X_data = np.reshape(data[:,0:-1],(-1,data.shape[1]-1))
    y_data = np.reshape(data[:, -1:], (data.shape[0]))
    return X_data, y_data

def load_data(data_path):
    phr_feature = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
    X_data, y_data = split_label(phr_feature, 2)
    return X_data, y_data

def calc_senspe(y_data, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_data, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

data_dir = './ae_padding/30_20_10'
data_name = 'earlystopping_ep4437_l4968_lr5'
data_path = data_dir+'/'+data_name+'.csv'
X_data, y_data = load_data(data_path)
print('dataset: {}'.format(data_name))
# X_data, y_data = load_data('./data/mean_padding.csv')
clf_list = ['SVM_linear', 'SVM_rbf', 'DecisionTree', 'RandomForest', 'AdaBoost', 'NaiveBayes', 'MLP', 'GradientBoosting']
clfs = [svm.SVC(C=1, kernel='linear'), svm.SVC(C=1, kernel='rbf'),
               DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10),
               AdaBoostClassifier(n_estimators=10), GaussianNB(), MLPClassifier(),
               GradientBoostingClassifier(random_state=0, learning_rate=0.15)]
with warnings.catch_warnings():
    warnings.simplefilter('ignore', ConvergenceWarning)
    for i in range(len(clfs)):
        train_acc = cross_val_score(clfs[i], X_data, y_data, cv=5)
        y_pred = cross_val_predict(clfs[i], X_data, y_data, cv=5)
        test_acc = accuracy_score(y_data, y_pred)
        sen, spe = calc_senspe(y_data, y_pred)
        print('classifier: {}; train_acc: {:.4f};'
              '\ntest_acc: {:.4f}; sen: {:.4f}; spe: {:.4f}'.format(clf_list[i], np.mean(train_acc), test_acc, sen, spe))