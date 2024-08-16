# -*- encoding: utf8 -*-

import numpy as np

from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB

import utils as utils

def main():
    # Load dataset
    dataset = load_digits()
    x = dataset.data
    y = dataset.target

    # Standardize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
    # Split training dataset into training set and calibration set
    x_train, x_calib, y_train, y_calib = train_test_split(x_train, y_train, test_size=0.2, random_state=8, shuffle=True, stratify=y_train)

    # Extra Trees
    model = ExtraTreesClassifier(criterion='gini', max_depth=10, n_estimators=100, random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='ExtraTreesClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('ExtraTreesClassifier: %.3f' % accuracy_score(y_test, y_predict))
    utils.show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.98)

    # GaussianNB
    model = GaussianNB()
    n_class = np.unique(y_train).shape[0]
    model.class_prior_ = [1 / n_class for _ in range(n_class)]
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='GaussianNB.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('GaussianNB: %.3f' % accuracy_score(y_test, y_predict))
    utils.show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.98)

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, loss='log_loss', max_depth=10, random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='GradientBoostingClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('GradientBoostingClassifier: %.3f' % accuracy_score(y_test, y_predict))
    utils.show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.98)

    # LogisticRegression
    model = OneVsRestClassifier(
        LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=1000, random_state=0)
    )
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='LogisticRegression.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('LogisticRegression: %.3f' % accuracy_score(y_test, y_predict))
    utils.show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.98)

    # MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='MLPClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('MLPClassifier: %.3f' % accuracy_score(y_test, y_predict))
    utils.show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.98)

    # RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=100, random_state=0, bootstrap=True)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='RandomForestClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('RandomForestClassifier: %.3f' % accuracy_score(y_test, y_predict))
    utils.show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.98)

    # SVC
    model = SVC(C=10, kernel='rbf', max_iter=1000, random_state=0, gamma='scale', probability=True)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='SVC.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('SVC: %.3f' % accuracy_score(y_test, y_predict))
    utils.show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.98)

if __name__ == '__main__':
    main()
