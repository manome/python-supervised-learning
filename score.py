# -*- encoding: utf8 -*-

import numpy as np

from sklearn.datasets import load_wine
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
    dataset = load_wine()
    x = dataset.data
    y = dataset.target

    # Standardize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

    # Decision Tree
    model = DecisionTreeClassifier(max_depth=10, criterion='entropy', splitter='best', random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='DecisionTreeClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('DecisionTreeClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # Extra Tree
    model = ExtraTreeClassifier(criterion='gini', max_depth=10, splitter='best', random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='ExtraTreeClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('ExtraTreeClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # Extra Trees
    model = ExtraTreesClassifier(criterion='gini', max_depth=10, n_estimators=100, random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='ExtraTreesClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('ExtraTreesClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # GaussianNB
    model = GaussianNB()
    n_class = np.unique(y_train).shape[0]
    model.class_prior_ = [1 / n_class for _ in range(n_class)]
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='GaussianNB.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('GaussianNB: %.3f' % accuracy_score(y_test, y_predict))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, loss='log_loss', max_depth=10, random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='GradientBoostingClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('GradientBoostingClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # KNeighborsClassifier
    model = KNeighborsClassifier(weights='distance', n_neighbors=10, p=1, algorithm='auto')
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='KNeighborsClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('KNeighborsClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # LinearSVC
    model = LinearSVC(loss='hinge', C=1, multi_class='ovr', penalty='l2', max_iter=1000, random_state=0, dual=True)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='LinearSVC.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('LinearSVC: %.3f' % accuracy_score(y_test, y_predict))

    # LogisticRegression
    model = OneVsRestClassifier(
        LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=1000, random_state=0)
    )
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='LogisticRegression.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('LogisticRegression: %.3f' % accuracy_score(y_test, y_predict))

    # MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='MLPClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('MLPClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # PassiveAggressiveClassifier
    model = PassiveAggressiveClassifier(C=1, max_iter=1000, random_state=0, loss='hinge')
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='PassiveAggressiveClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('PassiveAggressiveClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # Perceptron
    model = Perceptron(penalty='l1', max_iter=1000, random_state=0, eta0=1.0)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='Perceptron.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('Perceptron: %.3f' % accuracy_score(y_test, y_predict))

    # RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=100, random_state=0, bootstrap=True)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='RandomForestClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('RandomForestClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # SGDClassifier
    model = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=0, alpha=0.0001)
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='SGDClassifier.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('SGDClassifier: %.3f' % accuracy_score(y_test, y_predict))

    # SVC
    model = SVC(C=10, kernel='rbf', max_iter=1000, random_state=0, gamma='scale')
    model.fit(x_train, y_train)
    utils.save_model(model, model_name='SVC.pickle', directory_name='output')
    y_predict = model.predict(x_test)
    print('SVC: %.3f' % accuracy_score(y_test, y_predict))

if __name__ == '__main__':
    main()
