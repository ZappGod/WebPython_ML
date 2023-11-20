from MC import app
from MC.MClearn import get_classifier, get_parameters, train_and_test, plot_confusion_matrix
from flask import render_template, request
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    random_state = int(request.form.get("random_state"))
    classifier_name = request.form.get("classifier_name")
    p1 = int(request.form.get("p1")) if "p1" in request.form else None
    p2 = int(request.form.get("p2")) if ("p2" in request.form and request.form.get("p2").strip()) else None

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    classifiers = {
        "KNN": (KNeighborsClassifier, get_parameters("KNN", p1)),
        "SVM": (SVC, get_parameters("SVM", p1)),
        "MLP": (MLPClassifier, get_parameters("MLP", p1)),
        "DT": (DecisionTreeClassifier, get_parameters("DT", p1, p2)),
        "RF": (RandomForestClassifier, get_parameters("RF", p1, p2)),
    }

    class_names = np.unique(y).astype(str)
    accuracy, precision, recall, f1, confusion_matrix = train_and_test(
        get_classifier(classifier_name), classifiers[classifier_name][1], X_train, y_train, X_test, y_test
    )

    confusion_matrix_plot = plot_confusion_matrix(confusion_matrix, class_names)

    return render_template('result.html', classifier_name=classifier_name, accuracy=accuracy,
                           precision=precision, recall=recall, f1=f1, confusion_matrix_plot=confusion_matrix_plot)