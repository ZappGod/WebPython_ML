import numpy as np
from flask import g
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def get_classifier(classifier_name):
    if classifier_name == "KNN":
        return KNeighborsClassifier()
    elif classifier_name == "SVM":
        return SVC()
    elif classifier_name == "MLP":
        return MLPClassifier()
    elif classifier_name == "DT":
        return DecisionTreeClassifier()
    elif classifier_name == "RF":
        return RandomForestClassifier()

def get_parameters(classifier_name,p1,p2 = None):
    if classifier_name == "KNN":
        return {"n_neighbors": p1}
    elif classifier_name == "SVM":
        return {"C": p1}
    elif classifier_name == "MLP":
        return {"hidden_layer_sizes": p1}
    elif classifier_name == "DT":
        return {"max_depth": p1,"min_samples_split": p2}
    elif classifier_name == "RF":
        return {"n_estimators": p1, "max_depth": p2}

def train_and_test(classifier, parameters, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1, confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(confusion_matrix, class_names):
    if not hasattr(g, 'graph'):
        g.graph = plt.figure()

    fig = g.graph

    fig.clf()

    plt.imshow(confusion_matrix,cmap='viridis')
    plt.xlabel("Classe real")
    plt.ylabel("Classe predita")
    plt.title("Matriz de confusão")
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

    image_stream = BytesIO()
    fig.savefig(image_stream, format="png")
    image_stream.seek(0)

    image_base64 = base64.b64encode(image_stream.read()).decode("utf-8")

    return image_base64