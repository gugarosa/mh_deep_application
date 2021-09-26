import numpy as np
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.boolean import BPSO
from opytimizer.spaces import BooleanSpace
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def feature_selection(x):
    # Gathers features
    features = x[:, 0].astype(bool)

    # Loads digits dataset
    digits = load_digits()

    # Gathers samples and targets
    X = digits.data
    Y = digits.target

    # Splits data into training, validation and unused testing sets
    X_train, _, Y_train, _ = train_test_split(X, Y, train_size=0.9, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.5, random_state=0)

    # Remakes training and validation subgraphs with selected features
    X_train_selected = X_train[:, features]
    X_val_selected = X_val[:, features]

    # Creates a Support Vector Classifier
    svm = SVC()

    # Fits training data into the classifier
    svm.fit(X_train_selected, Y_train)

    # Predicts new data
    preds = svm.predict(X_val_selected)

    # Calculates accuracy
    val_acc = accuracy_score(Y_val, preds)

    return 1 - val_acc

# Random seeds for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 3
n_variables = 64

# Creates the space, optimizer and function
space = BooleanSpace(n_agents, n_variables)
optimizer = BPSO()
function = Function(feature_selection)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=True)

# Runs the optimization task
opt.start(n_iterations=100)

# Saves the optimization task
opt.save('finding_best_features.pkl')
