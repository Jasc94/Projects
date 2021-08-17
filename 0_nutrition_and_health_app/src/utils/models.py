import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from imblearn.over_sampling import SMOTE

import sys, os

# Helpers
abspath = os.path.abspath
dirname = os.path.dirname
sep = os.sep

################# ML TRAINING #################
#########
class ml_model:
    """A class that will store a machine learning model and all its metrics and relevant information. It has as well some useful methods.
    """
    #########
    def __init__(self, model):
        # Data to train the model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.kfold = None

        # Training metrics
        self.model = model
        self.train_scores = []
        self.val_scores = []
        self.train_set_structures = []
        self.val_set_structures = []
        self.feature_importances = None

        # Test metrics
        self.train_score = None
        self.test_score = None
        self.train_structure = None
        self.test_structure = None
        self.prediction = None
        self.cm = None        

        # Confusion matrix metrics
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.precisions = None
        self.recalls = None
        self.thresholds = None

    #########
    def load_data(self, X_train, X_test, y_train, y_test, features, kfold):
        """It loads the data to train the model

        Args:
            X_train (numpy.array): X train set
            X_test (numpy.array): X test set
            y_train (numpy.array): y train set
            y_test (numpy.array): y test set
            features (list): List with the descriptions of all the features
            kfold (object): Sklearn object with the cross validation folds
        """

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.kfold = kfold

    #########
    def ml_trainer(self, verb = False):
        """To train the model

        Args:
            verb (bool, optional): If True, it will print some information about the training process. Defaults to False.
        """
        count = 1

        # Iterate over all the folds of the cross validation
        for (train, val) in self.kfold.split(self.X_train, self.y_train):
            # Train-Validation sets
            x_t, y_t = self.X_train[train], self.y_train[train]
            x_v, y_v = self.X_train[val], self.y_train[val]

            # Internal structure of the data
            y_t_unique, y_t_counts = np.unique(y_t, return_counts=True)
            y_v_unique, y_v_counts = np.unique(y_v, return_counts=True)

            # Store the data about the structure in the object as a list of dicts
            self.train_set_structures.append(dict(zip(y_t_unique, y_t_counts / len(y_t))))
            self.val_set_structures.append(dict(zip(y_v_unique, y_v_counts / len(y_v))))

            # Training
            self.model.fit(x_t, y_t)

            # Scores
            train_score = self.model.score(x_t, y_t)
            val_score = self.model.score(x_v, y_v)

            # Store training and validation scores as a list
            self.train_scores.append(train_score)
            self.val_scores.append(val_score)

            # Information to show in case verb == True
            if verb:
                print(f"\n-- Model {count} --")
                print("-" * 25)
                print(">train score:", train_score)
                print(">test score:", val_score)
                print("-" * 25)
                print("Set structure:")
                print("Train structure:", dict(zip(y_t_unique, y_t_counts / len(y_t))))
                print("Validation structure:", dict(zip(y_v_unique, y_v_counts / len(y_v))))
                print("#" * 75)

            # To count the fold
            count += 1

        # Not all the models have an attribute for the importance of the features
        # Hence, try to get the data
        try:
            importances = self.model.feature_importances_
            feature_importances = list(zip(self.features, importances))

            self.feature_importances = pd.DataFrame(feature_importances, columns = ["features", "importance"]).sort_values(by = "importance", ascending = False)
        
        # If the model doesn't have it, go on to the next round
        except:
            pass

    #########
    def ml_tester(self, verb = False):
        """This method trains the model with the full training data and tests it with the test data. Ideally, it should be applied after cross validation.

        Args:
            verb (bool, optional): If True, it will print some information about the training process. Defaults to False.
        """
        # Internal structure
        y_train_unique, y_train_counts = np.unique(self.y_train, return_counts=True)
        y_test_unique, y_test_counts = np.unique(self.y_test, return_counts=True)

        # Store the data about the structure in the object as a list of dicts
        self.train_structure = dict(zip(y_train_unique, y_train_counts / len(self.y_train) * 100))
        self.test_structure = dict(zip(y_test_unique, y_test_counts / len(self.y_test) * 100))

        # Scores
        self.train_score = self.model.score(self.X_train, self.y_train)
        self.test_score = self.model.score(self.X_test, self.y_test)

        # Prediction
        self.prediction = self.model.predict(self.X_test)

        # Confusion matrix
        self.cm = metrics.confusion_matrix(self.y_test, self.prediction)

        ##### Precision metrics
        self.accuracy = (self.cm[0][0] + self.cm[1][1]) / self.cm.sum()
        self.precision = self.cm[1][1] / (self.cm[1][1] + self.cm[0][1])
        self.recall = self.cm[1][1] / (self.cm[1][1] + self.cm[1][0])
        self.f1_score = 2 * ((self.precision * self.recall) / (self.precision + self.recall))

        # Information to show in case verb == True
        if verb:
            print("Train structure:", self.train_structure)
            print("Test structure:", self.test_structure)
            print("#" * 75)
            print(">Train score:", self.train_score)
            print(">Test score:", self.test_score)
            print("#" * 75)
            print("Confusion matrix")
            print(self.cm)
            print("#" * 75)
            print("Precision metrics")
            print("Accuracy:", self.accuracy)
            print("Precision:", self.precision)
            print("Recall:", self.recall)
            print("F1 score:", self.f1_score)

    #########
    def ready_to_use(self, X, y, verb = False):
        """This method trains the model with the full data and then tests it. Ideally, it should be applied after testing the model.

        Args:
            X ([type]): [description]
            y ([type]): [description]
            verb (bool, optional): [description]. Defaults to False.
        """
        # Training
        self.model.fit(X, y)

        # Score
        score = self.model.score(X, y)
        # Predictions
        predictions = self.model.predict(X)
        # Confusion matrix
        cm = metrics.confusion_matrix(y, predictions)

        # Metrics derived from the confusion matrix
        accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        f1_score = 2 * ((precision * recall) / (precision + recall))

        # Information to show in case verb == True
        if verb:
            print("Model results - Full data")
            print(">Score:", score)
            print("#" * 75)
            print("Confusion matrix")
            print(cm)
            print("#" * 75)
            print("Precision metrics")
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 score:", f1_score)

    #########
    def ml_predictions(self, to_predict):
        """Method to predict using the trained model.

        Args:
            to_predict (np.array): X data to predict the target variable

        Returns:
            np.array: Prediction of the target value
        """
        # Predictions
        new_predictions = self.model.predict(to_predict)
        return new_predictions

    #########
    def model_saver(self, path):
        """It saves the model

        Args:
            path (str): Path to save the model

        Returns:
            str: Sucess/error message
        """
        try:
            # Save the model as pikle file
            joblib.dump(self.model, path + ".pkl")
            return "Succesfully saved"

        except:
            return "Something went wrong. Please check all the settings"

#########
class model_ensembler:
    """Class to work with several models at the same time and train/test them together.
    It is built on top of ml_model class and allows us to compare the models.
    """
    def __init__(self, models):
        """Constructor
        """
        # Models
        self.models = models
        self.model_names = [str(model) for model in models]
        self.ml_models = [ml_model(model) for model in models]

        # Data to train the models
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.kfold = None

        # Models' metrics
        self.metrics = None

    #########
    def load_data(self, X_train, X_test, y_train, y_test, features, kfold):
        """It loads the data to train the models

        Args:
            X_train (numpy.array): X train set
            X_test (numpy.array): X test set
            y_train (numpy.array): y train set
            y_test (numpy.array): y test set
            features (list): List with the descriptions of all the features
            kfold (object): Sklearn object with the cross validation folds
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.kfold = kfold

    #########
    def models_tester(self):
        """This method trains the models with the full training data and tests it with the test data.
        """
        # Metrics' names
        metric_names = ["Test_score", "Train_score", "Test_score_drop", "Accuracy", "Precision", "Recall", "F1_score", "Confusion_matrix"]
        # Empty lists to save the progression of the different metrics
        test_scores = []
        train_scores = []
        test_score_drops = []
        cms = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        # A list of all the metrics' lists
        metrics_lists = [test_scores, train_scores, test_score_drops, accuracies, precisions, recalls, f1_scores, cms]

        # Loop through all the models and train/test them and save the relevant metrics
        for model in self.ml_models:
            # Load data
            model.load_data(self.X_train, self.X_test, self.y_train, self.y_test, self.features, self.kfold)
            # Train model
            model.ml_trainer()
            # Test model
            model.ml_tester()
            # Save model metrics in the empty lists
            test_scores.append(model.test_score)
            train_scores.append(model.train_score)
            test_score_drops.append((model.test_score - model.train_score) / model.train_score)
            accuracies.append(model.accuracy)
            precisions.append(model.precision)
            recalls.append(model.recall)
            f1_scores.append(model.f1_score)
            cms.append(model.cm)

        # Stores all the metrics as a dataframe
        self.metrics = pd.DataFrame(metrics_lists, index = metric_names, columns = self.model_names).T
        self.metrics = self.metrics.sort_values(by = "Test_score", ascending = False)

    def models_saver(self, path_to_folder):
        """It saves the models' metrics and the models too.

        Args:
            path (str): Path to save the model

        Returns:
            str: Sucess/error message
        """
        try:
            # Create the folder if it doesn't exist
            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)
            # Dump all the models in there
            for ind, model in enumerate(self.models):
                model_name = self.model_names[ind]
                joblib.dump(model, path_to_folder + sep + model_name + ".pkl")

            return "Succesfully saved"

        except:
            return "Something went wrong. Please check all the settings"