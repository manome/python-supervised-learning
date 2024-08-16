# -*- encoding: utf8 -*-

# Author: Nobuhito Manome <manome@g.ecc.u-tokyo.ac.jp>
# License: BSD 3 clause

import os
import pickle
import warnings
import numpy as np
from matplotlib import pyplot as plt

def save_model(model, model_name='model', directory_name='output'):
    '''
    Save the model to a specified directory with a given name.
    ----------
    Parameters
    ----------
    model : model class
        The model to be saved.
    model_name : str, optional (default='model')
        The name to be used for the saved model file.
    directory_name : str, optional (default='output')
        The directory where the model will be saved.
    '''
    # Create the directory if it doesn't exist
    os.makedirs(directory_name, exist_ok=True)
    # Save the model as a pickle file in the specified directory
    with open('{}/{}'.format(directory_name, model_name), 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    '''
    Load a model from the specified path.
    ----------
    Parameters
    ----------
    model_path : str
        The path to the model file to be loaded.
    ----------
    Returns
    ----------
    model : model class
        The loaded model.
    '''
    # Load the model from the pickle file
    with open(model_path, mode='rb') as f:
        model = pickle.load(f)
    return model

def conformal_predict(model, x_calib, y_calib, x_test, confidence_level=0.95, proba_threshold=None, top_k=None, sort_by_proba=True):
    '''
    Generate conformal predictions for each input sample based on the given confidence level.
    ----------
    Parameters
    ----------
    model : object
        The trained model used to make conformal predictions.
    x_calib : array-like, shape = [n_samples, n_features]
        Calibration data.
    y_calib : array, shape = [n_samples]
        True labels for the calibration data.
    x_test : array-like, shape = [n_samples, n_features]
        Test input data for which predictions are to be made.
    confidence_level : float, optional (default=0.95)
        Confidence level for the conformal prediction. It should be between 0 and 1.
    proba_threshold : float, optional (default=None)
        Minimum probability threshold for including a label in the predictions.
    top_k : int, optional (default=None)
        Maximum number of labels to output per test sample.
    sort_by_proba : bool, optional (default=True)
        Whether to sort the output labels by their prediction probabilities.
    ----------
    Returns
    ----------
    conformal_predictions : list of lists
        For each test sample, a list of class indices that meet the conformal prediction criteria.
    '''
    # Calculate the number of samples in the calibration dataset
    n = x_calib.shape[0]
    # Obtain the predicted probabilities for the calibration data
    y_calib_proba = model.predict_proba(x_calib)
    # Calculate the probability of the true class for each calibration sample
    prob_true_class = y_calib_proba[np.arange(n), y_calib]
    # Convert to uncertainty scores by subtracting the true class probabilities from 1
    scores = 1 - prob_true_class
    # Set alpha as the complement of the confidence level
    alpha = 1 - confidence_level
    # Calculate the quantile level for the uncertainty scores
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if not (0 <= q_level <= 1):
        # Apply clipping to ensure q_level is within the range [0, 1]
        warnings.warn(f"Warning: q_level ({q_level}) is out of the range [0, 1]. It will be clipped.")
        q_level = min(max(q_level, 0), 1)
    qhat = np.quantile(scores, q_level, method='higher')
    # Obtain the predicted probabilities for the test data
    y_test_proba = model.predict_proba(x_test)
    # Calculate the uncertainty scores for the test data
    test_scores = 1 - y_test_proba
    # Determine which classes are valid based on the quantile threshold
    valid_classes_matrix = test_scores <= qhat
    # Filter predictions based on the probability threshold, if specified
    if proba_threshold is not None:
        valid_classes_matrix &= (y_test_proba >= proba_threshold)
    # If top_k is specified, limit the number of labels to the top k probabilities
    if top_k is not None:
        top_k_indices = np.argsort(y_test_proba, axis=1)[:, -top_k:]
        top_k_mask = np.zeros_like(y_test_proba, dtype=bool)
        for i, indices in enumerate(top_k_indices):
            top_k_mask[i, indices] = True
        valid_classes_matrix &= top_k_mask
    # Collect the indices of valid classes for each test sample
    conformal_predictions = []
    for i, valid_classes in enumerate(valid_classes_matrix):
        valid_classes_indices = np.where(valid_classes)[0]
        if sort_by_proba:
            # Sort valid classes by their predicted probability
            sorted_classes = sorted(valid_classes_indices, key=lambda x: y_test_proba[i, x], reverse=True)
            conformal_predictions.append(sorted_classes)
        else:
            conformal_predictions.append(valid_classes_indices.tolist())
    return conformal_predictions

def accuracy_score_conformal_predictions(y_test, conformal_predictions):
    '''
    Calculate the accuracy of conformal predictions.
    ----------
    Parameters
    ----------
    y_test : array-like, shape = [n_samples]
        True labels for the test data.
    conformal_predictions : list of lists, length = [n_samples]
        Each entry contains a list of predicted classes for the corresponding sample.
    ----------
    Returns
    ----------
    accuracy : float
        The accuracy of the conformal predictions, i.e., the proportion of test samples
        for which the true label is among the predicted classes.
    '''
    # Convert y_test and conformal_predictions to numpy arrays for easier processing
    y_test = np.asarray(y_test)
    conformal_predictions = np.asarray([set(preds) for preds in conformal_predictions])
    # Check if each true label is among the predicted classes for the corresponding sample
    correct_preds = np.array([y in preds for y, preds in zip(y_test, conformal_predictions)])
    # Calculate the proportion of correct predictions
    accuracy = np.mean(correct_preds)
    return accuracy

def show_conformal_predictions_summary(model, x_calib, y_calib, x_test, y_test, confidence_level=0.95, proba_threshold=None, top_k=None, sort_by_proba=True):
    '''
    Display a summary of conformal predictions made by the provided model.
    ----------
    Parameters
    ----------
    model : object
        The trained model used to make conformal predictions.
    x_calib : array-like, shape = [n_calibration_samples, n_features]
        Feature data used for calibration.
    y_calib : array-like, shape = [n_calibration_samples]
        True labels corresponding to the calibration data.
    x_test : array-like, shape = [n_test_samples, n_features]
        Feature data for which predictions are made.
    y_test : array-like, shape = [n_test_samples]
        True labels for the test data.
    confidence_level : float, optional (default=0.95)
        Confidence level for the conformal prediction. It should be between 0 and 1.
    proba_threshold : float, optional (default=None)
        Minimum probability threshold for including a label in the predictions.
    top_k : int, optional (default=None)
        Maximum number of labels to output per test sample.
    sort_by_proba : bool, optional (default=True)
        Whether to sort the output labels by their prediction probabilities.
    '''
   # Display the confidence level used
    print('** Confidence Level: %.2f' % confidence_level)
    # Generate conformal predictions using the model and provided calibration data
    conformal_predictions = conformal_predict(model, x_calib, y_calib, x_test, confidence_level=confidence_level, proba_threshold=proba_threshold, top_k=top_k, sort_by_proba=sort_by_proba)
    # Calculate and print the accuracy of the conformal predictions
    accuracy = accuracy_score_conformal_predictions(y_test, conformal_predictions)
    print('** Conformal Prediction Accuracy: %.3f' % accuracy)
    # Display the results of the first 10 conformal predictions
    print('** Displaying 10 sample conformal predictions')
    for idx in range(10):
        print('Test{}: True Label: {}, Predicted: {}'.format(idx, y_test[idx], conformal_predictions[idx]))
    # Print a separator for clarity
    print('*********************************************')
