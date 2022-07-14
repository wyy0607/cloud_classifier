""" This module is to train a random forest classifier to cloud data """
import logging.config
from typing import List
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)


def split_data(feature_path: str, target_path: str, test_size: float, random_state: int,
               x_train_path: str, x_test_path: str, y_train_path: str, y_test_path: str) -> None:
    """ Split features and target into train and test set

    Args:
        feature_path (`str`):  path to features (csv)
        target_path (`str`): path to target (csv)
        test_size (`float`): proportion of test set
        random_state (`int`):  pass an int for reproducible output
        x_train_path (`str`): path to training data for features (csv)
        x_test_path (`str`): path to test data for features (csv)
        y_train_path (`str`): path to training data for target (csv)
        y_test_path (`str`): path to training data for target (csv)
    Returns:
         None
    """
    logger.info("Loading features and target for splitting train and test set")

    # handle exception for file not found
    try:
        # load features
        features = pd.read_csv(feature_path, index_col=False)
    except FileNotFoundError:
        logger.error("Cannot find provided feature file")
        sys.exit(1)
    try:
        # load target
        target = pd.read_csv(target_path, index_col=False)
    except FileNotFoundError:
        logger.error("Cannot find provided target file")
        sys.exit(1)

    # handle exceptions for not enough samples
    try:
        # split train and test
        logger.info("Splitting train and test sets")
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size,
                                                            random_state=random_state)
    except ValueError:
        logger.error("Sample size in provided data is not enough. Please add more samples.")
        sys.exit(1)

    # handle exceptions for file not found
    try:
        # save x train
        x_train.to_csv(x_train_path, index=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load x train. Please try again.")
        sys.exit(1)
    try:
        # save x test
        x_test.to_csv(x_test_path, index=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load x test. Please try again.")
        sys.exit(1)
    try:
        # save y train
        y_train.to_csv(y_train_path, index=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load y train. Please try again.")
        sys.exit(1)
    try:
        # save y test
        y_test.to_csv(y_test_path, index=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load y test. Please try again.")
        sys.exit(1)

    logger.info("Train and test sets are successfully split and saved to given output paths")


def fit_model(x_train_path: str, y_train_path: str, initial_features: List[str],
              n_estimators: int, max_depth: int, random_state: int, output_path: str) -> None:
    """ Train a random forest model

    Args:
        x_train_path (`str`): path to training data for features (csv)
        y_train_path (`str`): path to training data for target (csv)
        initial_features (:obj:`list` of `str`): list of column names
        n_estimators (`int`): the number of trees in the forest
        max_depth (`int`): the maximum depth of the tree
`       random_state (`int`):  pass an int for reproducible output
        output_path (`str`): path to save acquired data (joblib)

    Returns:
        None
    """
    # load training set for fit model
    logger.info("Loading training set for model fitting")

    # handle exceptions for training data for features
    try:
        x_train = pd.read_csv(x_train_path, index_col=False)
    except FileNotFoundError:
        logger.error("Cannot find provided training data for features. Please try again.")
        sys.exit(1)

    # handle exceptions for training data for target
    try:
        y_train = pd.read_csv(y_train_path, index_col=False)
    except FileNotFoundError:
        logger.error("Cannot find provided training data for target. Please try again.")
        sys.exit(1)

    # fit random forest classifier
    logger.info("Fitting random forest classifier")

    if not all(isinstance(item, int) for item in [n_estimators, max_depth, random_state]):
        logger.error("All of n_estimators, max_depth, or random_state have to be an integer."
                     "Please try again.")
        raise ValueError("All of n_estimators, max_depth, or random_state have to be an integer.")

    if not all(item > 0 for item in [n_estimators, max_depth, random_state]):
        logger.error("All of n_estimators, max_depth, or random_state have to be greater than 0"
                     "Please try again.")
        raise ValueError("All of n_estimators, max_depth, or random_state have tbe greater than 0")

    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=random_state)

    try:
        rf_model.fit(x_train[initial_features], y_train.values.ravel())
    except KeyError:
        logger.error("Provided `Initial_features` are not all in provided `x_train`")
        sys.exit(1)
    except ValueError:
        logger.error("Sample sizes in provided training data are not enough or inconsistent")
        sys.exit(1)

    logger.info("Classifier is successfully fitted")

    # save model fit to given output path
    try:
        logger.info("Saving the fitted classifier to given output path")
        joblib.dump(rf_model, output_path)
    except FileNotFoundError:
        logger.error("No such file or directory to save model. Please try again.")
        sys.exit(1)

    logger.info("Fitted classifier is successfully saved to given output path")
