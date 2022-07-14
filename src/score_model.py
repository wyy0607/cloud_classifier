""" Predict test data with pre-trained model """
import logging.config
from typing import List
import sys

import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)


def predict(input_path: str, x_test_path: str, initial_features: List[str],
            proba_output_path: str, bin_output_path: str) -> None:
    """ predict test data with given model
    Args:
        input_path (str): input path to pretrained model (joblib)
        x_test_path (`str`): path to test data for features (csv)
        initial_features (:obj:`list` of `str`): list of column names
        proba_output_path (`str`): output path to save predicted probability (csv)
        bin_output_path (`str`): output path to save predicted class (csv)
    Returns:
        None
    """
    # load model
    logger.info("Loading model for prediction")
    try:
        model = joblib.load(input_path)
    except FileNotFoundError:
        logger.error("Cannot find the given model file")
        sys.exit(1)
    # load test data
    logger.info("Loading test data for prediction")
    try:
        x_test = pd.read_csv(x_test_path, index_col=False)
    except FileNotFoundError:
        logger.error("Cannot find the given test data file")
        sys.exit(1)

    # predict
    logger.info("Predicting with given model")
    try:
        ypred_proba_test = model.predict_proba(x_test[initial_features])[:, 1]
        ypred_bin_test = model.predict(x_test[initial_features])
    except KeyError:
        logger.error("Provided `Initial_features` are not all in provided test data or model")
        sys.exit(1)
    except ValueError:
        logger.error("Test sample is not enough. Please try again.")
        sys.exit(1)
    except IndexError:
        logger.error("Index 1 is out of bounds for axis 1 with size 1")
        sys.exit(1)

    # save prediction to given output path
    logger.info("Saving predictions to given output paths")
    try:
        np.savetxt(proba_output_path, ypred_proba_test, delimiter=",")
    except FileNotFoundError:
        logger.error("No such file or directory to save predicted probability. Please try again.")
        sys.exit(1)
    try:
        np.savetxt(bin_output_path, ypred_bin_test, delimiter=",")
    except FileNotFoundError:
        logger.error("No such file or directory to save predicted class. Please try again.")
        sys.exit(1)

    logger.info("Predicted values are successfully saved to given output path")
