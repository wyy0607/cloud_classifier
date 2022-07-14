"""
This module is to load cleaned clouds data, acquire features and target from the data,
and save features and target as separate csv files to given outuput path.
"""
import logging.config
from typing import List
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(input_path: str) -> pd.DataFrame:
    """ Load cleaned data

    Args:
        input_path (`str`): path to cleaned data (csv)
    Returns:
        data (:obj:`pandas.DataFrame): pandas dataframe
    """
    # load clouds data
    logger.info("Loading clouds data")

    try:
        data = pd.read_csv(input_path, index_col=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load clouds data. Please try again.")
        sys.exit(1)

    logger.info("Clouds data is successfully loaded")
    return data


def get_features(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """ Get features from data

    Args:
        data (:obj:`pandas.DataFrame`): data to get features from
        columns (:obj:`list` of `str`): list of column names

    Returns:
       features (:obj:`pandas.DataFrame`): dataframe with selected features
    """
    # get features from data
    logger.info("Getting features from clouds data")

    try:
        features = data[columns]
    except KeyError:
        logger.error("At least one of column in provided `columns` "
                     "are not included in provided `data`")
        sys.exit(1)

    logger.info("Features are successfully acquired from data")

    return features


def save_features(features: pd.DataFrame, output_path: str) -> None:
    """Save features to output_path

    Args:
        features (:obj:`pandas.DataFrame`): pandas dataframe
        output_path (`str`): path to save acquired data (csv)

    Returns:
        None
    """
    # Save features to given output path
    logger.info("Saving features to given output path")

    try:
        features.to_csv(output_path, index=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load features. Please try again.")
        sys.exit(1)

    logger.info("Features are successfully saved to given output path")


def get_target(data: pd.DataFrame, column: str, output_path: str) -> None:
    """ Get target from data

        Args:
            data (:obj:`pandas.DataFrame`): data to get features from
            column (`str`): column name for target
            output_path (`str`): path to save acquired data (csv)
        Returns:
            None
    """
    # acquire target from given data
    try:
        target = data[column]
    except KeyError:
        logger.error("Provided `column` is not in provided `data`")
        sys.exit(1)

    # save target to output path
    try:
        target.to_csv(output_path, index=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load target. Please try again.")
        sys.exit(1)
