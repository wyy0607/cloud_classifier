""" This module is to generate additional features """
import logging.config

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def log_transform(features: pd.DataFrame, log_col: str, additional_feature: str) -> pd.DataFrame:
    """ Take log of selected features

        Args:
            features (:obj:`pandas.DataFrame`): data that include columns to transform
            log_col (`str`): column name for the feature to take log transform
            additional_feature (`str`): column name for the additional feature

        Returns:
           features (:obj:`pandas.DataFrame`): dataframe with transformed feature
    """
    # raise TypeError if provided data is not a pandas dataframe
    if not isinstance(features, pd.DataFrame):
        logger.error("Provided argument `features` is not a Pandas DataFrame object")
        raise TypeError("Provided argument `features` is not a Pandas DataFrame object")

    # raise AttributeError if provided data does not contain the column to be transformed
    if log_col not in features.columns:
        logger.error("%s not in provided data", log_col)
        raise AttributeError(f"{log_col} not in provided data")

    features[additional_feature] = features[log_col].apply(np.log)
    logger.info("Log transformed feature is successfully generated")

    return features


def multiply(features: pd.DataFrame, col1: str, col2: str, additional_feature: str) -> pd.DataFrame:
    """Get feature equal to the product of two given features

    Args:
        features (:obj:`pandas.DataFrame`): data that include columns to multiply
        col1 (`str`): column name of the first feature
        col2: (`str`): column name of the second feature
        additional_feature (`str`): column name for the additional feature
    Returns:
        features (:obj:`pandas.DataFrame`): dataframe with transformed features
    """
    # raise TypeError if provided data is not a pandas dataframe
    if not isinstance(features, pd.DataFrame):
        logger.error("Provided argument `features` is not a Pandas DataFrame object")
        raise TypeError("Provided argument `features` is not a Pandas DataFrame object")

    # raise AttributeError if provided data does not contain the column to be transformed
    if col1 not in features.columns:
        logger.error("%s not in provided data", col1)
        raise AttributeError(f"{col1} not in provided data")
    if col2 not in features.columns:
        logger.error("%s not in provided data", col2)
        raise AttributeError(f"{col2} not in provided data")

    # features multiplication
    features[additional_feature] = features[col2].multiply(features[col1])
    logger.info("Multiplied feature is successfully generated")

    return features


def col_range(features: pd.DataFrame, min_col: str, max_col: str,
              additional_feature: str) -> pd.DataFrame:
    """ Get range as a feature

    Args:
        features (:obj:`pandas.DataFrame`): data that include IR max, min, mean
        min_col (`str`): column name for column containing information for min
        max_col (`str`): column name for column containing information for max
        additional_feature (`str`): column name for the additional feature
    Returns:
        features (:obj:`pandas.DataFrame`): dataframe with transformed features
    """
    # raise TypeError if provided data is not a pandas dataframe
    if not isinstance(features, pd.DataFrame):
        logger.error("Provided argument `features` is not a Pandas DataFrame object")
        raise TypeError("Provided argument `features` is not a Pandas DataFrame object")

    # raise AttributeError if provided data does not contain the column to be transformed
    if min_col not in features.columns:
        logger.error("%s not in provided data", min_col)
        raise AttributeError(f"{min_col} not in provided data")
    if max_col not in features.columns:
        logger.error("%s not in provided data", max_col)
        raise AttributeError(f"{max_col} not in provided data")

    # range
    features[additional_feature] = features[max_col] - features[min_col]
    logger.info("Range feature is successfully generated")

    return features


def norm_range(features: pd.DataFrame, min_col: str, max_col: str,
               mean_col: str, additional_feature: str) -> pd.DataFrame:
    """ Get normalized range as a feature

    Args:
        features (:obj:`pandas.DataFrame`): data that include IR max, min, mean
        min_col (`str`): column name for column containing information for min
        max_col (`str`): column name for column containing information for max
        mean_col (`str`): column name for column containing information for mean
        additional_feature (`str`): column name for the additional feature
    Returns:
        features (:obj:`pandas.DataFrame`): dataframe with transformed features
    """
    # raise TypeError if provided data is not a pandas dataframe
    if not isinstance(features, pd.DataFrame):
        logger.error("Provided argument `features` is not a Pandas DataFrame object")
        raise TypeError("Provided argument `features` is not a Pandas DataFrame object")

    # raise AttributeError if provided data does not contain the column to be transformed
    if min_col not in features.columns:
        logger.error("%s not in provided data", min_col)
        raise AttributeError(f"{min_col} not in provided data")
    if max_col not in features.columns:
        logger.error("%s not in provided data", max_col)
        raise AttributeError(f"{max_col} not in provided data")
    if mean_col not in features.columns:
        logger.error("%s not in provided data", mean_col)
        raise AttributeError(f"{mean_col} not in provided data")

    # Normalized range
    features[additional_feature] = (features[max_col] - features[min_col])\
        .divide(features[mean_col])
    logger.info("Normalized range feature is successfully generated")

    return features
