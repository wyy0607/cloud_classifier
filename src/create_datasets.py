""" This module is to acquire raw data and construct dataset for clouds """
import logging.config
from typing import List
import sys

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def acquire_raw_data(input_path: str, output_path: str) -> None:
    """ Acquire raw data from input path and save it to given output path

        Args:
            input_path (`str`): path to acquire data
            output_path (`str`): path to save acquired data

        Returns:
            None
    """
    logger.info("Acquiring raw data")

    # acquire raw data
    try:
        raw_data = requests.get(input_path)
    except requests.ConnectionError:
        logger.error("There is a connection error.")
        sys.exit(1)
    except (requests.exceptions.MissingSchema, requests.exceptions.InvalidURL):
        logger.error("Either the request schema http/https is missing or the URL is malformed.")
        sys.exit(1)

    logger.info("Raw data is successfully acquired")

    # save data to given output path
    try:
        with open(output_path, "wb") as output_file:
            output_file.write(raw_data.content)
    except FileNotFoundError:
        logger.error("No such file or directory to save raw data. Please try again.")
        sys.exit(1)

    logger.info("Raw data is successfully saved in given output path.")


def get_clouds(input_path: str, columns: List[str], first_cloud: List[int], second_cloud: List[int],
               output_path: str) -> None:
    """ Obtain cloud from raw data

    Args:
        input_path (`str`): path to acquire clouds
        columns (:obj:`list` of `str`): list of column names in clouds
        first_cloud(`list` of `int`): list like [first_index, second_index] to slice first cloud
        second_cloud(`list` of `int`): list like [first_index, second_index] to slice second cloud
        output_path (`str`): path to save acquired data (csv)

    Returns:
       None
    """
    # read raw data
    logger.info("Loading raw data")
    try:
        with open(input_path, mode="r", encoding="ASCII") as input_file:
            data = [[s for s in line.split(" ") if s != ""] for line in input_file.readlines()]
    except FileNotFoundError:
        logger.error("No such file or directory to load raw data. Please try again.")
        sys.exit(1)

    # get first cloud
    logger.info("Getting first cloud")
    try:
        first_cloud = data[first_cloud[0]:first_cloud[1]]
    except TypeError:
        logger.error("Provided first_cloud should be a list of two integers. Please try again.")
        sys.exit(1)

    first_cloud = [[float(s.replace("/n", "")) for s in cloud] for cloud in first_cloud]
    first_cloud = pd.DataFrame(first_cloud, columns=columns)
    first_cloud["class"] = np.zeros(len(first_cloud))  # first cloud is labeled as 0

    # get second cloud
    logger.info("Getting second cloud")
    try:
        second_cloud = data[second_cloud[0]:second_cloud[1]]
    except TypeError:
        logger.error("Provided second_cloud should be a list of two integers. Please try again.")
        sys.exit(1)

    second_cloud = [[float(s.replace("/n", "")) for s in cloud]
                    for cloud in second_cloud]
    second_cloud = pd.DataFrame(second_cloud, columns=columns)
    second_cloud["class"] = np.ones(len(second_cloud))  # second cloud is labeled as 1

    # concatenate the two clouds
    clouds_df = pd.concat([first_cloud, second_cloud])

    # save to output path
    try:
        clouds_df.to_csv(output_path, index=False)
    except FileNotFoundError:
        logger.error("No such file or directory to save cleaned clouds. Please try again.")
        sys.exit(1)

    logger.info("Two clouds are successfully concatenated and saved in given output path.")
