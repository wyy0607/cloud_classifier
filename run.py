"""
Receives command-line arguments from the user
and delegates instructions to the appropriate module in `src/`.
"""
import argparse
import logging.config
import sys

import yaml

from src import create_datasets, process_data, generate_additional_features
from src import train_model, score_model, evaluate_performance

logging.config.fileConfig('config/logging/local.conf')
logger = logging.getLogger('assignment3')

if __name__ == "__main__":
    actions = ["get_raw_data", "get_clouds", "generate_features",
               "train_model", "score_model", "evaluate"]

    parser = argparse.ArgumentParser(description="Model Pipeline for Classifying Clouds")
    parser.add_argument("action",
                        help="action to take",
                        choices=actions)
    parser.add_argument("--config_file",
                        help="path to configuration file",
                        default="config/model_config.yaml")

    args = parser.parse_args()

    # process configuration file
    try:
        with open(args.config_file, "r", encoding="ASCII") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error("Cannot find input configure file")
        sys.exit(1)

    # execute actions
    if args.action == "get_raw_data":
        create_datasets.acquire_raw_data(**config["create_datasets"]["acquire_raw_data"])
    if args.action == "get_clouds":
        create_datasets.get_clouds(**config["create_datasets"]["get_clouds"])
    if args.action == "generate_features":
        data = process_data.load_data(**config["process_data"]["load_data"])
        features = process_data.get_features(data, **config["process_data"]["get_features"])
        process_data.get_target(data, **config["process_data"]["get_target"])
        features = generate_additional_features.log_transform(
            features,
            **config["generate_additional_features"]["log_transform"])
        features = generate_additional_features.multiply(
            features,
            **config["generate_additional_features"]["multiply"])
        features = generate_additional_features.norm_range(
            features,
            **config["generate_additional_features"]["norm_range"])
        process_data.save_features(features, **config["process_data"]["save_features"])
    if args.action == "train_model":
        train_model.split_data(**config["train_model"]["split_data"])
        train_model.fit_model(**config["train_model"]["fit_model"])
    if args.action == "score_model":
        score_model.predict(**config["score_model"]["predict"])
    if args.action == "evaluate":
        evaluate_performance.evaluate(**config["evaluate_performance"]["evaluate"])
    if args.action not in actions:
        parser.print_help()
