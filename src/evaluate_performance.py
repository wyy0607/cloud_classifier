""" Compute test metrics to evaluate model performance """
import logging.config
import sys

import sklearn
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate(proba_input_path: str, bin_input_path: str,
             y_test_path: str, output_path: str) -> None:
    """ Calculate accuracy metrics and save it to output path

    Args:
        proba_input_path (`str`): path to saved predicted probability for test data (npy)
        bin_input_path (`str`): path to saved predicted class for test data (npy)
        y_test_path (`str`): path to saved test data for target (csv)
        output_path (`str`): path to save calculated metrics (txt)

    Returns:
        None
    """
    # load necessary data for evaluation
    logger.info("Loading predicted values for evaluation")
    try:
        ypred_proba_test = pd.read_csv(proba_input_path, index_col=False, header=None)
    except FileNotFoundError:
        logger.error("No such file or directory to load predicted probability. Please try again.")
        sys.exit(1)

    try:
        ypred_bin_test = pd.read_csv(bin_input_path, index_col=False, header=None)
    except FileNotFoundError:
        logger.error("No such file or directory to load predicted class. Please try again.")
        sys.exit(1)

    logger.info("Loading actual test target for evaluation")
    try:
        y_test = pd.read_csv(y_test_path, index_col=False)
    except FileNotFoundError:
        logger.error("No such file or directory to load actual test target. Please try again.")
        sys.exit(1)

    # compute test metrics
    logger.info("Calculating test metrics")

    # check if only two class are present in test data
    if y_test.nunique().values != 2:
        logger.error("Exactly two classes should be present in y_test")
        raise ValueError("Exactly two classes should be present in y_test")

    try:
        # compute test metrics with predicted probability
        auc = sklearn.metrics.roc_auc_score(y_test, ypred_proba_test)

    except ValueError:
        logger.error("Samples should be of same length (>1) and all numerical")
        sys.exit(1)

    try:
        # compute test metrics with predicted class
        confusion = sklearn.metrics.confusion_matrix(y_test, ypred_bin_test)
        accuracy = sklearn.metrics.accuracy_score(y_test, ypred_bin_test)
        classification_report = sklearn.metrics.classification_report(y_test, ypred_bin_test)
    except ValueError:
        logger.error("Samples should be of same length (>1) and all binary")
        sys.exit(1)

    confusion_df = pd.DataFrame(confusion, index=["Actual negative", "Actual positive"],
                                columns=["Predicted negative", "Predicted positive"])
    # transform confusion df to string for writing the output later
    confusion_df_string = confusion_df.to_string(header=True, index=True)

    # save output to given path
    logger.info("Saving test metrics to given output path")

    try:
        with open(output_path, "w", encoding="ASCII") as output_file:
            output_file.write(f"AUC on test: {auc:.3f}\n")
            output_file.write(f"Accuracy on test: {accuracy:.3f}\n")
            output_file.write(confusion_df_string)
            output_file.write("\n")
            output_file.writelines(classification_report)
        output_file.close()
    except FileNotFoundError:
        logger.error("No such file or directory to save test metrics. Please try again.")
        sys.exit(1)

    logger.info("Test metrics are successfully saved to given output path")
