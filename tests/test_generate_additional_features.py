"""
This module is to test functions to generate additional features.
It includes tests for log transformation, multiplication of two features,
and computation of normalized range.
"""

import pandas as pd
import pytest

from src import generate_additional_features

data = pd.read_csv("data/interim/clouds.csv", index_col=False)
columns = ["visible_mean", "visible_max", "visible_min", "visible_mean_distribution",
           "visible_contrast", "visible_entropy", "visible_second_angular_momentum",
           "IR_mean", "IR_max", "IR_min"]
features = data[columns]

subset_features = features.loc[(features.IR_min > 230) & (features.IR_max < 240)]

df_in_values = [
    [5.000000e+00, 7.800000e+01, 1.704300e+01, 2.990000e-02,
     1.403542e+02, 7.640000e-02, 2.961400e+00, 1.950000e+02,
     2.380000e+02, 2.306953e+02],
    [4.000000e+00, 6.100000e+01, 1.045700e+01, 1.660000e-02,
     4.697920e+01, 1.185000e-01, 2.457600e+00, 1.980000e+02,
     2.340000e+02, 2.306445e+02],
    [6.000000e+00, 6.400000e+01, 1.573050e+01, 2.490000e-02,
     8.929170e+01, 8.080000e-02, 2.818500e+00, 2.130000e+02,
     2.370000e+02, 2.316563e+02],
    [5.000000e+00, 1.160000e+02, 1.766020e+01, 3.320000e-02,
     2.632917e+02, 1.148000e-01, 2.812000e+00, 1.760000e+02,
     2.380000e+02, 2.302891e+02],
    [5.000000e+00, 7.000000e+01, 1.229690e+01, 1.860000e-02,
     6.952500e+01, 1.573000e-01, 2.464800e+00, 2.000000e+02,
     2.390000e+02, 2.335234e+02],
    [5.000000e+00, 5.600000e+01, 9.156300e+00, 1.010000e-02,
     3.266670e+01, 2.477000e-01, 1.888900e+00, 2.120000e+02,
     2.370000e+02, 2.347891e+02],
    [7.000000e+00, 8.500000e+01, 1.367970e+01, 1.800000e-02,
     8.567920e+01, 1.923000e-01, 2.279200e+00, 1.980000e+02,
     2.390000e+02, 2.329180e+02],
    [4.000000e+00, 2.600000e+01, 6.046900e+00, 4.200000e-03,
     6.125000e+00, 3.643000e-01, 1.326900e+00, 2.190000e+02,
     2.390000e+02, 2.338672e+02],
    [7.000000e+00, 1.290000e+02, 1.946480e+01, 3.270000e-02,
     1.836667e+02, 8.000000e-02, 2.969100e+00, 1.760000e+02,
     2.390000e+02, 2.320781e+02],
    [7.000000e+00, 9.500000e+01, 1.701560e+01, 2.280000e-02,
     8.443330e+01, 1.009000e-01, 2.705600e+00, 1.930000e+02,
     2.390000e+02, 2.327109e+02],
    [5.000000e+00, 1.100000e+02, 1.792190e+01, 2.710000e-02,
     1.356833e+02, 9.610000e-02, 2.834000e+00, 1.910000e+02,
     2.390000e+02, 2.323164e+02]
]

df_in_index = [25, 53, 58, 89, 90, 123, 159, 204, 282, 315, 463]

df_in_columns = ["visible_mean", "visible_max", "visible_min", "visible_mean_distribution",
                 "visible_contrast", "visible_entropy", "visible_second_angular_momentum",
                 "IR_mean", "IR_max", "IR_min"]

df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)

pd.testing.assert_frame_equal(df_in, subset_features)


def test_log_transform():
    """Test for log transformation for acceptable inputs"""
    df_true = pd.DataFrame(
        [[5.00000000e+00, 7.80000000e+01, 1.70430000e+01,
          2.99000000e-02, 1.40354200e+02, 7.64000000e-02,
          2.96140000e+00, 1.95000000e+02, 2.38000000e+02,
          2.30695300e+02, -2.57177258e+00],
         [4.00000000e+00, 6.10000000e+01, 1.04570000e+01,
          1.66000000e-02, 4.69792000e+01, 1.18500000e-01,
          2.45760000e+00, 1.98000000e+02, 2.34000000e+02,
          2.30644500e+02, -2.13284232e+00],
         [6.00000000e+00, 6.40000000e+01, 1.57305000e+01,
          2.49000000e-02, 8.92917000e+01, 8.08000000e-02,
          2.81850000e+00, 2.13000000e+02, 2.37000000e+02,
          2.31656300e+02, -2.51577831e+00],
         [5.00000000e+00, 1.16000000e+02, 1.76602000e+01,
          3.32000000e-02, 2.63291700e+02, 1.14800000e-01,
          2.81200000e+00, 1.76000000e+02, 2.38000000e+02,
          2.30289100e+02, -2.16456380e+00],
         [5.00000000e+00, 7.00000000e+01, 1.22969000e+01,
          1.86000000e-02, 6.95250000e+01, 1.57300000e-01,
          2.46480000e+00, 2.00000000e+02, 2.39000000e+02,
          2.33523400e+02, -1.84960047e+00],
         [5.00000000e+00, 5.60000000e+01, 9.15630000e+00,
          1.01000000e-02, 3.26667000e+01, 2.47700000e-01,
          1.88890000e+00, 2.12000000e+02, 2.37000000e+02,
          2.34789100e+02, -1.39553694e+00],
         [7.00000000e+00, 8.50000000e+01, 1.36797000e+01,
          1.80000000e-02, 8.56792000e+01, 1.92300000e-01,
          2.27920000e+00, 1.98000000e+02, 2.39000000e+02,
          2.32918000e+02, -1.64869863e+00],
         [4.00000000e+00, 2.60000000e+01, 6.04690000e+00,
          4.20000000e-03, 6.12500000e+00, 3.64300000e-01,
          1.32690000e+00, 2.19000000e+02, 2.39000000e+02,
          2.33867200e+02, -1.00977757e+00],
         [7.00000000e+00, 1.29000000e+02, 1.94648000e+01,
          3.27000000e-02, 1.83666700e+02, 8.00000000e-02,
          2.96910000e+00, 1.76000000e+02, 2.39000000e+02,
          2.32078100e+02, -2.52572864e+00],
         [7.00000000e+00, 9.50000000e+01, 1.70156000e+01,
          2.28000000e-02, 8.44333000e+01, 1.00900000e-01,
          2.70560000e+00, 1.93000000e+02, 2.39000000e+02,
          2.32710900e+02, -2.29362535e+00],
         [5.00000000e+00, 1.10000000e+02, 1.79219000e+01,
          2.71000000e-02, 1.35683300e+02, 9.61000000e-02,
          2.83400000e+00, 1.91000000e+02, 2.39000000e+02,
          2.32316400e+02, -2.34236596e+00]],
        index=[25, 53, 58, 89, 90, 123, 159, 204, 282, 315, 463],
        columns=["visible_mean", "visible_max", "visible_min", "visible_mean_distribution",
                 "visible_contrast", "visible_entropy", "visible_second_angular_momentum",
                 "IR_mean", "IR_max", "IR_min", "log_entropy"])
    df_test = df_in.copy()
    # compute test results
    df_results = generate_additional_features.log_transform(df_test, "visible_entropy",
                                                            "log_entropy")
    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_results)


def test_log_transform_no_log_col():
    """Test for log transformation with no transformed features"""
    df_test = pd.DataFrame([], index=[], columns=[])
    with pytest.raises(AttributeError):
        generate_additional_features.log_transform(df_test, "visible_entropy", "log_entropy")


def test_log_transform_non_df():
    """Test for log transformation for non dataframe"""
    df_test = 'I am not a dataframe'
    with pytest.raises(TypeError):
        generate_additional_features.log_transform(df_test, "visible_entropy", "log_entropy")


def test_multiply():
    """Test for multiplication for acceptable inputs"""
    df_true = pd.DataFrame(
        [[5.00000000e+00, 7.80000000e+01, 1.70430000e+01, 2.99000000e-02,
          1.40354200e+02, 7.64000000e-02, 2.96140000e+00, 1.95000000e+02,
          2.38000000e+02, 2.30695300e+02, 1.07230609e+01],
         [4.00000000e+00, 6.10000000e+01, 1.04570000e+01, 1.66000000e-02,
          4.69792000e+01, 1.18500000e-01, 2.45760000e+00, 1.98000000e+02,
          2.34000000e+02, 2.30644500e+02, 5.56703520e+00],
         [6.00000000e+00, 6.40000000e+01, 1.57305000e+01, 2.49000000e-02,
          8.92917000e+01, 8.08000000e-02, 2.81850000e+00, 2.13000000e+02,
          2.37000000e+02, 2.31656300e+02, 7.21476936e+00],
         [5.00000000e+00, 1.16000000e+02, 1.76602000e+01, 3.32000000e-02,
          2.63291700e+02, 1.14800000e-01, 2.81200000e+00, 1.76000000e+02,
          2.38000000e+02, 2.30289100e+02, 3.02258872e+01],
         [5.00000000e+00, 7.00000000e+01, 1.22969000e+01, 1.86000000e-02,
          6.95250000e+01, 1.57300000e-01, 2.46480000e+00, 2.00000000e+02,
          2.39000000e+02, 2.33523400e+02, 1.09362825e+01],
         [5.00000000e+00, 5.60000000e+01, 9.15630000e+00, 1.01000000e-02,
          3.26667000e+01, 2.47700000e-01, 1.88890000e+00, 2.12000000e+02,
          2.37000000e+02, 2.34789100e+02, 8.09154159e+00],
         [7.00000000e+00, 8.50000000e+01, 1.36797000e+01, 1.80000000e-02,
          8.56792000e+01, 1.92300000e-01, 2.27920000e+00, 1.98000000e+02,
          2.39000000e+02, 2.32918000e+02, 1.64761102e+01],
         [4.00000000e+00, 2.60000000e+01, 6.04690000e+00, 4.20000000e-03,
          6.12500000e+00, 3.64300000e-01, 1.32690000e+00, 2.19000000e+02,
          2.39000000e+02, 2.33867200e+02, 2.23133750e+00],
         [7.00000000e+00, 1.29000000e+02, 1.94648000e+01, 3.27000000e-02,
          1.83666700e+02, 8.00000000e-02, 2.96910000e+00, 1.76000000e+02,
          2.39000000e+02, 2.32078100e+02, 1.46933360e+01],
         [7.00000000e+00, 9.50000000e+01, 1.70156000e+01, 2.28000000e-02,
          8.44333000e+01, 1.00900000e-01, 2.70560000e+00, 1.93000000e+02,
          2.39000000e+02, 2.32710900e+02, 8.51931997e+00],
         [5.00000000e+00, 1.10000000e+02, 1.79219000e+01, 2.71000000e-02,
          1.35683300e+02, 9.61000000e-02, 2.83400000e+00, 1.91000000e+02,
          2.39000000e+02, 2.32316400e+02, 1.30391651e+01]],
        index=[25, 53, 58, 89, 90, 123, 159, 204, 282, 315, 463],
        columns=["visible_mean", "visible_max", "visible_min", "visible_mean_distribution",
                 "visible_contrast", "visible_entropy", "visible_second_angular_momentum",
                 "IR_mean", "IR_max", "IR_min", "entropy_x_contrast"])
    df_test = df_in.copy()
    # compute test results
    df_results = generate_additional_features.multiply(df_test, "visible_entropy",
                                                       "visible_contrast", "entropy_x_contrast")
    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_results)


def test_multiply_no_mult_col():
    """Test for multiplication for no transformed features"""
    df_test = pd.DataFrame([], index=[], columns=[])
    with pytest.raises(AttributeError):
        generate_additional_features.multiply(df_test, "visible_entropy", "visible_contrast",
                                              "entropy_x_contrast")


def test_multiply_non_df():
    """Test for multiplication for non dataframe"""
    df_test = "I am not a dataframe"
    with pytest.raises(TypeError):
        generate_additional_features.multiply(df_test, "visible_entropy", "visible_contrast",
                                              "entropy_x_contrast")


def test_col_range():
    """Test for range computation for acceptable inputs"""
    df_true = pd.DataFrame(
        [[5.000000e+00, 7.800000e+01, 1.704300e+01, 2.990000e-02,
          1.403542e+02, 7.640000e-02, 2.961400e+00, 1.950000e+02,
          2.380000e+02, 2.306953e+02, 7.304700e+00],
         [4.000000e+00, 6.100000e+01, 1.045700e+01, 1.660000e-02,
          4.697920e+01, 1.185000e-01, 2.457600e+00, 1.980000e+02,
          2.340000e+02, 2.306445e+02, 3.355500e+00],
         [6.000000e+00, 6.400000e+01, 1.573050e+01, 2.490000e-02,
          8.929170e+01, 8.080000e-02, 2.818500e+00, 2.130000e+02,
          2.370000e+02, 2.316563e+02, 5.343700e+00],
         [5.000000e+00, 1.160000e+02, 1.766020e+01, 3.320000e-02,
          2.632917e+02, 1.148000e-01, 2.812000e+00, 1.760000e+02,
          2.380000e+02, 2.302891e+02, 7.710900e+00],
         [5.000000e+00, 7.000000e+01, 1.229690e+01, 1.860000e-02,
          6.952500e+01, 1.573000e-01, 2.464800e+00, 2.000000e+02,
          2.390000e+02, 2.335234e+02, 5.476600e+00],
         [5.000000e+00, 5.600000e+01, 9.156300e+00, 1.010000e-02,
          3.266670e+01, 2.477000e-01, 1.888900e+00, 2.120000e+02,
          2.370000e+02, 2.347891e+02, 2.210900e+00],
         [7.000000e+00, 8.500000e+01, 1.367970e+01, 1.800000e-02,
          8.567920e+01, 1.923000e-01, 2.279200e+00, 1.980000e+02,
          2.390000e+02, 2.329180e+02, 6.082000e+00],
         [4.000000e+00, 2.600000e+01, 6.046900e+00, 4.200000e-03,
          6.125000e+00, 3.643000e-01, 1.326900e+00, 2.190000e+02,
          2.390000e+02, 2.338672e+02, 5.132800e+00],
         [7.000000e+00, 1.290000e+02, 1.946480e+01, 3.270000e-02,
          1.836667e+02, 8.000000e-02, 2.969100e+00, 1.760000e+02,
          2.390000e+02, 2.320781e+02, 6.921900e+00],
         [7.000000e+00, 9.500000e+01, 1.701560e+01, 2.280000e-02,
          8.443330e+01, 1.009000e-01, 2.705600e+00, 1.930000e+02,
          2.390000e+02, 2.327109e+02, 6.289100e+00],
         [5.000000e+00, 1.100000e+02, 1.792190e+01, 2.710000e-02,
          1.356833e+02, 9.610000e-02, 2.834000e+00, 1.910000e+02,
          2.390000e+02, 2.323164e+02, 6.683600e+00]],
        index=[25, 53, 58, 89, 90, 123, 159, 204, 282, 315, 463],
        columns=["visible_mean", "visible_max", "visible_min", "visible_mean_distribution",
                 "visible_contrast", "visible_entropy", "visible_second_angular_momentum",
                 "IR_mean", "IR_max", "IR_min", "IR_range"])
    df_test = df_in.copy()
    # compute test results
    df_results = generate_additional_features.col_range(df_test, "IR_min", "IR_max", "IR_range")
    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_results)


def test_col_range_no_col():
    """Test for range computation for no transformed features"""
    df_test = pd.DataFrame([], index=[], columns=[])
    with pytest.raises(AttributeError):
        generate_additional_features.col_range(df_test, "IR_min", "IR_max", "IR_range")


def test_col_range_non_df():
    """Test for range computation for non dataframe"""
    df_test = "I am not a dataframe"
    with pytest.raises(TypeError):
        generate_additional_features.col_range(df_test, "IR_min", "IR_max", "IR_range")


def test_norm_range():
    """Test for normalized range computation for acceptable inputs"""
    df_true = pd.DataFrame(
        [[5.00000000e+00, 7.80000000e+01, 1.70430000e+01, 2.99000000e-02,
          1.40354200e+02, 7.64000000e-02, 2.96140000e+00, 1.95000000e+02,
          2.38000000e+02, 2.30695300e+02, 3.74600000e-02],
         [4.00000000e+00, 6.10000000e+01, 1.04570000e+01, 1.66000000e-02,
          4.69792000e+01, 1.18500000e-01, 2.45760000e+00, 1.98000000e+02,
          2.34000000e+02, 2.30644500e+02, 1.69469697e-02],
         [6.00000000e+00, 6.40000000e+01, 1.57305000e+01, 2.49000000e-02,
          8.92917000e+01, 8.08000000e-02, 2.81850000e+00, 2.13000000e+02,
          2.37000000e+02, 2.31656300e+02, 2.50877934e-02],
         [5.00000000e+00, 1.16000000e+02, 1.76602000e+01, 3.32000000e-02,
          2.63291700e+02, 1.14800000e-01, 2.81200000e+00, 1.76000000e+02,
          2.38000000e+02, 2.30289100e+02, 4.38119318e-02],
         [5.00000000e+00, 7.00000000e+01, 1.22969000e+01, 1.86000000e-02,
          6.95250000e+01, 1.57300000e-01, 2.46480000e+00, 2.00000000e+02,
          2.39000000e+02, 2.33523400e+02, 2.73830000e-02],
         [5.00000000e+00, 5.60000000e+01, 9.15630000e+00, 1.01000000e-02,
          3.26667000e+01, 2.47700000e-01, 1.88890000e+00, 2.12000000e+02,
          2.37000000e+02, 2.34789100e+02, 1.04287736e-02],
         [7.00000000e+00, 8.50000000e+01, 1.36797000e+01, 1.80000000e-02,
          8.56792000e+01, 1.92300000e-01, 2.27920000e+00, 1.98000000e+02,
          2.39000000e+02, 2.32918000e+02, 3.07171717e-02],
         [4.00000000e+00, 2.60000000e+01, 6.04690000e+00, 4.20000000e-03,
          6.12500000e+00, 3.64300000e-01, 1.32690000e+00, 2.19000000e+02,
          2.39000000e+02, 2.33867200e+02, 2.34374429e-02],
         [7.00000000e+00, 1.29000000e+02, 1.94648000e+01, 3.27000000e-02,
          1.83666700e+02, 8.00000000e-02, 2.96910000e+00, 1.76000000e+02,
          2.39000000e+02, 2.32078100e+02, 3.93289773e-02],
         [7.00000000e+00, 9.50000000e+01, 1.70156000e+01, 2.28000000e-02,
          8.44333000e+01, 1.00900000e-01, 2.70560000e+00, 1.93000000e+02,
          2.39000000e+02, 2.32710900e+02, 3.25860104e-02],
         [5.00000000e+00, 1.10000000e+02, 1.79219000e+01, 2.71000000e-02,
          1.35683300e+02, 9.61000000e-02, 2.83400000e+00, 1.91000000e+02,
          2.39000000e+02, 2.32316400e+02, 3.49926702e-02]],
        index=[25, 53, 58, 89, 90, 123, 159, 204, 282, 315, 463],
        columns=["visible_mean", "visible_max", "visible_min", "visible_mean_distribution",
                 "visible_contrast", "visible_entropy", "visible_second_angular_momentum",
                 "IR_mean", "IR_max", "IR_min", "IR_norm_range"])
    df_test = df_in.copy()
    # compute test results
    df_results = generate_additional_features.norm_range(df_test, "IR_min", "IR_max", "IR_mean",
                                                         "IR_norm_range")
    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_results)


def test_norm_range_no_col():
    """Test for normalized range computation for no transformed features"""
    df_test = pd.DataFrame([], index=[], columns=[])
    with pytest.raises(AttributeError):
        generate_additional_features.norm_range(df_test, "IR_min", "IR_max", "IR_mean",
                                                "IR_norm_range")


def test_norm_range_non_df():
    """Test for normalized range computation for non dataframe"""
    df_test = "I am not a dataframe"
    with pytest.raises(TypeError):
        generate_additional_features.norm_range(df_test, "IR_min", "IR_max", "IR_mean",
                                                "IR_norm_range")
