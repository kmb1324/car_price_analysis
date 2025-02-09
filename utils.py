from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def create_X_martix(
    data: pd.DataFrame,
    feature_columns: list,
    transformed_category: Optional[np.ndarray] = None,
) -> np.ndarray:
    """creates X matrix for model

    Args:
        data (pd.DataFrame): input data
        feature_columns (list): feature columns
        transformed_category (Optional[np.ndarray], optional): Transformed Category

    Returns:
        np.ndarray: X matrix
    """
    feature_len = len(feature_columns)
    X_feat = data[feature_columns].values

    if feature_len == 1:
        X_feat = X_feat.reshape(-1, 1)

    if transformed_category is not None:
        X = np.hstack((X_feat, transformed_category))
    else:
        X = X_feat

    return X


def fit_model(
    data: pd.DataFrame,
    feature_columns: list,
    output_column: str,
    categorical_columns: Optional[list] = None,
) -> Tuple[Optional[OneHotEncoder], LinearRegression]:
    """Creats a linear regression model, using one hot encoder
      to encode categorical columns

    Args:
        data (pd.DataFrame): data
        feature_columns (list): features names
        output_column (str): output name
        categorical_columns (Optional[np.ndarray], optional): Any categorical columns. Defaults to None.

    Returns:
        Tuple[Optional[OneHotEncoder], LinearRegression]: encoder and model
    """
    if categorical_columns is not None:
        encoder = OneHotEncoder(sparse_output=False)
        category_transformed = encoder.fit_transform(data[categorical_columns])
        X = create_X_martix(data, feature_columns, category_transformed)
    else:
        encoder = None
        X = create_X_martix(data, feature_columns)

    fit = LinearRegression().fit(X, data[output_column].values.reshape(-1, 1))

    return encoder, fit


def predict_y(
    data: pd.DataFrame,
    feature_columns: list,
    model: LinearRegression,
    categorical_columns: Optional[list] = None,
    encoder: Optional[OneHotEncoder] = None,
) -> np.ndarray:
    """predict y given dataset, model, and optional encoder

    Args:
        data (pd.DataFrame): data
        feature_columns (list): feature names
        model (LinearRegression): trained model
        categorical_columns (Optional[list], optional): categorical variable names. Defaults to None.
        encoder (Optional[OneHotEncoder], optional): encoder. Defaults to None.

    Returns:
        np.ndarray: predicted values
    """    
    if categorical_columns is not None:
        category_transformed = encoder.transform(data[categorical_columns])
        X = create_X_martix(data, feature_columns, category_transformed)
    else:
        X = create_X_martix(data, feature_columns)

    y = model.predict(X)

    return y

