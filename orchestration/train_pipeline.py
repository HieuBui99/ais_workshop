import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import gdown
import numpy as np
import pandas as pd
import wandb
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from prefect import flow, task

@dataclass
class Config:
    train_data_url: str = "1P7h3ni1T-QY_ECw4EWPSZ8tdzzlgDgcU"
    test_data_url: str = "1eFdBVdQlYtxxfkqW_c-C9A5Kqsb06O80"
    model_path: str = "../weight"
    data_path: str = "../data/data_raw"


def download_data(train_url: str, val_url: str, output_path: str) -> str:
    """
    Download data from Google Drive using gdown.
    """
    train_data_path = f"{output_path}/green_taxi_data_01.csv"
    test_data_path = f"{output_path}/green_taxi_data_02.csv"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gdown.download(
        f"https://drive.google.com/uc?id={train_url}", train_data_path, quiet=False
    )
    gdown.download(
        f"https://drive.google.com/uc?id={val_url}", test_data_path, quiet=False
    )
    return train_data_path, test_data_path


def read_clean_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


def preprocess_data(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> Tuple[DictVectorizer, csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    """
    Preprocess the data for training.
    """
    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]
    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    return dv, X_train, y_train, X_val, y_val


def train_model(X_train: csr_matrix, y_train: np.ndarray) -> LinearRegression:
    """
    Train the model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression, X_val: csr_matrix, y_val: np.ndarray
) -> float:
    """
    Evaluate the model.
    """
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    return rmse


def save_model(
    model: LinearRegression,
    dv: DictVectorizer,
    model_path: str,
    run: wandb.Api.run = None,
):
    """
    Save the model and DictVectorizer.
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(f"{model_path}/model.bin", "wb") as f_out:
        pickle.dump(model, f_out)

    with open(f"{model_path}/dv.bin", "wb") as f_out:
        pickle.dump(dv, f_out)

    model_artifact = wandb.Artifact(
        name="model",
        type="model",
        description="Linear Regression model for taxi duration prediction",
    )
    model_artifact.add_file(f"{model_path}/model.bin")
    model_artifact.add_file(f"{model_path}/dv.bin")
    run.log_artifact(model_artifact)


def main():
    cfg = Config()

    run = wandb.init(
        project="taxi-duration-prediction",
        config=dict(
            train_data_url=cfg.train_data_url,
            test_data_url=cfg.test_data_url,
            model_path=cfg.model_path,
            data_path=cfg.data_path,
        ),
    )

    # Download data
    train_data_path, test_data_path = download_data(
        cfg.train_data_url, cfg.test_data_url, cfg.data_path
    )

    # Read and clean data
    df_train = read_clean_data(train_data_path)
    df_val = read_clean_data(test_data_path)

    # Preprocess data
    dv, X_train, y_train, X_val, y_val = preprocess_data(df_train, df_val)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    rmse = evaluate_model(model, X_val, y_val)

    run.summary["rmse"] = rmse
    print(f"RMSE: {rmse:.3f}")
    save_model(model, dv, cfg.model_path, run)


if __name__ == "__main__":
    main()
