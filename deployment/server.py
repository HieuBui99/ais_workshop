import pickle
from contextlib import asynccontextmanager

import wandb
from fastapi import FastAPI
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel


class TaxiRide(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Download model from model registry
    wandb_api = wandb.Api()
    artifact = wandb_api.artifact(
        "hieubui99/model-registry/taxi-model:latest", type="model"
    )
    artifact_dir = artifact.download()
    print("Downloaded latest model to", artifact_dir)


    # Load the model and DictVectorizer
    model_path = f"{artifact_dir}/model.bin"
    dv_path = f"{artifact_dir}/dv.bin"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(dv_path, "rb") as f:
        dv = pickle.load(f)

    app.state.model = model
    app.state.dv = dv

    yield

    # Cleanup code if needed
    # For example, close any database connections or release resources


app = FastAPI(lifespan=lifespan)



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(features: TaxiRide):
    """
    Predict the duration of a taxi ride.
    """
    model = app.state.model
    dv = app.state.dv

    # Convert the features to a dictionary
    ride_dict = features.model_dump()
    X = dv.transform([ride_dict])
    # Make prediction
    y_pred = model.predict(X)[0]

    return {"duration": y_pred}