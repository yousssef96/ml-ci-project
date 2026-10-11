from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(
    __file__
).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "breast_cancer_pipeline.joblib"
)

model = joblib.load(
    MODEL_PATH
)

dataset_info = (
    load_breast_cancer()
)

class_names = {
    0: "malignant",
    1: "benign"
}

api_to_model_feature = {
    feature.replace(" ", "_"):
        feature
    for feature
    in dataset_info.feature_names
}


class BreastCancerInput(BaseModel):

    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

    model_config = {
        "extra": "forbid"
    }


def prepare_input(
    request_data
):

    request_dict = (
        request_data.model_dump()
    )

    model_dict = {
        original_name:
            request_dict[
                api_name
            ]
        for api_name,
            original_name
        in api_to_model_feature.items()
    }

    return pd.DataFrame(
        [model_dict]
    )


app = FastAPI(
    title="ML CI Demo API",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message":
            "ML CI Demo API is running"
    }


@app.get("/health")
def health():

    return {
        "status":
            "healthy"
    }


@app.get("/model-info")
def model_info():

    return {
        "model_name":
            "breast_cancer_classifier",

        "model_version":
            "1.0.0",

        "number_of_features":
            len(
                dataset_info.feature_names
            ),

        "classes":
            [
                "malignant",
                "benign"
            ]
    }


@app.post("/predict")
def predict(
    input_data:
        BreastCancerInput
):

    input_df = prepare_input(
        input_data
    )

    prediction = int(
        model.predict(
            input_df
        )[0]
    )

    probabilities = (
        model.predict_proba(
            input_df
        )[0]
    )

    return {
        "prediction":
            prediction,

        "class_name":
            class_names[
                prediction
            ],

        "probability":
            float(
                probabilities[
                    prediction
                ]
            )
    }
