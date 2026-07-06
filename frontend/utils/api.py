from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import requests

# ============================================================
# Configuration
# ============================================================

# Change this when deploying
BASE_URL = "http://127.0.0.1:8000"

# Default timeout (seconds)
TIMEOUT = 300


# ============================================================
# Custom Exception
# ============================================================

class APIError(Exception):
    """Raised when the backend returns an error."""

    pass


# ============================================================
# Internal Helpers
# ============================================================

def _url(endpoint: str) -> str:
    """Build complete API URL."""

    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{BASE_URL}{endpoint}"


def _handle_response(response: requests.Response) -> Any:
    """Validate backend response."""

    try:
        data = response.json()

    except Exception:

        if response.ok:
            return response.content

        raise APIError("Backend returned an invalid response.")

    if response.ok:
        return data

    message = (
        data.get("detail")
        or data.get("message")
        or f"HTTP {response.status_code}"
    )

    raise APIError(message)


def _request(
    method: str,
    endpoint: str,
    **kwargs,
) -> Any:
    """
    Generic request handler used by all API functions.
    """

    try:

        response = requests.request(
            method=method,
            url=_url(endpoint),
            timeout=TIMEOUT,
            **kwargs,
        )

        return _handle_response(response)

    except requests.Timeout:
        raise APIError(
            "The backend took too long to respond."
        )

    except requests.ConnectionError:
        raise APIError(
            "Cannot connect to the backend. "
            "Is FastAPI running?"
        )

    except requests.RequestException as e:
        raise APIError(str(e))


# ============================================================
# Health Check
# ============================================================

def check_backend() -> bool:
    """
    Returns True if backend is online.
    """

    try:

        requests.get(
            BASE_URL,
            timeout=5,
        )

        return True

    except Exception:

        return False


# ============================================================
# Generic HTTP Wrappers
# ============================================================

def get(endpoint: str) -> Any:

    return _request(
        "GET",
        endpoint,
    )


def delete(endpoint: str) -> Any:

    return _request(
        "DELETE",
        endpoint,
    )


def post_json(
    endpoint: str,
    payload: Dict[str, Any],
) -> Any:

    return _request(
        "POST",
        endpoint,
        json=payload,
    )


def post_file(
    endpoint: str,
    uploaded_file,
    params: Optional[dict] = None,
) -> Any:
    """
    Upload CSV file to backend.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }

    return _request(
        "POST",
        endpoint,
        files=files,
        params=params,
    )


# ============================================================
# Dataset APIs
# ============================================================

def upload_dataset(uploaded_file):

    return post_file(
        "/upload",
        uploaded_file,
    )


def clean_dataset(uploaded_file):

    return post_file(
        "/clean",
        uploaded_file,
    )


def extract_metadata(uploaded_file):

    return post_file(
        "/metadata",
        uploaded_file,
    )


def run_eda(uploaded_file):

    return post_file(
        "/eda",
        uploaded_file,
    )


# ============================================================
# Machine Learning
# ============================================================

def train_model(
    df: pd.DataFrame,
    target_column=None,
):
    """
    Train AutoML pipeline.
    """

    params = {}

    if target_column:
        params["target_column"] = target_column

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    files = {
        "file": (
            "cleaned_dataset.csv",
            csv_bytes,
            "text/csv",
        )
    }

    return _request(
        "POST",
        "/train-model/upload",
        files=files,
        params=params,
    )
# ============================================================
# Prediction APIs
# ============================================================

def predict(
    uploaded_file,
    run_name: str,
    return_proba: bool = False,
):
    """
    Predict using a saved model.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }

    params = {
        "run_name": run_name,
        "return_proba": return_proba,
    }

    return _request(
        "POST",
        "/predict",
        files=files,
        params=params,
    )


def download_predictions(
    uploaded_file,
    run_name: str,
    return_proba: bool = False,
):
    """
    Download prediction CSV.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv",
        )
    }

    params = {
        "run_name": run_name,
        "return_proba": return_proba,
    }

    response = requests.post(
        _url("/predict/download"),
        files=files,
        params=params,
        timeout=TIMEOUT,
    )

    if response.ok:
        return response.content

    raise APIError(
        f"Prediction download failed ({response.status_code})"
    )


# ============================================================
# Reports
# ============================================================

def generate_report(
    run_name: str,
):
    """
    Generate all model reports.
    """

    return post_json(
        f"/model/{run_name}/report",
        {},
    )


# ============================================================
# Model Registry
# ============================================================

def get_models():
    """
    List all saved models.
    """

    return get("/models")


def get_leaderboard():
    """
    Get model leaderboard.
    """

    return get("/leaderboard")


def delete_model(
    run_name: str,
):
    """
    Delete a saved model.
    """

    return delete(
        f"/model/{run_name}"
    )


# ============================================================
# Downloads
# ============================================================

def download_report(
    report_path: str,
):
    """
    Download a generated report.
    """

    response = requests.get(
        report_path,
        timeout=TIMEOUT,
    )

    if response.ok:
        return response.content

    raise APIError(
        "Unable to download report."
    )


# ============================================================
# Utilities
# ============================================================

def dataframe_from_predictions(
    response: dict,
) -> pd.DataFrame:
    """
    Convert prediction JSON to DataFrame.
    """

    predictions = response.get(
        "predictions",
        [],
    )

    return pd.DataFrame(predictions)


def leaderboard_dataframe(
    response: dict,
) -> pd.DataFrame:
    """
    Convert leaderboard JSON to DataFrame.
    """

    board = response.get(
        "leaderboard",
        [],
    )

    return pd.DataFrame(board)


def models_dataframe(
    response: dict,
) -> pd.DataFrame:
    """
    Convert models JSON to DataFrame.
    """

    models = response.get(
        "models",
        [],
    )

    return pd.DataFrame(models)


# ============================================================
# API Wrapper
# ============================================================

class APIClient:

    def upload(
        self,
        uploaded_file,
    ):
        return upload_dataset(uploaded_file)

    def clean(
        self,
        uploaded_file,
    ):
        return clean_dataset(uploaded_file)

    def metadata(
        self,
        uploaded_file,
    ):
        return extract_metadata(uploaded_file)

    def eda(
        self,
        uploaded_file,
    ):
        return run_eda(uploaded_file)

    def train(
        self,
        uploaded_file,
        target_column=None,
    ):
        return train_model(
            uploaded_file,
            target_column,
        )

    def predict(
        self,
        uploaded_file,
        run_name,
        return_proba=False,
    ):
        return predict(
            uploaded_file,
            run_name,
            return_proba,
        )

    def models(self):
        return get_models()

    def leaderboard(self):
        return get_leaderboard()

    def report(
        self,
        run_name,
    ):
        return generate_report(run_name)

    def delete(
        self,
        run_name,
    ):
        return delete_model(run_name)


api = APIClient()


# from __future__ import annotations

# from typing import Any

# import pandas as pd
# import requests

# BASE_URL = "http://127.0.0.1:8000"

# TIMEOUT = 300


# class APIError(Exception):
#     pass


# def _url(endpoint: str) -> str:
#     endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
#     return f"{BASE_URL}{endpoint}"


# def _handle_response(response: requests.Response) -> Any:
#     try:
#         data = response.json()
#     except Exception:
#         response.raise_for_status()
#         raise APIError("Invalid response received from backend.")

#     if response.ok:
#         return data

#     message = data.get("detail") or data.get("message") or "Backend request failed."
#     raise APIError(message)


# def get(endpoint: str) -> Any:
#     response = requests.get(
#         _url(endpoint),
#         timeout=TIMEOUT,
#     )
#     return _handle_response(response)


# def delete(endpoint: str) -> Any:
#     response = requests.delete(
#         _url(endpoint),
#         timeout=TIMEOUT,
#     )
#     return _handle_response(response)


# def post_json(endpoint: str, payload: dict) -> Any:
#     response = requests.post(
#         _url(endpoint),
#         json=payload,
#         timeout=TIMEOUT,
#     )
#     return _handle_response(response)


# def post_file(endpoint: str, uploaded_file) -> Any:
#     files = {
#         "file": (
#             uploaded_file.name,
#             uploaded_file.getvalue(),
#             "text/csv",
#         )
#     }

#     response = requests.post(
#         _url(endpoint),
#         files=files,
#         timeout=TIMEOUT,
#     )

#     return _handle_response(response)


# def upload_dataset(uploaded_file):
#     return post_file("/upload", uploaded_file)


# def clean_dataset(uploaded_file):
#     return post_file("/clean", uploaded_file)


# def extract_metadata(uploaded_file):
#     return post_file("/metadata", uploaded_file)


# def run_eda(uploaded_file):
#     return post_file("/eda", uploaded_file)


# def train_model(uploaded_file):
#     return post_file("/train-model/upload", uploaded_file)


# def predict(uploaded_file, run_name: str):
#     files = {
#         "file": (
#             uploaded_file.name,
#             uploaded_file.getvalue(),
#             "text/csv",
#         )
#     }

#     response = requests.post(
#         _url("/predict"),
#         params={"run_name": run_name},
#         files=files,
#         timeout=TIMEOUT,
#     )

#     return _handle_response(response)


# def get_models():
#     return get("/models")


# def get_leaderboard():
#     return get("/leaderboard")


# def delete_model(run_name: str):
#     return delete(f"/model/{run_name}")


# def download_report(run_name: str):
#     response = requests.get(
#         _url(f"/report/{run_name}"),
#         timeout=TIMEOUT,
#     )

#     if response.ok:
#         return response.content

#     raise APIError("Unable to download report.")


# def dataframe_from_predictions(response: dict) -> pd.DataFrame:
#     predictions = response.get("predictions", [])
#     return pd.DataFrame(predictions)