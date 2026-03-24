from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from pathlib import Path
import subprocess
import sys
from keras.models import load_model

app = Flask(__name__)

# Load saved models
import sklearn.linear_model  # needed for unpickling

BASE_DIR = Path(__file__).resolve().parent
LR_MODEL_PATH = BASE_DIR / "lr_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
SCALED_DATA_PATH = BASE_DIR / "df_scaled.pkl"
LSTM_MODEL_PATH = BASE_DIR / "lstm_model.h5"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"
TARGET_DATA_FILE = "All_Stocks_Data.csv"
DEFAULT_SEQUENCE_LENGTH = 10


def ensure_model_artifacts():
    required_files = [
        LR_MODEL_PATH,
        SCALER_PATH,
        SCALED_DATA_PATH,
        LSTM_MODEL_PATH,
        METADATA_PATH,
    ]
    missing_files = [file for file in required_files if not file.exists()]
    should_retrain = bool(missing_files)

    if not should_retrain:
        try:
            with open(METADATA_PATH, "rb") as metadata_file:
                metadata = pickle.load(metadata_file)
            should_retrain = metadata.get("data_file") != TARGET_DATA_FILE
        except Exception:
            should_retrain = True

    if not should_retrain:
        return

    print("Model artifacts missing or outdated. Running train_models.py...")
    subprocess.run(
        [sys.executable, str(BASE_DIR / "train_models.py")],
        check=True,
        cwd=BASE_DIR,
    )


ensure_model_artifacts()

with open(LR_MODEL_PATH, "rb") as lr_file:
    lr_model = pickle.load(lr_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(SCALED_DATA_PATH, "rb") as scaled_data_file:
    df_scaled = pickle.load(scaled_data_file)

with open(METADATA_PATH, "rb") as metadata_file:
    metadata = pickle.load(metadata_file)

lstm_model = load_model(str(LSTM_MODEL_PATH))

sequence_length = int(metadata.get("seq_length", DEFAULT_SEQUENCE_LENGTH))
trained_symbol = str(metadata.get("symbol", "Unknown"))
df_scaled = np.asarray(df_scaled, dtype=np.float32).reshape(-1, 1)

if len(df_scaled) < sequence_length:
    raise ValueError(
        "Scaled dataset is shorter than the LSTM sequence length. Retrain models with more data."
    )


def predict_lstm_future(days_ahead):
    rolling_window = df_scaled[-sequence_length:].flatten().tolist()
    future_scaled = []

    for _ in range(days_ahead):
        x_input = np.array(rolling_window[-sequence_length:], dtype=np.float32).reshape(
            1, sequence_length, 1
        )
        next_value = float(lstm_model.predict(x_input, verbose=0)[0][0])
        future_scaled.append(next_value)
        rolling_window.append(next_value)

    return (
        scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1))
        .flatten()
        .tolist()
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        days_ahead = int(data.get("days", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "days must be an integer"}), 400

    if days_ahead < 1:
        return jsonify({"error": "days must be at least 1"}), 400

    if days_ahead > 365:
        return jsonify({"error": "days must be 365 or less"}), 400

    try:
        future_x = np.array(range(len(df_scaled), len(df_scaled) + days_ahead)).reshape(
            -1, 1
        )
        lr_future_scaled = lr_model.predict(future_x)
        lr_future = (
            scaler.inverse_transform(np.array(lr_future_scaled).reshape(-1, 1))
            .flatten()
            .tolist()
        )

        lstm_future = predict_lstm_future(days_ahead)
    except Exception as exc:
        return jsonify({"error": f"prediction failed: {exc}"}), 500

    return jsonify(
        {
            "symbol": trained_symbol,
            "linear_regression": lr_future,
            "lstm": lstm_future,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
