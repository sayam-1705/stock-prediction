import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "All_Stocks_Data.csv"
METADATA_FILE = BASE_DIR / "model_metadata.pkl"
PREFERRED_SYMBOL = os.environ.get("STOCK_SYMBOL", "").strip() or None
SEQ_LENGTH = 10


def _finalize_close_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = frame.copy()
    cleaned_df["Close"] = pd.to_numeric(cleaned_df["Close"], errors="coerce")
    cleaned_df.index = pd.to_datetime(cleaned_df.index, errors="coerce", format="mixed")
    cleaned_df = cleaned_df.dropna(subset=["Close"])
    cleaned_df = cleaned_df[~cleaned_df.index.isna()].sort_index()
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="last")]

    if cleaned_df.empty:
        raise ValueError("No valid Date/Close rows found after preprocessing.")

    return cleaned_df


def _load_from_wide_all_stocks(
    raw_df: pd.DataFrame, preferred_symbol: str | None = None
) -> tuple[pd.DataFrame, str]:
    descriptor_col = raw_df.columns[0]
    if descriptor_col != "symbol":
        close_rows = raw_df[
            raw_df[descriptor_col].astype(str).str.lower() == "closing_price"
        ]
        if not close_rows.empty:
            raw_df = close_rows

    if "symbol" not in raw_df.columns:
        raise ValueError("'symbol' column is missing from All_Stocks_Data.csv")

    parsed_columns = pd.to_datetime(raw_df.columns, errors="coerce", format="%Y-%m-%d")
    date_columns = [
        column
        for column, parsed_date in zip(raw_df.columns, parsed_columns)
        if not pd.isna(parsed_date)
    ]

    if not date_columns:
        raise ValueError("No date columns were detected in All_Stocks_Data.csv")

    candidate_rows = raw_df
    if preferred_symbol:
        symbol_filter = (
            candidate_rows["symbol"].astype(str).str.upper() == preferred_symbol.upper()
        )
        candidate_rows = candidate_rows[symbol_filter]
        if candidate_rows.empty:
            raise ValueError(
                f"Symbol '{preferred_symbol}' was not found in {DATA_FILE.name}."
            )

    candidate_prices = candidate_rows[date_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_points_per_row = candidate_prices.notna().sum(axis=1)

    if valid_points_per_row.empty or valid_points_per_row.max() == 0:
        raise ValueError(
            "No valid numeric closing prices were found in All_Stocks_Data.csv"
        )

    selected_row_index = valid_points_per_row.idxmax()
    selected_symbol = str(candidate_rows.loc[selected_row_index, "symbol"])
    close_values = candidate_prices.loc[selected_row_index]

    close_values.index = pd.to_datetime(
        close_values.index, errors="coerce", format="%Y-%m-%d"
    )
    series_frame = pd.DataFrame(
        {"Close": close_values.values}, index=close_values.index
    )
    return _finalize_close_frame(series_frame), selected_symbol


def load_stock_data(
    csv_path: Path = DATA_FILE, preferred_symbol: str | None = PREFERRED_SYMBOL
) -> tuple[pd.DataFrame, str]:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    raw_df = pd.read_csv(csv_path)

    if raw_df.empty:
        raise ValueError(f"{csv_path} is empty.")

    if "symbol" in raw_df.columns:
        return _load_from_wide_all_stocks(raw_df, preferred_symbol=preferred_symbol)

    close_column = "Close" if "Close" in raw_df.columns else "Adj Close"
    if close_column not in raw_df.columns:
        raise ValueError(
            f"Unsupported format in {csv_path}. Expected either a 'symbol' column or a Close/Adj Close series."
        )

    if "Date" in raw_df.columns:
        date_source = raw_df["Date"]
    else:
        date_source = raw_df.iloc[:, 0]

    frame = pd.DataFrame({"Close": raw_df[close_column].values}, index=date_source)
    return _finalize_close_frame(frame), "single_series"


df, selected_symbol = load_stock_data(DATA_FILE, preferred_symbol=PREFERRED_SYMBOL)
print(f"[OK] Training models using symbol: {selected_symbol}")

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[["Close"]])

train_size = int(len(df_scaled) * 0.8)
train_data = df_scaled[:train_size]
test_data = df_scaled[train_size:]

# --- Linear Regression ---
X_train = np.array(range(len(train_data)))
y_train = train_data.flatten()
X_test = np.array(range(len(train_data), len(df_scaled)))
y_test = test_data.flatten()

lr_model = LinearRegression()
lr_model.fit(X_train.reshape(-1, 1), y_train)

lr_predictions = lr_model.predict(X_test.reshape(-1, 1))
lr_predictions = scaler.inverse_transform(lr_predictions.reshape(-1, 1))
lr_mse = mean_squared_error(scaler.inverse_transform(test_data), lr_predictions)
print(f"[OK] Linear Regression MSE: {lr_mse:.4f}")


# --- LSTM ---
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = SEQ_LENGTH

if len(train_data) <= seq_length or len(test_data) <= seq_length:
    raise ValueError(
        "Not enough data points for LSTM sequence training. Fetch more historical data."
    )

X_train_lstm, y_train_lstm = create_sequences(train_data, seq_length)
X_test_lstm, y_test_lstm = create_sequences(test_data, seq_length)

X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], seq_length, 1))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], seq_length, 1))

lstm_model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1),
    ]
)
lstm_model.compile(optimizer="adam", loss="mean_squared_error")
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, verbose=0)

lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
lstm_mse = mean_squared_error(
    scaler.inverse_transform(y_test_lstm.reshape(-1, 1)), lstm_predictions
)
print(f"[OK] LSTM MSE: {lstm_mse:.4f}")

# Save models & scaler for Flask
with open(BASE_DIR / "lr_model.pkl", "wb") as lr_file:
    pickle.dump(lr_model, lr_file)

with open(BASE_DIR / "scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open(BASE_DIR / "df_scaled.pkl", "wb") as scaled_file:
    pickle.dump(df_scaled, scaled_file)

with open(METADATA_FILE, "wb") as metadata_file:
    pickle.dump(
        {
            "symbol": selected_symbol,
            "data_file": DATA_FILE.name,
            "seq_length": seq_length,
            "series_length": int(len(df_scaled)),
        },
        metadata_file,
    )

lstm_model.save(str(BASE_DIR / "lstm_model.h5"))
print("[OK] Models saved!")
