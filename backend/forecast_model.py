import pandas as pd
from prophet import Prophet

def load_and_prepare_data(filepath, city="Vijayawada", pollutant="PM10"):
    df = pd.read_csv(filepath)
    df = df[(df["city"] == city) & (df["pollutant_id"] == pollutant)].copy()
    df["last_update"] = pd.to_datetime(df["last_update"], format="%d-%m-%Y %H:%M:%S", errors='coerce')
    df = df.dropna(subset=["last_update", "pollutant_avg"])
    df = df.sort_values("last_update")
    df = df.rename(columns={"last_update": "ds", "pollutant_avg": "y"})
    return df[["ds", "y"]]

def train_and_forecast(df, periods=24):
    model = Prophet(weekly_seasonality=True, daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
