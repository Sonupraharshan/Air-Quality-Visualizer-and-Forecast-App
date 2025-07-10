from flask import Flask, jsonify, request
from forecast_model import load_and_prepare_data, train_and_forecast

app = Flask(__name__)

@app.route("/forecast", methods=["GET"])
def forecast():
    city = request.args.get("city", "Vijayawada")
    pollutant = request.args.get("pollutant", "PM10")

    try:
        df = load_and_prepare_data("data/aqi.csv", city, pollutant)
        forecast_df = train_and_forecast(df)
        output = forecast_df.tail(24).to_dict(orient="records")
        return jsonify({"status": "success", "forecast": output})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
