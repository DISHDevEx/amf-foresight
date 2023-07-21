import pandas as pd
import numpy as np
import argparse
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import datetime
import time
import matplotlib.pyplot as plt



class ARIMAModel:
    def __init__(self, data, metric):
        self.dataframe = data
        self.metric = metric
        self.values = self.dataframe['values'].values
        self.test_size = 0.2
        self.forecast_eval_size = 0.1
        self.model = None
        self.train, self.test, self.forecast_eval = self.train_test_split()
    
    def train_test_split(self):
        train_size = int(len(self.values) * (1 - self.test_size - self.forecast_eval_size))
        test_size = int(len(self.values) * (1 - self.forecast_eval_size))
        train, test, forecast_eval = self.values[0:train_size], self.values[train_size:test_size], self.values[test_size:]
        return train, test, forecast_eval
    
    def fit(self, order):
        self.model = ARIMA(self.train, order=order)
        self.model_fit = self.model.fit()
        return self.model_fit

    def predict(self):
        self.predictions = self.model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1)
        return self.predictions

    def evaluate(self):
        mse = mean_squared_error(self.test, self.predictions)
        return mse

    def tune_hyperparameters(self):
        p_values = [1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        best_mse, best_order = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        self.fit(order)
                        predictions = self.predict()
                        mse = mean_squared_error(self.test, predictions)
                        if mse < best_mse:
                            best_mse, best_order = mse, order
                    except:
                        continue
        return best_order

    def run(self):
        best_order = self.tune_hyperparameters()
        self.fit(best_order)
        self.predict()
        mse = self.evaluate()
        return best_order, mse

    def forecast(self, steps):
        if self.model_fit is None:
            raise Exception('The model must be fit before forecasting.')
        forecasted_values = self.model_fit.forecast(steps=steps)
        return forecasted_values

    def evaluate_forecast(self, steps):
        forecasted_values = self.forecast(steps)
        mse = mean_squared_error(self.forecast_eval[:steps], forecasted_values)
        return forecasted_values, mse

    def plot(self, forecast_steps):
        # Extract date values
        date_values = pd.to_datetime(self.dataframe['date_col'])

        # Create separate arrays for each segment of data (train, test, forecast_eval)
        train_dates = date_values[:len(self.train)]
        test_dates = date_values[len(self.train):len(self.train) + len(self.test)]
        forecast_eval_dates = date_values[len(self.train) + len(self.test):][:forecast_steps]

        plt.figure(figsize=(15, 7))

        # Plot the original data
        plt.plot(train_dates, self.train, label='Train')
        plt.plot(test_dates, self.test, label='Test')
        plt.plot(forecast_eval_dates, self.forecast_eval[:forecast_steps], label='Future true')

        # Forecast and plot the forecast
        forecasted_values = self.forecast(forecast_steps)

        plt.plot(forecast_eval_dates[:forecast_steps], forecasted_values[:forecast_steps], label='Forecast')

        plt.legend()
        if not os.path.exists("assets"):
            os.makedirs("assets")
        image_path = 'assets/ARIMA' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.png'
        plt.title(str(self.metric))
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.savefig(image_path)
        plt.show()
        return image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineer AMF data.")
    parser.add_argument("--data", type=str, required=False, help="Path to filtered AMF data")
    args = parser.parse_args()
    dataframe = pd.read_parquet(args.data)
    
    arima_model = ARIMAModel(dataframe)
    
    best_order, mse = arima_model.run()

    print(f"Best order: {best_order}")
    print(f"Test MSE: {mse}")

    # Forecast next 5 steps
    forecasted_values = arima_model.forecast(22)
    print(f"Forecasted values: {forecasted_values}")

    # Evaluate the forecast
    forecast_mse = arima_model.evaluate_forecast(22)
    print(f"Forecast MSE: {forecast_mse}")

    # Plot the data
    arima_model.plot(22)
    
