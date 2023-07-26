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
    """
    A class that creates an ARIMA model for time series forecasting.

    :param data: A DataFrame object that contains the data.
    :param metric: The name of the metric to predict.
    """
    def __init__(self, data, metric):
        """
        Initialize the ARIMAModel class with data and a metric.

        :param data: A DataFrame object that contains the data.
        :param metric: The name of the metric to predict.
        """
        self.dataframe = data
        self.metric = metric
        self.values = self.dataframe['values'].values
        self.test_size = 0.2
        self.forecast_eval_size = 0.1
        self.model = None
        self.train, self.test, self.forecast_eval = self.train_test_split()
    
    def train_test_split(self):
        """
        Splits the data into a training set, a test set, and a forecasting evaluation set.
        """
        train_size = int(len(self.values) * (1 - self.test_size - self.forecast_eval_size))
        test_size = int(len(self.values) * (1 - self.forecast_eval_size))
        train, test, forecast_eval = self.values[0:train_size], self.values[train_size:test_size], self.values[test_size:]
        return train, test, forecast_eval
    
    def fit(self, order):
        """
        Fits an ARIMA model to the training data with a given order.

        :param order: A tuple specifying the order of the ARIMA model.
        """
        self.model = ARIMA(self.train, order=order)
        self.model_fit = self.model.fit()
        return self.model_fit

    def predict(self):
        """
        Generates predictions for the test set using the fitted ARIMA model.
        """
        self.predictions = self.model_fit.predict(start=len(self.train), end=len(self.train)+len(self.test)-1)
        return self.predictions

    def evaluate(self):
        """
        Evaluates the model's predictions on the test set using the Mean Squared Error (MSE) metric.

        :return: The Mean Squared Error of the model's predictions on the test set.
        """
        mse = mean_squared_error(self.test, self.predictions)
        return mse

    def tune_hyperparameters(self):
        """
        Finds the best order of the ARIMA model by testing combinations of hyperparameters and choosing the one with the lowest MSE on the test set.
        """
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
        """
        Runs the entire process of the ARIMA model: hyperparameter tuning, model fitting, prediction, evaluation, and forecasting.

        :return: The best order, MSE of the model's predictions, forecasted values, MSE of the forecast, and the path to the plot.
        """
        best_order = self.tune_hyperparameters()
        self.fit(best_order)
        self.predict()
        mse = self.evaluate()
        forecasted_values, forecast_mse = self.evaluate_forecast()
        image_path = self.plot()
        return best_order, mse, forecasted_values, forecast_mse, image_path

    def forecast(self):
        """
        Uses the fitted ARIMA model to generate forecasts for future data points.

        :return: An array of forecasted values.
        """
        if self.model_fit is None:
            raise Exception('The model must be fit before forecasting.')
        forecasted_values = self.model_fit.forecast(steps=len(self.forecast_eval))
        return np.array(forecasted_values)


    def evaluate_forecast(self):
        """
        Evaluates the forecasted values against the actual future data using the Mean Squared Error (MSE) metric.

        :return: The forecasted values and the Mean Squared Error of the forecast.
        """
        forecasted_values = self.forecast()
        mse = mean_squared_error(self.forecast_eval, forecasted_values)
        return forecasted_values, mse

    def plot(self):
        """
        Creates a plot of the original data, the model's predictions on the test set, and the model's forecast for future data.

        :return: The path to the saved plot.
        """
        plt.figure(figsize=(15, 7))

        # Extract date values
        date_values = pd.to_datetime(self.dataframe['date_col'])

        # Create separate arrays for each segment of data (train, test, forecast_eval)
        train_dates = date_values[:len(self.train)]
        test_dates = date_values[len(self.train):len(self.train) + len(self.test)]
        forecast_eval_dates = date_values[-len(self.forecast_eval):]

        # Plot the original data
        plt.plot(train_dates, self.train, label='Train')
        plt.plot(test_dates, self.test, label='Test')
        plt.plot(forecast_eval_dates, self.forecast_eval, label='Future true')

        # Forecast and plot the forecast
        forecasted_values = self.forecast()

        plt.plot(forecast_eval_dates, forecasted_values, label='Forecast')

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
    """
    :param --data: Path to the dataset.
    :param --metric: Name of the metric to predict. If left empty, all metrics will be used.
    """
    parser = argparse.ArgumentParser(description="Feature Engineer AMF data.")
    parser.add_argument("--data", type=str, required=True, help="Path to filtered AMF data")
    parser.add_argument("--metric", type=str, required=True, help="Metric name to filter on. Leave empty for all metrics.")
    args = parser.parse_args()
    dataframe = pd.read_parquet(args.data)
    
    arima_model = ARIMAModel(dataframe, args.metric)
    
    best_order, mse = arima_model.run()

    print(f"Best order: {best_order}")
    print(f"Test MSE: {mse}")

    # Forecast next 5 steps
    forecasted_values = arima_model.forecast()
    print(f"Forecasted values: {forecasted_values}")

    # Evaluate the forecast
    forecast_mse = arima_model.evaluate_forecast()
    print(f"Forecast MSE: {forecast_mse}")

    # Plot the data
    arima_model.plot()
