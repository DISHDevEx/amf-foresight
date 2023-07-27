from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

class ProphetModel:
    """
    Class to create a model for time-series prediction using Facebook Prophet.

    :param dataframe: DataFrame containing 'date_col' and 'values' columns.
    :param metric: The name of the metric to predict.
    """

    def __init__(self, dataframe, metric):
        """
        Initialize Prophet model with dataset and metric.

        :param dataframe: DataFrame to be used for the model.
        :param metric: The name of the metric to be predicted.
        """
        self.df = dataframe[['date_col', 'values']]
        self.metric = metric
        self.df.rename(columns={'date_col': 'ds', 'values': 'y'}, inplace=True)
        self.model = None
        self.model_fit = None
        self.model_params = None

    def train_test_split(self):
        """
        Split the data into training, validation, and test datasets.
        """
        train_size = int(0.7 * len(self.df))
        test_size = int(0.2 * len(self.df))

        self.train_df = self.df[:train_size]
        self.test_df = self.df[train_size:train_size+test_size]
        self.forecast_eval = self.df[train_size+test_size:]

    
    def fit(self):
        """
        Train the Prophet model on the training dataset.
        """
        self.model = Prophet()
        self.model_fit = self.model.fit(self.train_df)
    
    def predict(self):
        """
        Use the trained Prophet model to predict the 'y' values for the test dataset.
        """
        self.predictions = self.model_fit.predict(self.test_df)

    def evaluate(self):
        """
        Evaluate the trained Prophet model using mean squared error (MSE).
        
        :return: MSE of the model's predictions.
        """
        mse = mean_squared_error(self.test_df['y'], self.predictions['yhat'])
        return mse

    def forecast(self):
        """
        Generate forecasts for future dates specified in 'forecast_eval' DataFrame.

        :return: DataFrame with forecasted 'y' values.
        """
        future = self.forecast_eval[['ds']]
        self.forecast_values = self.model_fit.predict(future)
        return self.forecast_values
    
    def evaluate_forecast(self):
        """
        Evaluate the forecasted values against actual values in 'forecast_eval' DataFrame using MSE.

        :return: DataFrame with forecasted 'y' values and MSE of the forecast.
        """

        future = self.forecast_eval[['ds']]
        self.forecast_values = self.model_fit.predict(future)
        mse = mean_squared_error(self.forecast_eval['y'], self.forecast_values['yhat'])
        return self.forecast_values, mse

    def run(self):
        """
        Execute all steps in the model pipeline: data split, model fitting, prediction, evaluation, and forecasting.
        
        :return: MSE of model's predictions, DataFrame with forecasted 'y' values, MSE of the forecast, and path to the plot.
        """
        self.train_test_split()
        self.fit()
        self.predict()
        mse = self.evaluate()
        forecasted_values, forecast_mse = self.evaluate_forecast()
        image_path = self.plot()
        return mse, forecasted_values, forecast_mse, image_path
    
    def plot(self):
        """
        Plot the training, testing, predicted, and forecasted data.

        :return: Path to the saved plot.
        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.train_df['ds'], self.train_df['y'], 'b-', label='Train')
        plt.plot(self.test_df['ds'], self.test_df['y'], 'r-', label='Test')
        plt.plot(self.test_df['ds'], self.predictions['yhat'], 'g-', label='Test prediction')
        plt.plot(self.forecast_values['ds'], self.forecast_values['yhat'], 'k-', label='Forecast')
        plt.plot(self.forecast_eval['ds'], self.forecast_eval['y'], 'm-', label='Actual Forecast')
        plt.legend()
        plt.legend()
        if not os.path.exists("assets"):
            os.makedirs("assets")
        image_path = 'assets/PROPHET' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.png'
        plt.title(str(self.metric))
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.savefig(image_path)
        plt.show()
        return image_path

if __name__ == "__main__":
    """
    :param --data: Path to the dataset.
    :param --metric: Name of the metric to predict.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train data on Prophet")
    parser.add_argument("--data", type=str, required=True, help="Path to filtered AMF data")
    parser.add_argument("--metric", type=str, required=True, help="Metric name to filter on. Leave empty for all metrics.")
    args = parser.parse_args()
    
    dataframe = pd.read_parquet(args.data)
    prophet_model = ProphetModel(dataframe, args.metric)
    mse = prophet_model.run()
    
    forecasted_values, forecast_mse = prophet_model.evaluate_forecast()
    print(f'Test MSE: {mse}')
    print(f'Forecast Values {forecasted_values} MSE: {forecast_mse}')
    prophet_model.plot()
