from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

class ProphetModel:

    def __init__(self, dataframe, metric):
        """
        Initialization method for the ProphetModel class.
        This method loads the dataset and renames the columns.
        """
        self.df = dataframe[['date_col', 'values']]
        self.metric = metric
        self.df.rename(columns={'date_col': 'ds', 'values': 'y'}, inplace=True)
        self.model = None
        self.model_fit = None
        self.model_params = None

    def train_test_split(self):
        """
        Splits the dataset into training, testing and forecasting datasets.
        """
        train_size = int(0.7 * len(self.df))
        test_size = int(0.2 * len(self.df))

        self.train_df = self.df[:train_size]
        self.test_df = self.df[train_size:train_size+test_size]
        self.forecast_eval = self.df[train_size+test_size:]

    
    def fit(self):
        """
        Fits the Prophet model on the training dataset.
        """
        self.model = Prophet()
        self.model_fit = self.model.fit(self.train_df)
    
    def predict(self):
        """
        Predicts on the testing dataset using the trained Prophet model.
        """
        self.predictions = self.model_fit.predict(self.test_df)

    def evaluate(self):
        """
        Evaluates the model's predictions on the testing dataset.
        Returns the mean squared error of the model's predictions.
        """
        mse = mean_squared_error(self.test_df['y'], self.predictions['yhat'])
        return mse

    def forecast(self):
        """
        Creates a future DataFrame with the same timestamps as in self.forecast_eval
        and forecasts the target variable for these dates.
        """
        future = self.forecast_eval[['ds']]
        self.forecast_values = self.model_fit.predict(future)
        return self.forecast_values
    
    def evaluate_forecast(self):
        """
        Evaluates the model's forecast on self.forecast_eval.
        Returns the mean squared error of the model's forecast.
        """
        future = self.forecast_eval[['ds']]
        self.forecast_values = self.model_fit.predict(future)
        mse = mean_squared_error(self.forecast_eval['y'], self.forecast_values['yhat'])
        return self.forecast_values, mse

    def run(self):
        """
        This method encapsulates the entire workflow of the model:
        data split, model fitting, prediction, evaluation, and forecasting.
        Returns the mean squared errors of the model's predictions and its forecast.
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
        Plots the model's predictions and its forecast using Prophet's built-in plot method.
        """
        # fig1 = self.model.plot(self.forecast)
        plt.figure(figsize=(10, 8))
        
        # Plotting training data
        plt.plot(self.train_df['ds'], self.train_df['y'], 'b-', label='Train')

        # Plotting test data
        plt.plot(self.test_df['ds'], self.test_df['y'], 'r-', label='Test')

        # Plotting test predictions
        plt.plot(self.test_df['ds'], self.predictions['yhat'], 'g-', label='Test prediction')

        # Plotting forecasted values
        plt.plot(self.forecast_values['ds'], self.forecast_values['yhat'], 'k-', label='Forecast')

        # Plotting actual values in forecast set
        plt.plot(self.forecast_eval['ds'], self.forecast_eval['y'], 'm-', label='Actual Forecast')

        plt.legend()  # Display the legend
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
