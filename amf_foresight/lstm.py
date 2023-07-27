import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class LSTMModel:
    """
    Class to create a LSTM based model for time-series prediction.

    :param df: DataFrame containing 'date_col' and 'values' columns.
    :param metric: The name of the metric to predict.
    :param look_back: How many previous time-steps to include in the input features.
    :param steps_ahead: How many time-steps in the future to predict.
    """
    
    def __init__(self, df, metric, look_back=10, steps_ahead=1):
        """
        Initialize LSTM model with dataset, look back period, steps ahead and metric.
        """
        self.look_back = look_back
        self.steps_ahead = steps_ahead
        self.metric = metric
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = df[['date_col', 'values']]
        self.df.set_index('date_col', inplace=True)
        self.original_timestamps = self.df.index.to_list()
        self.original_timestamps = [pd.Timestamp(x).to_pydatetime() for x in self.original_timestamps]
        self.original_data = self.df['values'].tolist()
        self.train = None
        self.val = None
        self.test = None

    def split_data(self):
        """
        Split data into training, test and forecast datasets.
        """
        data = self.df.values
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.2)

        self.scaler.fit(data[:train_size])
        transformed_data = self.scaler.transform(data)

        self.train, self.val, self.test = transformed_data[:train_size], transformed_data[train_size:train_size+val_size], transformed_data[train_size+val_size:]
        self.train_time, self.val_time, self.test_time = self.original_timestamps[:train_size], self.original_timestamps[train_size:train_size+val_size], self.original_timestamps[train_size+val_size:]
        self.train_original, self.val_original, self.test_original = self.original_data[:train_size], self.original_data[train_size:train_size+val_size], self.original_data[train_size+val_size:]

    def create_dataset(self):
        """
        Create input/output datasets for model training.

        :return: X_parts: list of input sequences for train, validation, and test datasets
                 Y_parts: list of output sequences for train, validation, and test datasets
        """
        data_parts = [self.train, self.val, self.test]
        X_parts, Y_parts = [], []
        for part in data_parts:
            X, Y = [], []
            for i in range(len(part) - self.look_back - self.steps_ahead + 1):
                X.append(part[i: i + self.look_back, 0])
                Y.append(part[i + self.look_back: i + self.look_back + self.steps_ahead, 0])
            X_parts.append(np.array(X))
            Y_parts.append(np.array(Y))
        return X_parts, Y_parts

    def reshape_data(self):
        """
        Reshape data to be suitable for LSTM.
        """
        self.X_train, self.X_val, self.X_test = [np.reshape(x, (x.shape[0], 1, x.shape[1])) for x in self.X]
        self.Y_train, self.Y_val, self.Y_test = self.Y  

    def training(self):
        """
        Train the LSTM model using Grid Search for hyperparameter tuning.

        :return: Best parameters from the hyperparameter tuning
        """

        def create_model(neurons=50, optimizer='adam'):
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(1, self.look_back)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            return model

        model = KerasRegressor(build_fn=create_model, verbose=0)

        neurons = [50, 100, 150]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(neurons=neurons, optimizer=optimizer)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(self.X_train, self.Y_train)
        print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

        self.model_fit = grid_result.best_estimator_.model
        
        return grid_result.best_params_
        
    
    def predict_and_plot(self):
        """
        Predict and plot the predicted data along with the original data.

        :return: train_mse: Training mean squared error
                 val_mse: Validation mean squared error
                 test_mse: Test mean squared error
                 image_path: Path of the plotted image
        """
        train_predict = self.model_fit.predict(self.X_train)
        val_predict = self.model_fit.predict(self.X_val)
        test_predict = self.model_fit.predict(self.X_test)

        train_predict = self.scaler.inverse_transform(train_predict.reshape(-1, 1))
        val_predict = self.scaler.inverse_transform(val_predict.reshape(-1, 1))
        test_predict = self.scaler.inverse_transform(test_predict.reshape(-1, 1))

        Y_train_inv = self.scaler.inverse_transform(self.Y_train.reshape(-1, 1))
        Y_val_inv = self.scaler.inverse_transform(self.Y_val.reshape(-1, 1))
        Y_test_inv = self.scaler.inverse_transform(self.Y_test.reshape(-1, 1))

        train_mse = mean_squared_error(Y_train_inv, np.mean(train_predict, axis=1))
        val_mse = mean_squared_error(Y_val_inv, np.mean(val_predict, axis=1))
        test_mse = mean_squared_error(Y_test_inv, np.mean(test_predict, axis=1))
         
        val_predict_flat = val_predict.flatten()
        test_predict_flat = test_predict.flatten()
        
        plt.figure(figsize=(15,7))
        plt.title(str(self.metric))
        
        plt.plot(self.train_time, self.train_original, label='Training data')
        
        plt.scatter(self.val_time, self.val_original, label='Testing Data', color='orange', s=5)
        plt.scatter(self.test_time, self.test_original, label='Forecasting Data', color='green', s=5)
        plt.scatter(self.val_time[self.look_back:len(val_predict)+self.look_back+1], val_predict, s=5, color='purple', label='Testing predictions')
        plt.scatter(self.test_time[self.look_back:len(test_predict)+self.look_back+1], test_predict, s=5, color='brown', label='Forecasting predictions')
        
        
        plt.legend()

        if not os.path.exists("assets"):
            os.makedirs("assets")
        image_path = 'assets/LSTM' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.png'
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.savefig(image_path)
        plt.show()

        return train_mse, val_mse, test_mse, image_path

    def run(self):
        """
        Run all the steps from data split, reshape, training, and prediction.

        :return: best_params: Best parameters from the hyperparameter tuning
                 train_mse: Training mean squared error
                 val_mse: Validation mean squared error
                 test_mse: Test mean squared error
                 image_path: Path of the plotted image
        """

        self.split_data()  
        self.X, self.Y = self.create_dataset() 
        self.reshape_data()

        best_params = self.training()

        train_mse, val_mse, test_mse, image_path = self.predict_and_plot()

        return best_params, train_mse, val_mse, test_mse, image_path


if __name__ == "__main__":
    """
    :param --data: Path to the dataset
    :param --metric: Metric name to filter on.
    """
    parser = argparse.ArgumentParser(description="Train data on LSTM")
    parser.add_argument("--data", type=str, required=True, help="Path to filtered AMF data")
    parser.add_argument("--metric", type=str, required=True, help="Metric name to filter on. Leave empty for all metrics.")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.data, args.metric)
    metric = "memory"
    model = LSTMModel(df)
    hyper, val_mse, test_mse, image_path = model.run()
    print(f"hyper: {hyper}, val_mse: {val_mse}, test_mse: {test_mse}, image_path: {image_path}")
