import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

class LSTMModel:

    def __init__(self, df, metric, look_back=10):
        """
        Initialize the LSTMModel class with the path to the CSV file containing the time series data.
        """
        self.df = df
        self.metric = metric
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.hyper = None
        self.look_back = look_back

    def load_and_preprocess_data(self):
        """
        Load and preprocess the time series data.

        Returns:
        None
        """
        self.df = self.df.set_index('date_col')
        self.scaled_data = self.scaler.fit_transform(self.df.values.reshape(-1, 1))

        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        self.X, self.Y = create_dataset(self.scaled_data, look_back=self.look_back)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))

    def split_data(self):
        """
        Split the data into training, validation, and forecast sets.

        Returns:
        None
        """
        # Splitting the data into train, validation and forecast
        train_size = int(len(self.X) * 0.70)
        val_size = int(len(self.X) * 0.20)
        self.X_train, self.Y_train = self.X[:train_size], self.Y[:train_size]
        self.X_val, self.Y_val = self.X[train_size:train_size+val_size], self.Y[train_size:train_size+val_size]
        self.X_forecast, self.Y_forecast = self.X[train_size+val_size:], self.Y[train_size+val_size:]

    def train_and_predict(self):
        """
        Train the LSTM model and make predictions.
        """

        # Function to create model, required for KerasRegressor
        def create_model(neurons=50, optimizer='adam'):
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(self.look_back, 1)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            return model

        # create model
        model = KerasRegressor(build_fn=create_model, verbose=0)

        # grid search epochs, batch size and optimizer
        neurons = [50, 100, 150]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(neurons=neurons, optimizer=optimizer)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid_result = grid.fit(np.concatenate((self.X_train, self.X_val)), np.concatenate((self.Y_train, self.Y_val)))
        
        self.hyper = grid_result.best_params_
        # Train the model with best hyperparameters
        best_neuron = grid_result.best_params_['neurons']
        best_optimizer = grid_result.best_params_['optimizer']

        model = create_model(neurons=best_neuron, optimizer=best_optimizer)
        model.fit(np.concatenate((self.X_train, self.X_val)), np.concatenate((self.Y_train, self.Y_val)), epochs=50, batch_size=72, verbose=2, shuffle=False)

        # Make predictions
        train_predict = model.predict(self.X_train)
        val_predict = model.predict(self.X_val)

        # Invert predictions
        train_predict = self.scaler.inverse_transform(train_predict)
        self.Y_train = self.scaler.inverse_transform([self.Y_train])
        val_predict = self.scaler.inverse_transform(val_predict)
        self.Y_val = self.scaler.inverse_transform([self.Y_val])
        
        # Calculate MSE for the validation/test data
        mse_test = mean_squared_error(self.Y_val[0], val_predict[:,0])
        
        self.model = model
        self.train_predict = train_predict
        self.val_predict = val_predict
        self.mse_test = mse_test
        

    def forecast(self):
        """
        Make a forecast using the trained LSTM model.

        Returns:
        None
        """
        self.forecast_predict = self.model.predict(self.X_forecast)
        self.forecast_predict = self.scaler.inverse_transform(self.forecast_predict)
        self.Y_forecast = self.scaler.inverse_transform([self.Y_forecast])
        
        # Calculate forecast MSE
        self.forecast_mse = mean_squared_error(self.Y_forecast[0], self.forecast_predict[:,0])

    def plot_predictions(self):
        """
        Plot the original time series data and the LSTM model's predictions.

        Returns:
        None
        """
        # Plotting original data
        plt.figure(figsize=(20,10))
        plt.plot(self.scaler.inverse_transform(self.scaled_data), color='blue')

        # Plotting training predictions
        plt.plot([x for x in range(self.look_back, len(self.train_predict)+self.look_back)], self.train_predict, color='red')

        # Plotting validation predictions
        plt.plot([x for x in range(len(self.train_predict)+(self.look_back*2), len(self.train_predict)+len(self.val_predict)+(self.look_back*2))], self.val_predict, color='green')

        # Plotting forecast predictions
        plt.plot([x for x in range(len(self.train_predict)+len(self.val_predict)+(self.look_back*3), len(self.train_predict)+len(self.val_predict)+len(self.forecast_predict)+(self.look_back*3))], self.forecast_predict, color='orange')

        if not os.path.exists("assets"):
            os.makedirs("assets")
        image_path = 'assets/LSTM' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.png'
        plt.title(str(self.metric))
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.savefig(image_path)
        plt.show()
        return image_path

    def run(self):
        """
        Run the main steps.

        Returns:
        None
        """
        self.load_and_preprocess_data()
        self.split_data()
        self.train_and_predict()
        self.forecast()
        self.plot_predictions()

if __name__ == "__main__":
    df = pd.read_parquet('parquet/sample::pipeline.py::metric:container_memory_usage_bytes;pod:None;level:amf;start:2023-06-15 00:00:00;end:2023-06-15 04:00:00')
    metric = "memory"
    model = LSTMModel(df, metric)
    model.run()
