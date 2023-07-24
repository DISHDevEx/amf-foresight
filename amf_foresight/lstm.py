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

    def __init__(self, df, look_back=10, steps_ahead=1):
        self.look_back = look_back
        self.steps_ahead = steps_ahead
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = df[['date_col', 'values']]
        self.df.set_index('date_col', inplace=True)
        self.train = None
        self.val = None
        self.test = None

    def split_data(self):
        data = self.df.values
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.2)

        # Scale the data
        self.scaler.fit(data[:train_size])
        data = self.scaler.transform(data)

        # Split the data
        self.train, self.val, self.test = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]

    def create_dataset(self):
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
        self.X_train, self.X_val, self.X_test = [np.reshape(x, (x.shape[0], 1, x.shape[1])) for x in self.X]
        self.Y_train, self.Y_val, self.Y_test = self.Y  

    def training(self):
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
        # Predict
        train_predict = self.model_fit.predict(self.X_train)
        val_predict = self.model_fit.predict(self.X_val)
        test_predict = self.model_fit.predict(self.X_test)

        # Inverse scaling
        train_predict = self.scaler.inverse_transform(train_predict.reshape(-1, 1))
        val_predict = self.scaler.inverse_transform(val_predict.reshape(-1, 1))
        test_predict = self.scaler.inverse_transform(test_predict.reshape(-1, 1))

        Y_train_inv = self.scaler.inverse_transform(self.Y_train.reshape(-1, 1))
        Y_val_inv = self.scaler.inverse_transform(self.Y_val.reshape(-1, 1))
        Y_test_inv = self.scaler.inverse_transform(self.Y_test.reshape(-1, 1))

        # Compute MSE
        train_mse = mean_squared_error(Y_train_inv, np.mean(train_predict, axis=1))
        val_mse = mean_squared_error(Y_val_inv, np.mean(val_predict, axis=1))
        test_mse = mean_squared_error(Y_test_inv, np.mean(test_predict, axis=1))

        print(f"Validation MSE: {val_mse}")
        print(f"Test MSE: {test_mse}")

        # Create a figure
        plt.figure(figsize=(15,5))
        plt.plot(Y_train_inv, label='Training data')
        plt.plot([x for x in range(len(self.Y_train), len(self.Y_train)+len(Y_val_inv))], Y_val_inv, label='Testing Data')
        plt.plot([x for x in range(len(self.Y_train)+len(Y_val_inv), len(self.Y_train)+len(Y_val_inv)+len(Y_test_inv))], Y_test_inv, label='Forecasting Data')

        # Since our predictions are for multiple steps ahead, we should take the mean for the plot
        plt.plot(np.mean(train_predict, axis=1), label='Training predictions')
        plt.plot([x for x in range(len(train_predict), len(train_predict)+len(val_predict))], np.mean(val_predict, axis=1), label='Testing predictions')
        plt.plot([x for x in range(len(train_predict)+len(val_predict), len(self.Y_train)+len(Y_val_inv)+len(test_predict))], np.mean(test_predict, axis=1), label='Forecasting predictions')

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
        # Prepare the data
        self.split_data()  # Split data first
        self.X, self.Y = self.create_dataset()  # Create dataset after splitting
        self.reshape_data()

        # Train the model and get the best hyperparameters
        best_params = self.training()

        # Predict and plot the data
        train_mse, val_mse, test_mse, image_path = self.predict_and_plot()

        return best_params, train_mse, val_mse, test_mse, image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train data on LSTM")
    parser.add_argument("--data", type=str, required=True, help="Path to filtered AMF data")
    # parser.add_argument("--metric", type=str, required=True, help="Metric name to filter on. Leave empty for all metrics.")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.data)
    metric = "memory"
    model = LSTMModel(df)
    hyper, val_mse, test_mse, image_path = model.run()
    print(f"hyper: {hyper}, val_mse: {val_mse}, test_mse: {test_mse}, image_path: {image_path}")
