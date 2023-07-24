import numpy as np
import matplotlib.pyplot as plt
import os
import time
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
        self.df = self.df.values
        self.X, self.Y = self.create_dataset()
        self.model_fit = None

    def create_dataset(self):
        data = self.scaler.fit_transform(self.df)
        X, Y = [], []
        for i in range(len(data) - self.look_back - self.steps_ahead + 1):
            X.append(data[i: i + self.look_back, 0])
            Y.append(data[i + self.look_back: i + self.look_back + self.steps_ahead, 0])
        return np.array(X), np.array(Y)

    def split_data(self):
        train_size = int(len(self.X) * 0.7)
        val_size = int(len(self.X) * 0.15)
        self.X_train, self.Y_train = self.X[:train_size], self.Y[:train_size]
        self.X_val, self.Y_val = self.X[train_size:train_size+val_size], self.Y[train_size:train_size+val_size]
        self.X_test, self.Y_test = self.X[train_size+val_size:], self.Y[train_size+val_size:]

    def reshape_data(self):
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], 1, self.X_val.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
        

    def train(self):
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
        train_predict = self.model_fit.predict(self.X_train)
        val_predict = self.model_fit.predict(self.X_val)
        test_predict = self.model_fit.predict(self.X_test)

        train_predict = self.scaler.inverse_transform(train_predict)
        val_predict = self.scaler.inverse_transform(val_predict)
        test_predict = self.scaler.inverse_transform(test_predict)  # Removed reshaping

        # Compute MSE for Validation and Test predictions
        Y_val_inv = self.scaler.inverse_transform(self.Y_val)  # Inverse transform once
        Y_test_inv = self.scaler.inverse_transform(self.Y_test)  # Inverse transform once

        print(f"valpred: {val_predict}")
        print(f"yval: {Y_val_inv}")
        val_mse = mean_squared_error(Y_val_inv, val_predict)
        test_mse = mean_squared_error(Y_test_inv, test_predict)

        print(f"Validation MSE: {val_mse}")
        print(f"Test MSE: {test_mse}")

        plt.figure(figsize=(15,5))
        plt.plot(self.scaler.inverse_transform(self.Y_train), label='Training data')
        plt.plot([x for x in range(len(self.Y_train), len(self.Y_train)+len(Y_val_inv))], Y_val_inv, label='Validation data')
        plt.plot([x for x in range(len(self.Y_train)+len(Y_val_inv), len(self.Y_train)+len(Y_val_inv)+len(Y_test_inv))], Y_test_inv, label='Test data')

        plt.plot([x for x in range(self.look_back,len(train_predict)+self.look_back)], train_predict, label='Training predictions')
        plt.plot([x for x in range(len(train_predict)+self.look_back,len(train_predict)+len(val_predict)+self.look_back)], val_predict, label='Validation predictions')
        plt.plot([x for x in range(len(train_predict)+len(val_predict)+self.look_back,len(self.Y_train)+len(Y_val_inv)+len(test_predict)+self.look_back)], test_predict, label='Test predictions')

        plt.legend()

        if not os.path.exists("assets"):
            os.makedirs("assets")
        image_path = 'assets/LSTM' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.png'
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.savefig(image_path)
        plt.show()
        return val_mse, test_mse, image_path




    
    def run(self):
        """
        Run the main steps.
        Returns:
        None
        """
        self.split_data()
        self.reshape_data()
        hyper = self.train()
        val_mse, test_mse, image_path = self.predict_and_plot()
        return hyper, val_mse, test_mse, image_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train data on LSTM")
    parser.add_argument("--data", type=str, required=True, help="Path to filtered AMF data")
    # parser.add_argument("--metric", type=str, required=True, help="Metric name to filter on. Leave empty for all metrics.")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.data)
    metric = "memory"
    model = LSTMModel(df)
    hyper, val_mse, test_mse, image_path = model.run()
    print(f"hyper: {hyper}, val_mse: {val_mse}, test_mse: {test_mse}, image_path: {image_path}")
