import pandas as pd
from load_data import AMFDataProcessor
from prophet_model import ProphetModel
from datetime import datetime
from setup_logger import setup_logger
from feature_engineering import FeatureEngineer
from arima import ARIMAModel
from autoregression import AutoRegressionModel
from lstm import LSTMModel
from utils import Utils
import argparse
import logging
import time 
import os
import sys
import pickle
import inspect
setup_logger()

class Orchestrator:
    """
    Orchestrator Class:
    A class that handles the orchestration of preprocessing, training and evaluating AMF data.
    """
    def __init__(self):
        self.args = args
        self.processor = AMFDataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.utils = Utils()
        self.data = None
        self.selected_model = None
        self.model = None
    
    def preprocessing(self, args):
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Preprocessing Data..")
        raw = self.processor.run(args)
        processed = self.feature_engineer.value_modifier(raw, args.type)
        self.feature_engineer.plot(processed, args)
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Preprocessed Data.")
        return processed
    
    def train(self, args):
        self.selected_model = args.model
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Selected model is {self.selected_model}")
        hyper = None
        if self.selected_model == 'ARIMA':
            self.model = ARIMAModel(self.data, args.metric)
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Tuning hyperparamers..")
            hyper, mse, forecasted_values, forecast_mse, image_path = self.model.run()
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Best hyperparameters for this model: {hyper} Test MSE: {mse}")
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Forecasted Values {forecasted_values} Forecast MSE: {forecast_mse}")
        elif self.selected_model == 'PROPHET':
            self.model = ProphetModel(self.data, args.metric)
            mse, forecasted_values, forecast_mse, image_path = self.model.run()
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Test MSE: {mse}")
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Forecasted Values {forecasted_values} Forecast MSE: {forecast_mse}")
        elif self.selected_model == 'LSTM':
            self.model = LSTMModel(self.data, args.metric)
            hyper, train_mse, mse, forecast_mse, image_path = self.model.run()
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Train MSE: {train_mse} Test MSE: {mse} Forecast MSE: {forecast_mse}")       
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::(Locally) Saved Plot: {image_path}")
        self.utils.upload_file(image_path, image_path)
        if not os.path.exists("models"):
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Creating folder: models")
            os.makedirs("models")
        model_file_path = "models/" + self.selected_model.lower() + ";params:" + str(hyper) + ";time:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        with open(model_file_path, 'wb') as model_file:
            pickle.dump(self.model.model_fit, model_file)
        self.utils.upload_file(model_file_path, model_file_path)
    
    def save(self, processed):
        summary_str = processed.describe().to_string().replace('\n', ' | ')
        head_str = processed.head().to_string().replace('\n', ' | ')
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Summary of Requested data:")
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::{summary_str}")
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::First few entries of requested data:")
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::{head_str}")
        filename = "sample::" + os.path.basename(__file__) + "::metric:" + str(args.metric) + ";pod:" + str(args.pod) + ";level:" + str(args.level) + ";start:" + args.start + ";end:" + args.end
        if not os.path.exists("parquet"):
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Creating folder: models")
            os.makedirs("parquet")
        path = "parquet/" + filename                                                                                 
        processed.to_parquet(path, index=False)
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::(Locally) Data Saved to: {path}")
        self.utils.upload_file(path, path)
        return path
        
     
    def run(self, args):
        """
        This function runs the entire pipeline 
        :param
        """
        time1 = time.time()
        processed = None
        if any([args.download, args.process, args.generate]):
            processed = self.preprocessing(args)
        if isinstance(processed, pd.DataFrame):
            self.data = processed
            path = self.save(self.data)
        if args.train:
            if isinstance(processed, pd.DataFrame):
                self.data = processed
            else:
                self.data = pd.read_parquet(args.data)
            self.train(args)
         
            
            
            
            
        
        
        
        
        
    
    
if __name__ == "__main__":    
    
    
    parser = argparse.ArgumentParser(description="Process some AMF data.")
    parser.add_argument("--download", action='store_true', help="Include this flag to download chunks. --download does not require any argument")
    parser.add_argument("--process", action='store_true', help="Include this flag to process chunks into JSON format. --process requires --start, and --end.")
    parser.add_argument("--generate", action='store_true', help="Include this flag to generate the data frame and save the data as a paraquet file. --generate requires --metric, --type and --level.")
    parser.add_argument("--train", action='store_true', help="Include this flag to train your data. --train requires --generate and --model or --data and --model")
    
    parser.add_argument("--data", type=str, help="Path where the processed CSV are/should be stored as a paraquet file. These files are generated from the JSONs.")
    parser.add_argument("--start", type=str, help="Start time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--end", type=str, help="End time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--level", type=str, help="Container level to filter on. Could be 'amf', 'support', 'upperlimit', 'amf+support' ")
    parser.add_argument("--type", type=str, help="Type of feature you want. Could be 'memory', 'cpu' or 'utilization'")
    parser.add_argument("--metric", type=str, help="Metric name to filter on. Leave empty for all metrics.")
    parser.add_argument("--pod", type=str, help="Pod name to filter on. Leave empty for all pods.")    
    parser.add_argument("--model", type=str, help="Model you would like to use.")
    parser.add_argument("--steps", type=int, help="Number of timesteps you want to forecast")
    
    
    args = parser.parse_args()
    
    
    if not any([args.download, args.process, args.generate, args.train]):
        parser.error("One of --download, --process, --generate, --train must be provided.")
    if args.process and not all([args.start, args.end]):
        parser.error("--process requires --start and --end")
    if args.generate and not all([args.level, args.type, args.metric]):
        parser.error("--generate requires --metric, --type and --level.")
    if args.train and not ((all([args.generate, args.model])) or (all([args.data, args.model]))):
        parser.error("--train requires --data and --model or --generate and --model")
       
    orchestra = Orchestrator()
    orchestra.run(args)
    
    
   
    
    