from datetime import datetime
from setup_logger import setup_logger
import plotly.express as px
import pandas as pd
import argparse
import logging
import time
import os
import sys
import inspect
from utils import Utils

setup_logger()

class FeatureEngineer:
    """
    A class to carry out feature engineering on AMF data. This class provides methods to read the data, 
    modify its values based on the type of metric, and plot the processed data. It also enables fetching 
    the processed data.
    """
    
    def __init__(self):
        """
        Initialize FeatureEngineer and create an instance of Utils class.
        """
        self.utils = Utils()
        
    def get_data(self, args):
        """
        Orchestrates the feature engineering process.

        :param args: Arguments passed to the script.
        :return: DataFrame after the feature engineering process.
        """
        panda = self.read_data(args.data)
        data = self.value_modifier(panda, args.type)
        return data
    
    def read_data(self, path):
        """
        Reads a parquet file of filtered AMF data and returns a DataFrame.

        :param path: Path to parquet file with filtered AMF Data.
        :return: DataFrame from the file.
        """
        data = pd.read_parquet(path)
        return data
    
    def value_modifier(self, data, metric):
        """
        Modifies the values in a DataFrame based on the type of metric.

        :param data: DataFrame with AMF data.
        :param metric: Type of feature engineering to be performed.
        :return: DataFrame with modified values.
        """
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Feature Engineering Data")
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Feature Type={metric}")
        if metric == 'memory':
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Dividing {metric} by 1048576")
            data['values'] = data['values'] / 1048576
        elif metric == 'cpu':
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Leaving {metric} as is")
            data['values'] = data['values']
        elif metric == 'cpu_utilization':
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Calculation {metric} by dividing difference in consecutive difference in metric divided by time elapsed time 100")
            data['time_diff'] = data['date_col'].diff().dt.total_seconds()
            data['usage_diff'] = data['values'].diff()
            data['utilization'] = (data['usage_diff'].diff()/data['time_diff']) * 100
            data.fillna(0, inplace=True)
        return data
    
    def plot(self, df, args):
        """
        Plots the DataFrame and saves the plot locally and in cloud storage.

        :param df: DataFrame to be plotted.
        :param args: Arguments related to the plot.
        """
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Plotting dataframe")
        fig = px.line(df, x='date_col', y='values', color='container')
        fig.update_layout(
            title=args.metric,
            xaxis=dict(title='date_col'),
            yaxis=dict(title='values')
        )
        fig.show()
        if not os.path.exists("assets"):
            os.makedirs("assets")
        image = "plot::" + os.path.basename(__file__) + "::metric:" + str(args.metric) + ";pod:" + str(args.pod) + ";level:" + str(args.level) + ";start:" + args.start + ";end:" + args.end + ".png"
        image_path = os.path.join("assets", image)
        fig.write_image(image_path, width=900, height=600)
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::(Locally) Saved Plot to {image_path}")
        self.utils.upload_file(image_path, image_path)
            

if __name__ == "__main__": 
    """
    :param args: Command line arguments parsed by argparse.ArgumentParser. It should include:
        data: The path to the filtered AMF data.
        type: The type of feature engineering to be applied.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Feature Engineer AMF data.")
    parser.add_argument("--data", type=str, required=False, help="Path to filtered AMF data")
    parser.add_argument("--type", type=str, required=False, help="Type of Feature Engineering")
    args = parser.parse_args()
    
    feature_engineer = FeatureEngineer()
    data = feature_engineer.get_data(args)
    
    summary_str = data.describe().to_string().replace('\n', ' | ')
    head_str = data.head().to_string().replace('\n', ' | ')
    
    logging.info(f"{os.path.basename(__file__)}::Summary of Requested data:")
    logging.info(f"{os.path.basename(__file__)}::{summary_str}")
    logging.info(f"{os.path.basename(__file__)}::First few entries of requested data:")
    logging.info(f"{os.path.basename(__file__)}::{head_str}")

    filename = "sample::" + str(os.path.basename(__file__)) + "::" + args.data.split("::")[-1] + ";type:" + args.type + ";time:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    path = "parquet/" + filename                                                                                 
    data.to_parquet(path, index=False)
    logging.info(f"{os.path.basename(__file__)}::Data Saved to: {path}")