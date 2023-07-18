from datetime import datetime
from setup_logger import setup_logger
import pandas as pd
import argparse
import logging
import time
import os
import sys

setup_logger()

class FeatureEngineer:
        
    
    def get_data(self, args):
        """
        This function orchestrases the feature engineering process when used as a script
        :param args: arguments passed to the script
        """
        panda = self.read_data(args.data)
        data = self.value_modifier(panda, args.type)
        return data
    
    def read_data(self, path):
        """
        This function takes in a path to a CSV of filtered AMF Data and returns a dataframe
        :param directory: Path to CSV with filtered AMF Data
        """
        data = pd.read_parquet(path)
        return data
    
    def value_modifier(self, data, metric):
        """
        This function takes in a pandas dataframe and modifies the values based on the type of metric
        :param directory: Path to JSON files
        """
        logging.info("Feature Engineering Data")
        logging.info(f"Feature Type={metric}")
        if metric == 'memory':
            logging.info(f"Dividing {metric} by 1048576")
            data['values'] = data['values'] / 1048576
        elif metric == 'cpu':
            logging.info(f"Leaving {metric} as is")
            data['values'] = data['values']
        elif metric == 'cpu_utilization':
            logging.info(f"Calculation {metric} by dividing difference in consecutive difference in metric divided by time elapsed time 100")
            data['time_diff'] = data['date_col'].diff().dt.total_seconds()
            data['usage_diff'] = data['values'].diff()
            data['utilization'] = (data['usage_diff'].diff()/data['time_diff']) * 100
            data.fillna(0, inplace=True)
        return data
            

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Feature Engineer AMF data.")
    parser.add_argument("--data", type=str, required=False, help="Path to filtered AMF data")
    parser.add_argument("--type", type=str, required=False, help="Type of Feature Engineering")
    args = parser.parse_args()
    
    sys.stdout = open('logs/console_output.log', 'w')
    feature_engineer = FeatureEngineer()
    data = feature_engineer.get_data(args)
    
    summary_str = data.describe().to_string().replace('\n', ' | ')
    head_str = data.head().to_string().replace('\n', ' | ')
    
    logging.info("Summary of Requested data:")
    logging.info(summary_str)
    logging.info("First few entries of requested data:")
    logging.info(head_str)

    filename = "sample::" + str(os.path.basename(__file__)) + "::" + args.data.split("::")[-1] + ";type:" + args.type + ";time:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    path = "parquet/" + filename                                                                                 
    data.to_parquet(path, index=False)
    logging.info(f"Data Saved to: {path}")