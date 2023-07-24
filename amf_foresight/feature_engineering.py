from datetime import datetime
from setup_logger import setup_logger
import plotly.express as px
import pandas as pd
import argparse
import logging
import time
import os
import sys
from utils import Utils

setup_logger()

class FeatureEngineer:
        
    def __init__(self):
        self.utils = Utils()
        
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
    
    def plot(self, df, args):
        logging.info("Plotting dataframe")
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
        logging.info(f"(Locally) Saved Plot to {image_path}")
        self.utils.upload_file(image_path, image_path)
            

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