from devex_sdk import Spark_Data_Connector, Nested_Json_Connector
from pyspark.sql.types import FloatType, TimestampType, LongType
from pyspark.sql import SparkSession
from datetime import datetime
from setup_logger import setup_logger
from collections import defaultdict
from utils import Utils
import plotly.express as px
import pyspark.sql.functions as F
import pandas as pd
import time
import csv
import os
import json
import boto3
import itertools
import pyspark
import logging
import argparse
import subprocess
import shutil
import sys
import inspect
setup_logger()

class AMFDataProcessor:
    """
    A class to process AMF data. It downloads chunks of data, processes it and generates a data frame 
    from the processed data. It also provides functionalities to plot data and get data for specific 
    conditions.
    """
    
    def __init__(self):
        """
        Initialize AMFDataProcessor and create an instance of Utils class.
        """
        self.utils = Utils()
    
    def clear_folders(self, folders):
        """
        Clear all files in the given folders. If a folder does not exist, create it.

        :param folders: List of folders to be cleared.
        """
        for folder in folders:
            if os.path.exists(folder):
                logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Clearing files in folder: {folder}")
                files = os.listdir(folder)
                for file in files:
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            else:
                logging.info(f"Creating folder: {folder}")
                os.makedirs(folder)
    
    def run(self, args):
        """
        The main function to download, process and generate dataframe. 

        :param args: Arguments passed through command line
        """
        time1 = time.time()
        if args.download:
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Downloading chunks..")
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Running with arguments for download: None")
            self.clear_folders(["chunks"])
            self.download_chunks("chunks")
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Downloaded chunks.")

        if args.process:
            self.clear_folders(["jsons"])
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Processing chunks..")
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Running with arguments for process: start={args.start}, end={args.end}")
            self.run_go("chunks", "jsons", args.start, args.end)
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Processed chunks.")

        if args.generate:
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Generating dataframe..")
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Running with arguments for generate: metric={args.metric}, level={args.level}, pod={args.pod}")
            spark, panda = self.get_data("jsons", args.level, args.metric, args.pod)
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Generated dataframe")
            logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Plotting dataframe")
            return panda
    
    def plot(self, df, args):
        """
        Plot the dataframe and save the plot as a png file.

        :param df: The dataframe to plot.
        :param args: The arguments passed from command line.
        """
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
        fig.write_image(image_path, width=800, height=600)
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::(Locally) Saved Plot to {image_path}")
        self.utils.upload_file(image_path, image_path)
        
        
    
    def get_data(self, directory, container_level="all", metric=None, pod=None):
        """
        Get data from the directory, transform the dataframe and convert it into pandas dataframe.

        :param directory: Directory from where to fetch data.
        :param container_level: Level of the container. Default is "all".
        :param metric: The metric for which data is to be fetched. Default is None.
        :param pod: The pod for which data is to be fetched. Default is None.
        """
        spark_dataframes = self.get_dataframes(directory, container_level)
        spark_dataframe = self.transform_dataframe(spark_dataframes, metric, pod)
        pandas_dataframe = self.get_values(spark_dataframe, container_level)
        return spark_dataframe, pandas_dataframe

    def get_dataframes(self, directory, container_level):
        """
        Get dataframes from all the json files in the directory.

        :param directory: Directory from where to fetch data.
        :param container_level: Level of the container.
        """
        flag = False
        for i, filename in enumerate(os.listdir(directory)):
            if os.path.isfile(os.path.join(directory, filename)) and not flag:
                appended_df = self.get_amf_data(os.path.join(directory, filename), container_level)
                logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Generated Dataframe for: {os.path.join(directory, filename)}")
                flag = True
            elif flag:
                if os.path.isfile(os.path.join(directory, filename)):
                    new_df = self.get_amf_data(os.path.join(directory, filename), container_level)
                    logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Generated Dataframe for: {os.path.join(directory, filename)}")
                    appended_df = appended_df.union(new_df)
        return appended_df

    def transform_dataframe(self, amf_data, metric_name= None, pod_name=None):
        """
        Filter the dataframe based on given metric_name and pod_name. 

        :param amf_data: The AMF data to be transformed.
        :param metric_name: The metric name to filter on. Default is None.
        :param pod_name: The pod name to filter on. Default is None.
        """
        amf_data = amf_data.withColumn("date_col", F.expr("transform(timestamps, x -> from_unixtime(x/1000))"))
        if metric_name:
            amf_data = amf_data.filter(F.col('metric___name__') == metric_name)
        if pod_name:
            amf_data = amf_data.filter(F.col("metric_pod").startswith(pod_name))
        return amf_data
    
    def get_all_data(self, json_object_path):
        """
        Get all data from a given json object path.

        :param json_object_path: The path of the json object.
        """
        obj = Nested_Json_Connector(json_object_path)
        err, data = obj.read_nested_json()
        data = data.filter(F.col('metric_pod').startswith('open5gs'))
        return data
    
    def get_all_amf_data(self, json_object_path):
        """
        Get all AMF data from a given json object path.

        :param json_object_path: The path of the json object.
        """
        obj = Nested_Json_Connector(json_object_path)
        err, data = obj.read_nested_json()
        data = data.filter(F.col('metric_pod').startswith('open5gs-amf'))
        return data
    
    def get_amf_data(self, json_object_path, container_level="all"):
        """
        Get AMF data from a given json object path and filter data based on the container level.

        :param json_object_path: The path of the json object.
        :param container_level: Level of the container. Default is "all".
        """
        obj = Nested_Json_Connector(json_object_path, setup = "32gb")
        err, data = obj.read_nested_json()
        data = data.select('timestamps', 'metric___name__', 'values', 'metric_pod', 'metric_container', 'metric_name', 'metric_image', 'metric_id', 'metric_namespace')
        data = data.filter(F.col("metric_namespace") == "openverso")
        data = data.filter(F.col('metric_pod').startswith('open5gs-amf'))
        if container_level == 'amf':
            data = data.filter(F.col('metric_container')=='open5gs-amf')
        elif container_level == 'upperlimit':
            data = data.filter(F.col('metric_container').isNull() & F.col('metric_image').isNull())
        elif container_level == 'support':
            data = data.filter(F.col('metric_container').isNull() & F.col('metric_image').isNotNull())
        elif container_level == 'amf+support':
            data = data.filter(F.col('metric_name').isNotNull())
        return data

    def get_min_value(self, amf_data):
        """
        Get the minimum value from the given AMF data.

        :param amf_data: The AMF data to find the minimum value from.
        """
        min_vals = amf_data.filter(F.col("metric_container").isNull() & F.col('metric_image').isNotNull()).select("values").rdd.flatMap(lambda x: x).collect()
        min_val = None
        if min_vals:
            min_val = min_vals[0][0]
        return min_val

    def get_values(self, data, container_level):
        """
        Get values from the data and convert it into a pandas dataframe. 

        :param data: The data to get values from.
        :param container_level: Level of the container.
        """
        min_val = None
        max_val = None
        if container_level == "amf+support":
            min_val = self.get_min_value(data)
            data = data.filter(F.col('metric_container')=='open5gs-amf')
        
            
        x_values = data.select("date_col").rdd.flatMap(lambda x: x).collect()
        y_values = data.select("values").rdd.flatMap(lambda x: x).collect()
        z_values = data.select("metric_name").rdd.flatMap(lambda x: x).collect()

        x_flat = [item for sublist in x_values for item in sublist]
        y_flat = [item for sublist in y_values for item in sublist]
        z_repeat = [label for sublist, label in zip(x_values, z_values) for _ in sublist]

        if min_val:
            y_flat = [element + min_val for element in y_flat]

        df = pd.DataFrame(
            {'date_col': x_flat,
             'values': y_flat,
             'container': z_repeat
             })
        
        df['container'] = [(str(i)[:7] + '..') if len(str(i)) > 7 else str(i) for i in df['container']]
        df['date_col'] = pd.to_datetime(df['date_col'])
        df = df.sort_values('date_col')
        return df

    def run_go(self, folder_path, destination_path, given_min_time_str, given_max_time_str):
        """
        Run Go program to process chunks based on given time interval.

        :param folder_path: Path of the folder where the chunks are stored.
        :param destination_path: Path of the folder where the processed json files are to be stored.
        :param given_min_time_str: Start time of the data extraction in string format.
        :param given_max_time_str: End time of the data extraction in string format.
        """
        given_min_time_dt = datetime.strptime(given_min_time_str, '%Y-%m-%d %H:%M:%S')
        given_max_time_dt = datetime.strptime(given_max_time_str, '%Y-%m-%d %H:%M:%S')
        given_min_time = int(given_min_time_dt.timestamp() * 1000)
        given_max_time = int(given_max_time_dt.timestamp() * 1000)
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Requested time interval: {given_min_time_str} - {given_max_time_str}")
        
        for chunk_folder in os.listdir(folder_path):
            chunk_folder_path = os.path.join(folder_path, chunk_folder)
            if os.path.isdir(chunk_folder_path) and chunk_folder != '.ipynb_checkpoints':
                meta_json_path = os.path.join(chunk_folder_path, "meta.json")
                with open(meta_json_path) as file:
                    meta = json.load(file)
                min_time = meta["minTime"]
                max_time = meta["maxTime"]
                

                if min_time >= given_min_time and max_time <= given_max_time:
                    min_time_str = datetime.fromtimestamp(min_time/1000).strftime('%Y-%m-%d %H-%M-%S')
                    max_time_str = datetime.fromtimestamp(max_time/1000).strftime('%Y-%m-%d %H-%M-%S')
                    file_min_time_str = datetime.fromtimestamp(min_time/1000).strftime('%Y%m%d %H%M%S')
                    file_max_time_str = datetime.fromtimestamp(max_time/1000).strftime('%Y%m%d %H%M%S')
                    logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Selected chunk's time interval: {min_time_str} - {max_time_str}")
                    command = ['./prometheus-tsdb-dump/main','-bucket','open5gs-respons-logs','-prefix','prometheus-metrics/respons-amf-forecaster/'+ str(chunk_folder),'-local-path','tsdb-json','-block','tsdb-json/prometheus-metrics/respons-amf-forecaster/' + str(chunk_folder)]
                    output = subprocess.run(command, capture_output=True)
                    filename = str(file_min_time_str) + "-" + str(file_max_time_str) + '.json'
                    with open(os.path.join(destination_path, filename), 'w') as file:
                        file.write(output.stdout.decode())
                    with open(os.path.join(destination_path, filename), 'r') as file:
                        lines = file.readlines()
                    with open(os.path.join(destination_path, filename), 'w') as file:
                        file.writelines(lines[3:])
                    logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::(Locally) Saved JSON to {os.path.join(destination_path, filename)}")
                    self.utils.upload_file(str(os.path.join(destination_path, filename)), str(os.path.join(destination_path, filename)))
                    
                    
                    
    def download_chunks(self, local_path):
        """
        Download chunks of raw data and save it to the given local path.

        :param local_path: The local path where chunks of raw data are to be saved.
        """
        aws_command = f"{os.environ.get('s3')} {local_path} --recursive"
        result = subprocess.run(aws_command, shell=True, check=True)
        
    
                
if __name__ == "__main__":
    """
    Entry point to parse command line arguments and run AMFDataProcessor.
    """
    parser = argparse.ArgumentParser(description="Process some AMF data.")
    parser.add_argument("--download", action='store_true', help="Include this flag to download chunks. --download does not require any argument")
    parser.add_argument("--process", action='store_true', help="Include this flag to process chunks into JSON format. --process requires --start, and --end.")
    parser.add_argument("--generate", action='store_true', help="Include this flag to generate the data frame and save the data as a paraquet file. --generate requires --metric and --level")
    
    parser.add_argument("--start", type=str, help="Start time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--end", type=str, help="End time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--level", type=str, help="Container level to filter on. Could be 'amf', 'support', 'upperlimit', 'amf+support'")
    parser.add_argument("--metric", type=str, help="Metric name to filter on. CPU: container_cpu_usage_seconds_total and Memory: container_memory_usage_bytes")
    parser.add_argument("--pod", type=str, help="Pod name to filter on. Leave empty for all pods.")
    
    args = parser.parse_args()
    
    if not any([args.download, args.process, args.generate]):
        parser.error("One of --download, --process, or --generate must be provided.")
    if args.process and not all([args.start, args.end]):
        parser.error("--process requires --start, and --end.")
    if args.generate and not all([args.level, args.metric]):
        parser.error("--generate requires --metric and --level")
    
    sys.stdout = open('logs/console_output.log', 'w')
    processor = AMFDataProcessor()
    panda = processor.run(args)
    
    summary_str = panda.describe().to_string().replace('\n', ' | ')
    head_str = panda.head().to_string().replace('\n', ' | ')

    logging.info(f"{os.path.basename(__file__)}::Summary of Requested data:")
    logging.info(f"{os.path.basename(__file__)}::{summary_str}")
    logging.info(f"{os.path.basename(__file__)}::First few entries of requested data:")
    logging.info(f"{os.path.basename(__file__)}::{head_str}")
        
    filename = "sample::" + os.path.basename(__file__) + "::metric:" + str(args.metric) + ";pod:" + str(args.pod) + ";level:" + str(args.level) + ";start:" + datetime.fromtimestamp(args.start).strftime('%Y-%m-%d %H:%M:%S') + ";end:" + datetime.fromtimestamp(args.end).strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists("parquet"):
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Creating folder: models")
        os.makedirs("parquet")
    filepath = os.path.join("parquet", filename)                                                                                 
    panda.to_parquet(filepath, compression='gzip')
    logging.info(f"{os.path.basename(__file__)}::Data Saved to:{filepath}")    
        
                                                                                     
    

    