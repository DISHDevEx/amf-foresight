from devex_sdk import Spark_Data_Connector, Nested_Json_Connector
from pyspark.sql.types import FloatType, TimestampType, LongType
from pyspark.sql import SparkSession
from datetime import datetime
from setup_logger import setup_logger
from collections import defaultdict
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
setup_logger()

class AMFDataProcessor:
    
    def clear_folders(self, folders):
        """
        This function creates a folder if it does not exist and clears all the files in the folders passed in
        :param folders: list of folders 
        """
        for folder in folders:
            if os.path.exists(folder):
                logging.info(f"Clearing files in folder: {folder}")
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
        This function runs the entire pipeline and returns a dataframe
        :param
        """
        time1 = time.time()
        if args.download:
            logging.info("Downloading chunks..")
            logging.info(f"Running with arguments for download: chunks={args.chunks}")
            self.clear_folders([args.chunks])
            self.download_chunks(args.chunks)
            logging.info("Downloaded chunks.")

        if args.process:
            self.clear_folders([args.jsons])
            logging.info("Processing chunks..")
            logging.info(f"Running with arguments for process: chunks={args.chunks}, jsons={args.jsons}, start={args.start}, end={args.end}")
            self.run_go(args.chunks, args.jsons, args.start, args.end)
            logging.info("Processed chunks.")

        if args.generate:
            logging.info("Generating dataframe..")
            logging.info(f"Running with arguments for generate: jsons={args.jsons}, level={args.level}, metric={args.metric}, pod={args.pod}")
            spark, panda = self.get_data(args.jsons, args.level, args.metric, args.pod)
            logging.info("Generated dataframe")
          
            return panda
    
    def get_data(self, directory, container_level="all", metric=None, pod=None):
        """
        This function takes in the directory of JSON files and returns a combined pandas dataframes
        :param directory: Path to JSON files
        """
        spark_dataframes = self.get_dataframes(directory, container_level)
        spark_dataframe = self.transform_dataframe(spark_dataframes, metric, pod)
        pandas_dataframe = self.get_values(spark_dataframe, container_level)
        return spark_dataframe, pandas_dataframe

    def get_dataframes(self, directory, container_level):
        """
        This function extracts spark dataframes from a given directory of JSON files
        :param directory: Path to JSON files
        """
        flag = False
        for i, filename in enumerate(os.listdir(directory)):
            if os.path.isfile(os.path.join(directory, filename)) and not flag:
                appended_df = self.get_amf_data(os.path.join(directory, filename), container_level)
                logging.info(f"Generated Dataframe for: {os.path.join(directory, filename)}")
                flag = True
            elif flag:
                if os.path.isfile(os.path.join(directory, filename)):
                    new_df = self.get_amf_data(os.path.join(directory, filename), container_level)
                    logging.info(f"Generated Dataframe for: {os.path.join(directory, filename)}")
                    appended_df = appended_df.union(new_df)
        return appended_df

    def transform_dataframe(self, amf_data, metric_name= None, pod_name=None):
        """
        This function tranforms the timestamps and filters based on metric_name and pod_name
        :param amf_data: AMF Data
        :param metric_name: Metric name to filter on
        :param pod_name: Pod ID to filter on
        """
        amf_data = amf_data.withColumn("date_col", F.expr("transform(timestamps, x -> from_unixtime(x/1000))"))
        if metric_name:
            amf_data = amf_data.filter(F.col('metric___name__') == metric_name)
        if pod_name:
            amf_data = amf_data.filter(F.col("metric_pod").startswith(pod_name))
        return amf_data
    
    def get_all_data(self, json_object_path, container_level="all"):
        """
        This function grabs all data without filtering for EDA purposes 
        :param json_object_path: Path to JSON
        """
        obj = Nested_Json_Connector(json_object_path)
        data = data.filter(F.col("metric_namespace") == "openverso")
        data = data.filter(F.col('metric_pod').startswith('open5gs'))
        return data
    
    def get_amf_data(self, json_object_path, container_level="all"):
        """
        This function extracts the dataframe from JSON and filters out AMF data with a container name
        :param json_object_path: Path to JSON
        """
        obj = Nested_Json_Connector(json_object_path, setup = "32gb")
        err, data = obj.read_nested_json()
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
        This function gets the value of a metric that is used to support the application
        :param data: AMF Data
        """
        min_vals = amf_data.filter(F.col("metric_container").isNull() & F.col('metric_image').isNotNull()).select("values").rdd.flatMap(lambda x: x).collect()
        min_val = None
        if min_vals:
            min_val = min_vals[0][0]
        return min_val

    def get_values(self, data, container_level):
        """
        This function extracts timestamps and values of a spark dataframe returns a pandas dataframe
        :param data: AMF Data
        """
        min_val = None
        max_val = None
        if container_level == "amf+support":
            min_val = self.get_min_value(data)
            data = data.filter(F.col('metric_container')=='open5gs-amf')
        
            
        x_values = data.select("date_col").rdd.flatMap(lambda x: x).collect()
        y_values = data.select("values").rdd.flatMap(lambda x: x).collect()

        x_flat = [item for sublist in x_values for item in sublist]
        y_flat = [item for sublist in y_values for item in sublist]

        if min_val:
            y_flat = [element + min_val for element in y_flat]

        df = pd.DataFrame(
            {'date_col': x_flat,
             'values': y_flat,
             })

        df['date_col'] = pd.to_datetime(df['date_col'])
        df = df.sort_values('date_col')
        return df

    def run_go(self, folder_path, destination_path, given_min_time_str, given_max_time_str):
        """
        This function processes the chunks which are within the requested time interval
        :param data: AMF Data
        """
        given_min_time_dt = datetime.strptime(given_min_time_str, '%Y-%m-%d %H:%M:%S')
        given_max_time_dt = datetime.strptime(given_max_time_str, '%Y-%m-%d %H:%M:%S')
        given_min_time = int(given_min_time_dt.timestamp() * 1000)
        given_max_time = int(given_max_time_dt.timestamp() * 1000)
        logging.info(f"Requested time interval: {given_min_time_str} - {given_max_time_str}")
        
        for chunk_folder in os.listdir(folder_path):
            chunk_folder_path = os.path.join(folder_path, chunk_folder)
            if os.path.isdir(chunk_folder_path) and chunk_folder != '.ipynb_checkpoints':
                meta_json_path = os.path.join(chunk_folder_path, "meta.json")
                with open(meta_json_path) as file:
                    meta = json.load(file)
                min_time = meta["minTime"]
                max_time = meta["maxTime"]
                

                if min_time >= given_min_time and max_time <= given_max_time:
                    min_time_str = datetime.fromtimestamp(min_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                    max_time_str = datetime.fromtimestamp(max_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                    logging.info(f"Selected chunk's time interval: {min_time_str} - {max_time_str}")
                    command = ['./prometheus-tsdb-dump/main','-bucket','open5gs-respons-logs','-prefix','prometheus-metrics/respons-amf-forecaster/'+ str(chunk_folder),'-local-path','tsdb-json','-block','tsdb-json/prometheus-metrics/respons-amf-forecaster/' + str(chunk_folder)]
                    output = subprocess.run(command, capture_output=True)
                    filename = str(chunk_folder) + '.json'
                    with open(os.path.join(destination_path, filename), 'w') as file:
                        file.write(output.stdout.decode())
                    with open(os.path.join(destination_path, filename), 'r') as file:
                        lines = file.readlines()
                    with open(os.path.join(destination_path, filename), 'w') as file:
                        file.writelines(lines[3:])
                    logging.info(f"Saved JSON to {os.path.join(destination_path, filename)}")
                    
                    
    def download_chunks(self, local_path):
        """
        This function takes in the path to save the chunks and saves the raw data in the given path
        :param
        """
        aws_command = f"{os.environ.get('s3')} {local_path} --recursive"
        result = subprocess.run(aws_command, shell=True, check=True)
        
    
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some AMF data.")
    parser.add_argument("--download", action='store_true', help="Include this flag to download chunks. --download requires --chunks.")
    parser.add_argument("--process", action='store_true', help="Include this flag to process chunks into JSON format. --process requires --chunks, --jsons, --start, and --end.")
    parser.add_argument("--generate", action='store_true', help="Include this flag to generate the data frame and save the data as a paraquet file. --generate requires --jsons and --level.")
    parser.add_argument("--chunks", type=str, help="Path where the chunks are/should be downloaded. The chunks contain the raw data from the AMF.")
    parser.add_argument("--jsons", type=str, help="Path where the processed JSONs are/should be stored. These JSONs are generated from the chunks.")
    parser.add_argument("--parquet", type=str, help="Path where the processed CSV are/should be stored as a paraquet file. These files are generated from the JSONs.")
    parser.add_argument("--start", type=str, help="Start time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--end", type=str, help="End time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--level", type=str, help="Container level to filter on. Could be 'amf', 'support', 'upperlimit', 'amf+support' ")
    parser.add_argument("--metric", type=str, help="Metric name to filter on. Leave empty for all metrics.")
    parser.add_argument("--pod", type=str, help="Pod name to filter on. Leave empty for all pods.")
    
    args = parser.parse_args()
    
    if not any([args.download, args.process, args.generate]):
        parser.error("One of --download, --process, or --generate must be provided.")
    if args.download and not args.chunks:
        parser.error("--download requires --chunks.")
    if args.process and not all([args.chunks, args.jsons, args.start, args.end]):
        parser.error("--process requires --chunks, --jsons, --start, and --end.")
    if args.generate and not all([args.jsons, args.parquet, args.level]):
        parser.error("--generate requires --jsons, --parquet, and --level.")
    
    sys.stdout = open('logs/console_output.log', 'w')
    processor = AMFDataProcessor()
    panda = processor.run(args)
    
    summary_str = panda.describe().to_string().replace('\n', ' | ')
    head_str = panda.head().to_string().replace('\n', ' | ')

    logging.info("Summary of Requested data:")
    logging.info(summary_str)
    logging.info("First few entries of requested data:")
    logging.info(head_str)
        
    filename = "sample::" + os.path.basename(__file__) + "::metric:" + str(args.metric) + ";pod:" + str(args.pod) + ";level:" + str(args.level) + ";time:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    filepath = os.path.join(args.parquet, filename)                                                                                 
    panda.to_parquet(filepath, compression='gzip')
    logging.info(f"Data Saved to:{filepath}")    
        
                                                                                     
    

    