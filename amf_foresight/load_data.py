from devex_sdk import Spark_Data_Connector, Nested_Json_Connector
from pyspark.sql.types import FloatType, TimestampType, LongType
from pyspark.sql import SparkSession
from datetime import datetime
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


class AMFDataProcessor:

    def get_data(self, directory, metric=None, pod=None):
        """
        This function takes in the directory of JSON files and returns a combined pandas dataframes
        :param directory: Path to JSON files
        """
        spark_dataframes = self.get_dataframes(directory)
        spark_dataframe = self.transform_dataframe(spark_dataframes, metric, pod)
        pandas_dataframe = self.get_values(spark_dataframe)
        return pandas_dataframe

    def get_dataframes(self, directory):
        """
        This function extracts spark dataframes from a given directory of JSON files
        :param directory: Path to JSON files
        """
        flag = False
        for i, filename in enumerate(os.listdir(directory)):
            if os.path.isfile(os.path.join(directory, filename)) and not flag:
                appended_df = self.get_amf_data(os.path.join(directory, filename))
                flag = True
            elif flag:
                if os.path.isfile(os.path.join(directory, filename)):
                    new_df = self.get_amf_data(os.path.join(directory, filename))
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

    def get_all_amf_data(json_object_path):
        """
        This function extracts the dataframe from JSON and filters out all the AMF data regardless of whether a container name exists
        :param json_object_path: Path to JSON
        """
        obj = Nested_Json_Connector(json_object_path)
        err, data = obj.read_nested_json()
        data = data.filter(F.col('metric_pod').startswith('open5gs-amf') & \
                           (F.col("metric_namespace") == "openverso"))
        return data
    
    def get_amf_data(self, json_object_path):
        """
        This function extracts the dataframe from JSON and filters out AMF data with a container name
        :param json_object_path: Path to JSON
        """
        obj = Nested_Json_Connector(json_object_path)
        err, data = obj.read_nested_json()
        data = data.select('timestamps', 'metric___name__', 'values', 'metric_pod', 'metric_container', 'metric_name', 'metric_namespace')
        data = data.filter(F.col('metric_pod').startswith('open5gs-amf') & \
                           (F.col("metric_namespace") == "openverso") & (F.col('metric_name').isNotNull()))
        return data

    def get_min_value(self, amf_data):
        """
        This function gets the value of a metric that is used to support the application
        :param data: AMF Data
        """
        min_vals = amf_data.filter(F.col("metric_container").isNull()).select("values").rdd.flatMap(lambda x: x).collect()
        min_val = None
        if min_vals:
            min_val = min_vals[0][0]
        return min_val

    def get_values(self, data):
        """
        This function extracts timestamps and values of a spark dataframe returns a pandas dataframe
        :param data: AMF Data
        """
        min_val = self.get_min_value(data)
        data = data.filter(F.col("metric_container").isNotNull())

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

        df = df.groupby('date_col').agg({'values': 'sum'}).reset_index()
        return df


    def download_chunks(self, local_path):
        """
        This function takes in the path to save the chunks and saves the raw data in the given path
        :param
        """
        aws_command = f"aws s3 cp s3://open5gs-respons-logs/prometheus-metrics/respons-amf-forecaster/ {local_path} --recursive"
        result = subprocess.run(aws_command, shell=True, check=True)
           
            
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Process some AMF data.")
    parser.add_argument("--directory", type=str, required=True, help="Path to JSON files directory.")
    parser.add_argument("--metric", type=str, required=False, help="Metric name to filter on.")
    parser.add_argument("--pod", type=str, required=False, help="Pod name to filter on.")
    
    args = parser.parse_args()
    
    processor = AMFDataProcessor()
    data = processor.get_data(args.directory, args.metric, args.pod)
    
    
    print("Summary of Requested data:")
    print(data.describe())
    print("First few entries of requested data:")
    print(data.head())
    filename = "sample::" + os.path.basename(__file__) + "::metric:" + str(args.metric) + ";pod:" + str(args.pod) + ";time:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '.csv'
    path = "csv/" + filename                                                                                 
    data.to_csv(path, index=False)
    print("Data Saved to: ", path)
                                                                                     
    

    