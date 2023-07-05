from devex_sdk import Spark_Data_Connector, Nested_Json_Connector
from pyspark.sql.types import FloatType, TimestampType, LongType
from pyspark.sql import SparkSession
from datetime import datetime
from collections import defaultdict
import pyspark.sql.functions as F
import pandas as pd
import time
import csv
import itertools


def get_amf_data(json_object_path):
    obj = Nested_Json_Connector(json_object_path)
    err, data = obj.read_nested_json()
    data = data.filter(F.col('metric_pod').startswith('open5gs-amf') & \
                       (F.col("metric_namespace") == "openverso") & (F.col('metric_name').isNotNull()))
    data = data.withColumn("date_col", F.expr("transform(timestamps, x -> from_unixtime(x/1000))"))
    return data


def transform_dataframe(amf_data, column_name, pod_name):
    if column_name:
        amf_data = amf_data.filter(F.col('metric___name__') == column_name)
    if pod_name:
        amf_data = amf_data.filter(F.col("metric_pod").startswith(pod_name))
    return amf_data


def get_min_value(data):
    min_vals = data.filter(F.col("metric_container").isNull()).select("values").rdd.flatMap(lambda x: x).collect()
    min_val = None
    if min_vals:
        min_val = min_vals[0][0]
    return min_val


def get_values(data, metric):
    min_val = get_min_value(data)
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

    if 'memory' in metric.split('_'):
        df['values'] = df['values'] / 1048576

    df = df.groupby('date_col').agg({'values': 'sum'}).reset_index()

    return df

