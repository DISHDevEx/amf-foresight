from pyspark.sql import functions as F
from load_data import AMFDataProcessor
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time
import os
import random
import time
import argparse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas




class EDA:
    def retrieve_duplicates(self, transformed_data, column_name, pod_name):
        """
        Retrieves duplicates for a random timestamp and prints values and dataframe structure the timestamp has.
        :param transformed_data: A PySpark DataFrame
        :param column_name: Name of the column to transform
        :param pod_name: Name of the pod
        """
        duplicates = self.find_duplicates(transformed_data)
        print("These are some of the timestamps where data was duplicated")
        datetime_string = random.choice(duplicates)
        print("Let's take a look at the dataframe and values at:", datetime_string)
        filtered_df = transformed_data.filter(F.array_contains(F.col("date_col"), datetime_string))
        filtered_df = self.focus_dataframe(filtered_df)
        values = self.get_all_values(filtered_df)
        print("Dataframe:")
        print(filtered_df.show())
        print("Values:")
        print(values[datetime_string])
        return filtered_df, values[datetime_string]
    
    def find_duplicates(self, data):
        """
        Find all timestamps where values are duplicated
        :param data: A PySpark DataFrame
        """
        x_values = data.select("date_col").rdd.flatMap(lambda x: x).collect()
        y_values = data.select("values").rdd.flatMap(lambda x: x).collect()
        result_dict = {}
        duplicate_timestamps = []
        for x_list, y_list in zip(x_values, y_values):
            for x, y in zip(x_list, y_list):
                if x in result_dict:
                    duplicate_timestamps.append(x)
                else:
                    result_dict[x] = [float(y)]
        return duplicate_timestamps
    
    def get_all_values(self, data):
        """
        Get all values for each timestamp
        :param data: A PySpark DataFrame
        """
        x_values = data.select("date_col").rdd.flatMap(lambda x: x).collect()
        y_values = data.select("values").rdd.flatMap(lambda x: x).collect()
        result_dict = {}
        for x_list, y_list in zip(x_values, y_values):
            for x, y in zip(x_list, y_list):
                if x in result_dict:
                    result_dict[x].append(float(y))
                else:
                    result_dict[x] = [float(y)]
        return result_dict


    def plot_duplicates(self, data, metric):
        """
        Plot duplicates based on a metric name
        :param data: A PySpark DataFrame
        :param metric: The metric to plot
        """
        x_values = data.select("date_col").rdd.flatMap(lambda x: x).collect()
        y_values = data.select("values").rdd.flatMap(lambda x: x).collect()
        z_values = data.select("metric_id").rdd.flatMap(lambda x: x).collect()

        x_flat = [item for sublist in x_values for item in sublist]
        y_flat = [item for sublist in y_values for item in sublist]
        z_repeat = [label for sublist, label in zip(x_values, z_values) for _ in sublist]

        df = pd.DataFrame(
        {'Time': x_flat,
         'Values': y_flat,
         'Metric ID': z_repeat
        })
        
        df['Time'] = pd.to_datetime(df['Time'])

        fig = px.line(df, x='Time', y='Values', color='Metric ID')
        fig.update_layout(
            title=metric,
            xaxis=dict(title='Time'),
            yaxis=dict(title='Values')
        )
        fig.show()

    def data_plot(self, df):
        fig = px.line(df, x='date_col', y='values', color='container')
        fig.update_layout(
            xaxis=dict(title='Time'),
            yaxis=dict(title='Values')
        )
        fig.show()
    
    def plotly_plot(self, data, metric):
        """
        Matplotlib plot for data based on a metric name
        :param data: A Pandas DataFrame
        :param metric: The metric to plot
        """
        time1 = time.time()
        x_values = data.select("date_col").rdd.flatMap(lambda x: x).collect()
        y_values = data.select("values").rdd.flatMap(lambda x: x).collect()
        z_values = data.select("metric_name").rdd.flatMap(lambda x: x).collect()

        x_flat = [item for sublist in x_values for item in sublist]
        y_flat = [item for sublist in y_values for item in sublist]
        z_repeat = [label for sublist, label in zip(x_values, z_values) for _ in sublist]

        df = pd.DataFrame(
        {'Time': x_flat,
         'Values': y_flat,
         'Container Name': z_repeat
        })
        df['Container Name'] = [(str(i)[:7] + '..') if len(str(i)) > 7 else str(i) for i in df['Container Name']]
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time')
        
        fig = px.line(df, x='Time', y='Values', color='Container Name')
        fig.update_layout(
            title=metric,
            xaxis=dict(title='Time'),
            yaxis=dict(title='Values')
        )
        fig.show()
        time2 = time.time()
        print("Time taken to plot this: ", str(time2 - time1))

    def matplot_plot(self, data, metric):
        """
        Matplotlib plot for data based on a metric name
        :param data: A Pandas DataFrame
        :param metric: The metric to plot
        """
        time1 = time.time()
        x_values = data.select("date_col").rdd.flatMap(lambda x: x).collect()
        y_values = data.select("values").rdd.flatMap(lambda x: x).collect()
        z_values = data.select("metric_name").rdd.flatMap(lambda x: x).collect()

        x_flat = [item for sublist in x_values for item in sublist]
        y_flat = [item for sublist in y_values for item in sublist]
        z_repeat = [label for sublist, label in zip(x_values, z_values) for _ in sublist]

        df = pd.DataFrame(
            {'Time': x_flat,
             'Values': y_flat,
             'Container Name': z_repeat
            })

        df['Time'] = pd.to_datetime(df['Time'])

        fig, ax = plt.subplots()
        for container, group in df.groupby('Container Name'):
            ax.plot(group['Time'], group['Values'], label=container)

        ax.set(title=metric, xlabel='Time', ylabel='Values')
        ax.legend()
        plt.show()

        time2 = time.time()
        print("Time taken to plot this: ", str(time2 - time1))
        
    def focus_dataframe(self, df):
        """
        This function drops all columns which contain only null values.
        :param df: A PySpark DataFrame
        """
        null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        total_rows = df.count()
        to_drop = [k for k, v in null_counts.items() if v == total_rows]
        df = df.drop(*to_drop)
        common_columns = ['metric_container', 'metric_endpoint', 'metric_id', 'metric_image', 'metric_instance', 'metric_job', 'metric_name', 'metric_namespace', 'metric_node', 'metric_node', 'metric_service', 'date_col']
        df = df.drop(*common_columns)
        print("Focused Dataframe:")
        print(df)
        return df          
        
        