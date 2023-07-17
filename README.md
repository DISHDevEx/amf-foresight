# AMF Foresight

## AMF Data Processing and Feature Engineering Pipeline: Orchestrator
This script integrates the `AMFDataProcessor` and `FeatureEngineer` classes through an `Orchestrator` class to create a comprehensive data pipeline for AMF data. 
1. `AMFDataProcessor`: Reads and processes the AMF data from the provided directory. Can filter the data based on a specified metric and pod name.
2. `FeatureEngineer`: Performs value modifications on the DataFrame based on the type of metric specified. The available metrics include 'memory', 'cpu', and 'cpu_utilization'

### Requirements
Before you begin, ensure you have met the following requirements:
* You have installed Python 3.6 or later.
* You have installed the following Python libraries: `argparse`, `pandas`.

### Using the Pipeline
To use this pipeline, follow these steps:
1. Navigate to the directory containing the script using the terminal.
2. Run the following command:
    ```bash
    python your_script_name.py --directory [DIRECTORY_PATH] --metric [METRIC_NAME] --pod [POD_NAME] --type [FEATURE_TYPE]
    ```
    * your_script_name.py: Replace with the name of your Python script.
    * --download: Include this flag to download chunks. --download requires --chunks.
    * --process: Include this flag to process chunks into JSON format. --process requires --chunks, --jsons, --start, and --end.
    * --generate: Include this flag to generate the data frame and save the data as a paraquet file. --generate requires --jsons and --level.
    * Replace `[CHUNKS_PATH]` with the path to save the chunks.
    * Replace `[JSONS_PATH]` with the path to save the JSON files.
    * Replace `[START_TIME]` with the start time in %Y-%m-%d %H:%M:%S format.
    * Replace `[END_TIME]` with the end time in %Y-%m-%d %H:%M:%S format. 
    * Replace `[CONTAINER_LEVEL]` with the container level to filter on. (options: 'amf', 'support', 'upperlimit', 'amf+support). 
    * Replace `[FEATURE_TYPE]` with the type of feature engineering you want to apply (options: 'memory', 'cpu', 'cpu_utilization').    
    * (Optional) Replace `[METRIC_NAME]` with the name of the metric to filter on.
    * (Optional) Replace `[POD_NAME]` with the name of the pod to filter on.
3. The script will read and process the data, perform feature engineering, and print a summary of the requested data and the first few entries of the data.

## Load Data: AMF Data Processor
The `AMFDataProcessor` class is used to process AMF data. This script reads JSON files, performs transformations based on given parameters, provides a summary of the requested data, and saves it into a CSV file.

### Requirements
* You have installed Python 3.6 or later.
* You have installed the following Python libraries: `devex_sdk` `pyspark`, `datetime`, `collections`, `pandas`, `time`, `csv`, `os`, `json`, `boto3`, `itertools`, `pyspark`, `logging`, `argparse`

### Usage
To use `AMFDataProcessor`, follow these steps:
1. Navigate to the directory containing the script using the terminal.
2. Run the following command:
    ```bash
    python your_script_name.py --download --process --generate --chunks [CHUNKS_PATH] --jsons [JSONS_PATH] --parquet [PARQUET_PATH] --start [START_TIME] --end [END_TIME] --level [CONTAINER_LEVEL] --metric [METRIC_NAME] --pod [POD_NAME]
    ```
    * your_script_name.py: Replace with the name of your Python script.
    * --download: Include this flag to download chunks. --download requires --chunks.
    * --process: Include this flag to process chunks into JSON format. --process requires --chunks, --jsons, --start, and --end.
    * --generate: Include this flag to generate the data frame and save the data as a paraquet file. --generate requires --jsons and --level.
    * Replace `[CHUNKS_PATH]` with the path to save the chunks.
    * Replace `[JSONS_PATH]` with the path to save the JSON files.
    * Replace `[START_TIME]` with the start time in %Y-%m-%d %H:%M:%S format.
    * Replace `[END_TIME]` with the end time in %Y-%m-%d %H:%M:%S format.
    * Replace `[CONTAINER_LEVEL]` with the container level to filter on. (options: 'amf', 'support', 'upperlimit', 'amf+support)
    * (Optional) Replace `[METRIC_NAME]` with the name of the metric to filter on.
    * (Optional) Replace `[POD_NAME]` with the name of the pod to filter on.
   
    Note: You can use one or a combination of the flags --download, --process, and --generate to specify whether you want to download chunks, process chunks into JSON format, or generate the data frame and save the data as a parquet file respectively.

3. The script will process the data, print a summary of the requested data and the first few entries of the data, and save the data to a CSV file in the `csv` directory.
4. The name of the CSV file will be in the following format: `sample::your_script_name.py::metric:MetricName;pod:PodName;level:ContainerLevel;time:current_timestamp.csv`. The `MetricName`, `PodName`, and `current_timestamp` will be replaced with your provided metric name, pod name, and the current timestamp, respectively.
5. The path of the saved CSV file will be printed at the end of the script.

### Description of Key Functions
The script `amf_data_extractor.py` includes several functions each designed to perform a specific task in the process of extracting and processing AMF data. Here are descriptions for the key functions:
1. **`clear_folders(self, folders)`**: This function takes a list of folder paths as an argument. It clears any files in these folders if they exist, or creates them if they don't.
2. **`get_data(self, directory, container_level, metric=None, pod=None)`**: This function takes in the directory of JSON files and returns combined Spark and Pandas dataframes. It uses other defined functions to get data, transform it, and convert it to a pandas dataframe.
3. **`get_dataframes(self, directory, container_level)`**: This function extracts Spark dataframes from a given directory of JSON files. 
4. **`transform_dataframe(self, amf_data, metric_name= None, pod_name=None)`**: This function transforms the timestamps and filters based on `metric_name` and `pod_name` in the AMF Data.
5. **`get_amf_data(self, json_object_path, container_level="all")`**: This function extracts the dataframe from a JSON file and filters out AMF data based on the container level.
6. **`get_min_value(self, amf_data)`**: This function gets the value of a metric that is used to support the application.
7. **`get_values(self, data, container_level)`**: This function extracts timestamps and values from a Spark dataframe and returns a pandas dataframe.
8. **`run_go(self, folder_path, destination_path, given_min_time_str, given_max_time_str)`**: This function processes chunks into JSON format.
9. **`download_chunks(self, local_path)`**: This function takes in a path to save the chunks and saves the raw data from AWS S3 into the given path.
10. **`__main__`**: This section of the code checks for arguments and runs the required processes accordingly. The key processes being downloading chunks from S3, processing those chunks into JSON files, and then generating a dataframe from those JSON files. This is where the command-line arguments are parsed and used to control the script's functionality.




## Feature Engineering: Feature Engineer
The `FeatureEngineer` class is used to modify AMF data. This script reads CSV files, performs transformations based on given parameters, provides a summary of the requested data, and saves it into a CSV file.

### Requirements
* You have installed Python 3.6 or later.
* You have installed the following Python libraries: `pandas`, `argparse`, `os`.

### Usage

To use `FeatureEngineer`, follow these steps:
1. Navigate to the directory containing the script using the terminal.
2. Run the following command:
    ```bash
    python your_script_name.py --data [PATH_TO_CSV_FILE] --type [FEATURE_TYPE]
    ```
    * Replace `your_script_name.py` with the name of your Python script.
    * Replace `[PATH_TO_CSV_FILE]` with the directory path where your CSV file is located.
    * Replace `[FEATURE_TYPE]` with the type of feature engineering you want to apply (options: 'memory', 'cpu', 'cpu_utilization').
3. The script will read the data, perform feature engineering, print a summary of the requested data and the first few entries of the data, and save the data to a CSV file in the `csv` directory.
4. The name of the CSV file will be in the following format: `sample::your_script_name.py::filename;type:feature_type.csv`. The `filename` will be replaced with the name of your provided CSV file and `feature_type` with the type of feature engineering applied.
5. The path of the saved CSV file will be printed at the end of the script.

### Description of Key Functions
The `FeatureEngineer` class includes the following key methods:
1. `get_data(self, data, feature_type)`: This method is used to read and modify the data based on the specified feature type.
2. `read_data(self, path)`: This method reads a CSV file from a specified path and returns a pandas DataFrame.
3. `value_modifier(self, data, metric)`: This method modifies the DataFrame based on the type of metric specified. Available metrics include 'memory', 'cpu', and 'cpu_utilization'.




