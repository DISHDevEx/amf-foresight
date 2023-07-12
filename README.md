# AMF Foresight

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
    python your_script_name.py --directory [PATH_TO_JSON_FILES] --metric [METRIC_NAME] --pod [POD_NAME]
    ```
    * Replace `your_script_name.py` with the name of your Python script.
    * Replace `[PATH_TO_JSON_FILES]` with the directory path where your JSON files are located.
    * Replace `[METRIC_NAME]` with the name of the metric you want to filter on (optional).
    * Replace `[POD_NAME]` with the name of the pod you want to filter on (optional).

3. The script will process the data, print a summary of the requested data and the first few entries of the data, and save the data to a CSV file in the `csv` directory.
4. The name of the CSV file will be in the following format: `sample::your_script_name.py::metric:MetricName;pod:PodName;time:current_timestamp.csv`. The `MetricName`, `PodName`, and `current_timestamp` will be replaced with your provided metric name, pod name, and the current timestamp, respectively.
5. The path of the saved CSV file will be printed at the end of the script.

### Description of Key Functions
- `get_data(self, directory, metric=None, pod=None)`: Main function that gets and processes data from the directory containing JSON files.
- `get_dataframes(self, directory)`: Extracts Spark dataframes from a given directory of JSON files.
- `transform_dataframe(self, amf_data, metric_name= None, pod_name=None)`: Transforms the timestamps and filters data based on metric_name and pod_name.
- `get_amf_data(self, json_object_path)`: Extracts the dataframe from JSON and filters out AMF data.
- `get_min_value(self, amf_data)`: Gets the value of a metric that is used to support the application.
- `get_values(self, data)`: Extracts timestamps and values of a spark dataframe, and returns a pandas dataframe.
- `download_chunks(self, local_path)`: Downloads data chunks from the AWS S3 bucket and saves them to the local path.


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





