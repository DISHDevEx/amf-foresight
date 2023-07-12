# AMF Foresight

## Load Data: AMF Data Processor

This script retrieves AMF data from a JSON file, performs transformations, optional filtering and returns a pandas dataframe.

### Requirements
- Python: `Python 3.9.15
- Python libraries: `devex_sdk` `pyspark`, `datetime`, `collections`, `pandas`, `time`, `csv`, `os`, `json`, `boto3`, `itertools`, `pyspark`, `logging`, `argparse`

### Usage

1. Clone the repository.
2. Open your terminal.
3. Navigate to the repository on your local macine.
4. Execute the script using the command:
```bash
python amf_foresight/load_data.py --directory /path/to/json --metric metric_name --pod pod_name
```

### Optional arguments: 
- `--metric <metric_name>`: If you want to filter the results based on a specific metric, replace `<metric_name>` with the name of that metric.
- `--pod <pod_name>`: If you want to filter the results based on a specific pod, replace `<pod_name>` with the name of that pod.

The script will print a summary and the first few entries of the requested data.

### Description of Key Functions

- `get_data(self, directory, metric=None, pod=None)`: Main function that gets and processes data from the directory containing JSON files.
- `get_dataframes(self, directory)`: Extracts Spark dataframes from a given directory of JSON files.
- `transform_dataframe(self, amf_data, metric_name= None, pod_name=None)`: Transforms the timestamps and filters data based on metric_name and pod_name.
- `get_amf_data(self, json_object_path)`: Extracts the dataframe from JSON and filters out AMF data.
- `get_min_value(self, amf_data)`: Gets the value of a metric that is used to support the application.
- `get_values(self, data)`: Extracts timestamps and values of a spark dataframe, and returns a pandas dataframe.
- `download_chunks(self, local_path)`: Downloads data chunks from the AWS S3 bucket and saves them to the local path.


