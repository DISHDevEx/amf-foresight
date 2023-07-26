# AMF Foresight

## AMF Data Processing and Forecasting Pipeline: Orchestrator
This script integrates the `AMFDataProcessor`, `FeatureEngineer`, and multiple machine learning model classes (`ARIMAModel`, `ProphetModel`, `LSTMModel`) through an `Orchestrator` class to create a comprehensive data pipeline for AMF data.

1. `AMFDataProcessor`: Downloads and processes the AMF data.
2. `FeatureEngineer`: Performs value modifications on the DataFrame based on the type of feature specified. The available types include 'memory', 'cpu', and 'utilization'
3. `ARIMAModel`, `ProphetModel`, `LSTMModel`: Train the specified model on the processed data.

### Requirements
Before you begin, ensure you have met the following requirements:
* You have installed Python 3.6 or later.
* You have installed the following Python libraries: `argparse`, `pandas`, `pyarrow`, `prophet`, `statsmodels`, `tensorflow`, `matplotlib`.

### Using the Pipeline
* Downloading TSDB chunks:
    - To download the TSDB chunks, run the script as follows (This should be done once before any of the next steps):
        ```bash
        python amf_foresight/pipeline.py --download
        ```
* Processing the TSDB chunks into JSON files:
    - To convert the TSDB chunks in your required timeframe into JSON files, you'll need to provide the start and end dates for the data extraction:
        ```bash
        python amf_foresight/pipeline.py --process --start "2023-06-06 14:00:00" --end "2023-06-06 16:00:00"
        ```
* Generating the dataframe:
    - To generate a pandas dataframe and save it as a Parquet file, you'll need to specify the metric, type, and level:
        ```bash
        python amf_foresight/pipeline.py --generate --metric "container_cpu_usage_seconds_total" --type "utilization" --level "amf"
        ```    
* Training a model:
    - To train a model, you'll need to specify the data source and the model type. 
        - You can either load data from the specified Parquet file and train a model with it.
            ```
            python amf_foresight/pipeline.py.py --train --data "/path/to/your/data.parquet" --model "ARIMA"
            ```
        - Or you can process the data, generate the data frame, save it as a Parquet file, and then train the model, you can use the following command:
            ```
            python amf_foresight/pipeline.py --process --start "2023-06-06 14:00:00" --end "2023-06-06 16:00:00" --generate --metric "container_cpu_usage_seconds_total" --type "utilization" --level "amf" --train --model "ARIMA"
            ```
* Run the pipleine (Download TSDB chunks -> Process the TSDB chunks into JSON files -> Generate dataframe -> Train a model):
    ```bash
    python amf_foresight/pipeline.py --download --process --start [START_TIME] --end [END_TIME] --generate --level [CONTAINER_LEVEL] --type [FEATURE_TYPE] --metric [METRIC] --train --model [MODEL_TYPE]
    ```
    * `--download`: Include this flag to download chunks.
    * `--process`: Include this flag to process chunks into JSON format.
    * `--start` and `--end`: Replace `[START_TIME]` and `[END_TIME]` with the start and end times for the data processing window. Times must be in '%Y-%m-%d %H:%M:%S' format.
    * `--generate`: Include this flag to generate the data frame and save the data as a parquet file.
    * `--level`: Replace `[CONTAINER_LEVEL]` with the container level to filter on (options: 'amf', 'support', 'upperlimit', 'amf+support'). 
    * `--type`: Replace `[FEATURE_TYPE]` with the type of feature engineering you want to apply (options: 'memory', 'cpu', 'utilization').    
    * `--metric`: Replace `[METRIC]` with the metric name to filter on. Leave empty for all metrics.
    * `--train`: Include this flag to train your data.
    * `--model`: Replace `[MODEL_TYPE]` with the model you would like to use.
    
