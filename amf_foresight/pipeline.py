from load_data import AMFDataProcessor
from datetime import datetime
from setup_logger import setup_logger
from feature_engineering import FeatureEngineer
import argparse
import logging
import time 
import os

setup_logger()

class Orchestrator:
    """
    Orchestrator Class:
    A class that handles the orchestration of preprocessing, training and evaluating AMF data.
    """
    def __init__(self):
        self.processor = AMFDataProcessor()
        self.feature_engineer = FeatureEngineer()

    def preprocessing(self, args):
        logging.info("Preprocessing Data..")
        raw = self.processor.run(args)
        processed = self.feature_engineer.value_modifier(raw, args.type)
        logging.info("Preprocessed Data.")
        
        filename = "sample::" + os.path.basename(__file__) + "::metric:" + str(args.metric) + ";pod:" + str(args.pod) + ";level:" + str(args.level) + ";time:" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        path = "parquet/" + filename                                                                                 
        processed.to_parquet(path, index=False)
        print("Data Saved to: ", path)
        
    
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Process some AMF data.")
    parser.add_argument("--download", action='store_true', help="Include this flag to download chunks. --download requires --chunks.")
    parser.add_argument("--process", action='store_true', help="Include this flag to process chunks into JSON format. --process requires --chunks, --jsons, --start, and --end.")
    parser.add_argument("--generate", action='store_true', help="Include this flag to generate the data frame and save the data as a paraquet file. --generate requires --jsons and --level.")
    parser.add_argument("--chunks", required=True, type=str, help="Path where the chunks are/should be downloaded. The chunks contain the raw data from the AMF.")
    parser.add_argument("--jsons", required=True, type=str, help="Path where the processed JSONs are/should be stored. These JSONs are generated from the chunks.")
    parser.add_argument("--parquet", required=True, type=str, help="Path where the processed CSV are/should be stored as a paraquet file. These files are generated from the JSONs.")
    parser.add_argument("--start", required=True, type=str, help="Start time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--end", required=True, type=str, help="End time of the data extraction in the format %%Y-%%m-%%d %%H:%%M:%%S.")
    parser.add_argument("--level", required=True, type=str, help="Container level to filter on. Could be 'amf', 'support', 'upperlimit', 'amf+support' ")
    parser.add_argument("--type", type=str, required=False, help="Type of feature you want.")
    parser.add_argument("--metric", required=False, type=str, help="Metric name to filter on. Leave empty for all metrics.")
    parser.add_argument("--pod", required=False, type=str, help="Pod name to filter on. Leave empty for all pods.")    
    
    args = parser.parse_args()
    
    orchestra = Orchestrator()
    orchestra.preprocessing(args)
    
    
    
    