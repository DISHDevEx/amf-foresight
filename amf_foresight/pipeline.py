from load_data import AMFDataProcessor
from feature_engineering import FeatureEngineer
import argparse

class Orchestrator:
    def run_pipeline(self, args):
        processor = AMFDataProcessor()
        print("Processing Data...")
        data = processor.get_data(args.directory, args.metric)
        print("...Completed Processing Data")
        print(" ")
        print('Feature Engineering...')
        feature_engineer = FeatureEngineer()
        data = feature_engineer.value_modifier(data, args.type)
        print('...Completed Feature Engineering')
        print("Summary of Requested data:")
        print(data.describe())
        print("First few entries of requested data:")
        print(data.head())
    
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Process some AMF data.")
    parser.add_argument("--directory", type=str, required=True, help="Path to JSON files directory.")
    parser.add_argument("--metric", type=str, required=False, help="Metric name to filter on.")
    parser.add_argument("--pod", type=str, required=False, help="Pod name to filter on.")
    parser.add_argument("--type", type=str, required=False, help="Type of Feature Engineering")
    args = parser.parse_args()
    
    orchestra = Orchestrator()
    orchestra.run_pipeline(args)
    
    
    
    