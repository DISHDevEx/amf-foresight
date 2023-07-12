import pandas as pd
import argparse
import os

class FeatureEngineer:
    
    def get_data(self, data, feature_type):
        data = self.read_data(data)
        data = self.value_modifier(data, feature_type)
        return data
    
    def read_data(self, path):
        """
        This function takes in a path to a CSV of filtered AMF Data and returns a dataframe
        :param directory: Path to CSV with filtered AMF Data
        """
        data = pd.read_csv(path)
        return data
    
    def value_modifier(self, data, metric):
        """
        This function takes in a pandas dataframe and modifies the values based on the type of metric
        :param directory: Path to JSON files
        """
        if metric == 'memory':
            data['values'] = data['values'] / 1048576
        elif metric == 'cpu':
            data['values'] = data['values']
        elif metric == 'cpu_utilization':
            data['time_diff'] = data['date_col'].diff().dt.total_seconds()
            data['usage_diff'] = data['values'].diff()
            data['utilization'] = (data['usage_diff'].diff()/data['time_diff']) * 100
            data.fillna(0, inplace=True)
        return data
            

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Feature Engineer AMF data.")
    parser.add_argument("--data", type=str, required=False, help="Path to filtered AMF data")
    parser.add_argument("--type", type=str, required=False, help="Type of Feature Engineering")
    args = parser.parse_args()
    
    feature_engineer = FeatureEngineer()
    data = feature_engineer.get_data(args.data, args.type)
    
    print("Summary of Requested data:")
    print(data.describe())
    print("First few entries of requested data:")
    print(data.head())
    filename = "sample::" + str(os.path.basename(__file__)) + "::" + args.data.split("::")[-1] + ";type:" + args.type + ".csv"
    path = "csv/" + filename                                                                                 
    data.to_csv(path, index=False)
    print("Data Saved to: ", path)
    
    
    
    