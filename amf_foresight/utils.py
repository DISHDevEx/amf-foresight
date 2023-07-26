import boto3
import pandas as pd
from io import BytesIO
import logging
from setup_logger import setup_logger
from botocore.exceptions import ClientError
import os
import inspect

setup_logger()

class Utils:
    
    def __init__(self):
        """
        class constructor.
        """
        self.bucket_name = 'open5gs-amf-foresight'
        self.resource = boto3.resource('s3')
        self.client = boto3.client('s3')
    
    def upload_file(self, local_path, key):
        """
        upload any file to s3.
        """
        try:
            self.client.upload_file(local_path, self.bucket_name, key)
        except ClientError as error:
            logging.error(error)
            return False
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::(S3) Uploaded File to: s3://{self.bucket_name}/{key}")
        return True


    def download_file(self, local_path, key):
        """ 
        downloads any file to s3.
        """
        with open(local_path, 'wb') as file:
            self.client.download_fileobj(self.bucket_name, key, file)
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::Downloaded file to: {local_path}")
        return True
    
    def read_parquet_to_pandas_df(self, key):
        """
        reads parquet to df.
        """
        buffer = BytesIO()
        object_ = self.resource.Object(self.bucket_name, key)
        object_.download_fileobj(buffer)
        dataframe = pd.read_parquet(buffer)
        return dataframe
    
    def pandas_dataframe_to_s3(self, input_datafame, key):
        """
        upload a pandas dataframe to s3.
        """
        out_buffer = BytesIO()
        input_datafame.to_parquet(out_buffer, index=False)
        try:
            self.client.put_object(Bucket=self.bucket_name, Key=key, Body=out_buffer.getvalue())
        except ClientError as error:
            logging.error(error)
            return False
        logging.info(f"{os.path.basename(__file__)}::{self.__class__.__name__}::{inspect.currentframe().f_code.co_name}::(S3) Uploaded Dataframe to: s3://{self.bucket_name}/{key}")
        return True
    
