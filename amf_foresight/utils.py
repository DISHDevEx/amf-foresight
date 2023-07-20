import boto3
from setup_logger import setup_logger

setup_logger()

class Utils:
    
    def __init__(self):
        """
        class constructor.
        """
        self.bucket_name = 'open5gs-amf-foresight'
        self.client = boto3.client('s3')
    
    def upload_model(self, local_path, key):
        """
        upload any file to s3.
        """
        try:
            self.client.upload_file(local_path, key, self.bucket_name)
        except ClientError as error:
            logging.error(error)
            return False
        return print(f'Uploaded Path: s3://{bucket_name}/{key}')


    def download_model(self, local_path, key):
        """ 
        downloads any file to s3.
        """
        with open(local_path, 'wb') as file:
            self.client.download_fileobj(self.bucket_name, key, file)
        return print(f"Downloaded file to: {local_path}")

if __name__ == "__main__":    
    bucket_name = 'open5gs-amf-foresight'
    model_file_path = 'models/model100.pkl'
    s3_key = 'models/model.pkl'

    uploader = Utils()
    uploader.download_model(model_file_path, s3_key)