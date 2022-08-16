'''
This class is to manage data transfer.
'''
import boto3
import os
import logging
from botocore.exceptions import ClientError

session = boto3.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)



