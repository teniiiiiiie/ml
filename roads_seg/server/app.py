#!/usr/bin/env python3
"""Storage initialization service."""
import os
import time
import logging
import json
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "images")
MAX_RETRIES = 10
RETRY_DELAY = 5

def wait_for_minio():
    """Wait for MinIO to become available."""
    logger.info("Waiting for MinIO to become available...")
    
    for i in range(MAX_RETRIES):
        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=MINIO_ENDPOINT,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                verify=False,
            )
            s3_client.list_buckets()
            logger.info("✅ MinIO is available!")
            return s3_client
        except (EndpointConnectionError, ClientError) as e:
            if i < MAX_RETRIES - 1:
                logger.warning("MinIO not ready yet (attempt %d/%d): %s", i + 1, MAX_RETRIES, e)
                time.sleep(RETRY_DELAY)
            else:
                logger.error("❌ MinIO failed to become available after %d attempts", MAX_RETRIES)
                raise

def initialize_bucket(s3_client):
    """Initialize bucket if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        logger.info("✅ Bucket %s already exists", BUCKET_NAME)
        return False
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            try:
                s3_client.create_bucket(Bucket=BUCKET_NAME)
                logger.info("✅ Bucket %s created successfully", BUCKET_NAME)
                
                # Set bucket policy for public read access (adjust as needed)
                bucket_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": [
                                "s3:GetObject"
                            ],
                            "Resource": [
                                f"arn:aws:s3:::{BUCKET_NAME}/*"
                            ]
                        }
                    ]
                }
                s3_client.put_bucket_policy(
                    Bucket=BUCKET_NAME,
                    Policy=json.dumps(bucket_policy)
                )
                logger.info("✅ Bucket policy set for %s", BUCKET_NAME)
                return True
            except ClientError as e:
                logger.error("❌ Failed to create bucket %s: %s", BUCKET_NAME, e)
                raise
        else:
            logger.error("❌ Error checking bucket %s: %s", BUCKET_NAME, e)
            raise

def main():
    """Main initialization function."""
    try:
        logger.info("🚀 Starting storage initialization...")
        s3_client = wait_for_minio()
        initialize_bucket(s3_client)
        logger.info("✅ Storage initialization completed successfully")
        return 0
    except Exception as e:
        logger.error("❌ Storage initialization failed: %s", e)
        return 1

if __name__ == "__main__":
    exit(main())
