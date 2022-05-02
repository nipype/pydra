"""List of helper methods and clients."""
# TODO It would be nice to rewrite this to some class (e.g. AwsManager).

try:
    import boto3
except ImportError:
    pass

import logging

logger = logging.getLogger("pydra")

s3_client: boto3.client = None


def get_s3_client():
    """Lazy getter for S3 client."""

    global s3_client

    if not s3_client:
        s3_client = boto3.client("s3")

    return s3_client
