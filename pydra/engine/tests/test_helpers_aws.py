import boto3
import botocore
import pytest

from ..helpers_aws import (
    s3_client,
    get_s3_client,
)


def test_get_s3_client():
    assert s3_client is None
    r = get_s3_client()

    assert isinstance(r, botocore.client.BaseClient)
