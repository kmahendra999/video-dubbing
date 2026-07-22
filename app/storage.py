"""Object storage.

Everything the pipeline produces lands in MinIO/S3, keyed by project. Workers
pull inputs down and push outputs back, so a worker can live on any host —
which is what makes the GPU stages portable (docs/ARCHITECTURE.md §6).
"""

import logging
import time

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .config import settings

log = logging.getLogger(__name__)

_session = boto3.session.Session()


def client():
    return _session.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name="us-east-1",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def ensure_bucket(retries: int = 40, delay: float = 2.0) -> None:
    """Wait for MinIO and create the bucket. Called on API startup."""
    last = None
    for _ in range(retries):
        try:
            c = client()
            try:
                c.head_bucket(Bucket=settings.s3_bucket)
            except ClientError:
                c.create_bucket(Bucket=settings.s3_bucket)
            log.info("bucket %s ready", settings.s3_bucket)
            return
        except Exception as e:  # MinIO still booting
            last = e
            time.sleep(delay)
    raise RuntimeError(f"object storage unreachable at {settings.s3_endpoint}: {last}")


def upload_file(local_path: str, key: str, content_type: str | None = None) -> None:
    extra = {"ContentType": content_type} if content_type else None
    client().upload_file(local_path, settings.s3_bucket, key, ExtraArgs=extra)


def download_file(key: str, local_path: str) -> None:
    client().download_file(settings.s3_bucket, key, local_path)


def get_object(key: str, range_header: str | None = None):
    kwargs = {"Bucket": settings.s3_bucket, "Key": key}
    if range_header:
        kwargs["Range"] = range_header
    return client().get_object(**kwargs)


def delete_prefix(prefix: str) -> None:
    c = client()
    paginator = c.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
        contents = page.get("Contents") or []
        if not contents:
            continue
        c.delete_objects(
            Bucket=settings.s3_bucket,
            Delete={"Objects": [{"Key": o["Key"]} for o in contents]},
        )
