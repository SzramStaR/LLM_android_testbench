"""
Cloudflare R2 Bucket Downloader
=================================

Download objects from a Cloudflare R2 bucket to a local directory.
"""

import argparse
import datetime
import logging
import os

import boto3
from botocore.exceptions import ClientError


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_r2_client(account_id: str, access_key: str, secret_key: str):
    """Create and return a Cloudflare R2 client"""
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    return client


def list_all_objects(client, bucket_name: str):
    """List all objects in the given bucket."""
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                objects.append(obj)
    except ClientError as e:
        logger.error(f"Error listing objects: {e}")
        raise
    return objects


def list_objects_on_date(client, bucket_name: str, date_filter: datetime.date):
    """List objects whose LastModified falls on the given UTC calendar date"""
    all_objects = list_all_objects(client, bucket_name)
    filtered = []
    for obj in all_objects:
        last_modified = obj.get("LastModified")
        if isinstance(last_modified, datetime.datetime):
            if last_modified.date() == date_filter:
                filtered.append(obj)
    return filtered


def download_object(client, bucket_name: str, key: str, local_dir: str) -> str:
    """Download a single object from the bucket if not already present"""
    local_path = os.path.join(local_dir, key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return "skipped"
    try:
        logger.info(f"Downloading {key}")
        client.download_file(bucket_name, key, local_path)
        return "downloaded"
    except ClientError as e:
        logger.error(f"Error downloading {key}: {e}")
        return "error"


def download_objects(
    client, bucket_name: str, local_dir: str, date_filter: datetime.date | None
):
    """Download objects from the bucket, optionally filtered by a specific date (UTC)"""
    if date_filter:
        objects = list_objects_on_date(client, bucket_name, date_filter)
        logger.info(f"Filtered to {len(objects)} objects on {date_filter.isoformat()}")
    else:
        objects = list_all_objects(client, bucket_name)
        logger.info(f"Found {len(objects)} objects in bucket '{bucket_name}'")

    if not objects:
        logger.info("No objects to download.")
        return 0, 0, 0

    os.makedirs(local_dir, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    for obj in objects:
        key = obj["Key"]
        status = download_object(client, bucket_name, key, local_dir)
        if status == "downloaded":
            downloaded += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
    return downloaded, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download objects from a Cloudflare R2 bucket"
    )
    parser.add_argument(
        "--account-id",
        default=os.getenv("R2_ACCOUNT_ID"),
        help="Cloudflare account ID or R2_ACCOUNT_ID env",
    )
    parser.add_argument(
        "--access-key",
        default=os.getenv("R2_ACCESS_KEY_ID"),
        help="R2 access key or R2_ACCESS_KEY_ID env",
    )
    parser.add_argument(
        "--secret-key",
        default=os.getenv("R2_SECRET_ACCESS_KEY"),
        help="R2 secret key or R2_SECRET_ACCESS_KEY env",
    )
    parser.add_argument("--bucket", required=True, help="Bucket name to download from")
    parser.add_argument(
        "--date",
        help="Optional filter date (UTC) in YYYY-MM-DD to download only that day",
    )
    args = parser.parse_args()

    if not args.account_id or not args.access_key or not args.secret_key:
        parser.error(
            "Missing credentials. Provide via flags or env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
        )

    parsed_date = None
    if args.date:
        try:
            parsed_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            parser.error("--date must be in YYYY-MM-DD format")
    args.parsed_date = parsed_date
    return args


def main():
    try:
        args = parse_args()
        client = create_r2_client(args.account_id, args.access_key, args.secret_key)
        download_dir = os.path.join("./r2_downloads", args.bucket)

        if args.parsed_date:
            logger.info(
                f"Starting download from bucket '{args.bucket}' to '{download_dir}' for files on {args.parsed_date.isoformat()}"
            )
        else:
            logger.info(
                f"Starting download from bucket '{args.bucket}' to '{download_dir}' (no date filter)"
            )

        downloaded, skipped, failed = download_objects(
            client, args.bucket, download_dir, args.parsed_date
        )
        if failed:
            logger.info(
                f"Downloaded {downloaded} new files, skipped {skipped} existing, {failed} failed."
            )
        else:
            logger.info(
                f"Downloaded {downloaded} new files, skipped {skipped} existing."
            )
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
