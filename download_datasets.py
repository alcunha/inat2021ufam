# Copyright 2021 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
import threading

from absl import app
from absl import flags
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from boto3.s3.transfer import TransferConfig

FLAGS = flags.FLAGS

flags.DEFINE_string(
    's3_bucket_url', default=None,
    help=('URL of the file to be downloaded using the S3://bucket format.'))

flags.DEFINE_string(
    'file_path', default=None,
    help=('The path to the file to download to.'))

flags.DEFINE_string(
    'file_name', default=None,
    help=('The file name to save to. If not defined, it will use from url.'))

flags.DEFINE_integer(
    'max_concurrent_ops', default=10,
    help=('Number of concurrent S3 API transfer operations. Tune it to adjust'
          ' bandwidth usage.'))

flags.mark_flag_as_required('s3_bucket_url')
flags.mark_flag_as_required('file_path')

class ProgressPercentage(object):
  def __init__(self, client, bucket, object_name, file_path):
    self._filename = file_path
    self._size = self._get_content_size(client, bucket, object_name)
    self._seen_so_far = 0
    self._lock = threading.Lock()

  def _get_content_size(self, client, bucket, object_name):
    response = client.head_object(Bucket=bucket,
                                  Key=object_name)['ResponseMetadata']
    return int(response['HTTPHeaders']['content-length'])

  def __call__(self, bytes_amount):
    with self._lock:
      self._seen_so_far += bytes_amount
      percentage = (self._seen_so_far / self._size) * 100
      sys.stdout.write(
        "\r%s  %s / %s  (%.2f%%)" % (
            self._filename, self._seen_so_far, self._size,
            percentage))
      sys.stdout.flush()

def decode_s3_bucket_url(s3_bucket_url):
  s3_bucket_url = s3_bucket_url[len('s3://'):]
  keys = s3_bucket_url.split('/')
  bucket = keys[0]
  object_name = '/'.join(keys[1:])

  return bucket, object_name

def download_from_s3(bucket, object_name, file_path, file_name=None):
  if file_name is None:
    file_name = object_name.split('/')[-1]
  file_path = os.path.join(file_path, file_name)

  s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  progress_callback = ProgressPercentage(s3, bucket, object_name, file_path)
  config = TransferConfig(max_concurrency=FLAGS.max_concurrent_ops)
  s3.download_file(bucket, object_name, file_path,
                   Callback=progress_callback, Config=config)

def main(_):
  bucket, object_name = decode_s3_bucket_url(FLAGS.s3_bucket_url)
  download_from_s3(bucket, object_name, FLAGS.file_path, FLAGS.file_name)

if __name__ == '__main__':
  app.run(main)
