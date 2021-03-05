import os

from absl import app
from absl import flags
import boto3
from botocore import UNSIGNED
from botocore.client import Config

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

flags.mark_flag_as_required('s3_bucket_url')
flags.mark_flag_as_required('file_path')

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
  s3.download_file(bucket, object_name, file_path)

def main(_):
  bucket, object_name = decode_s3_bucket_url(FLAGS.s3_bucket_url)
  download_from_s3(bucket, object_name, FLAGS.file_path, FLAGS.file_name)

if __name__ == '__main__':
  app.run(main)
