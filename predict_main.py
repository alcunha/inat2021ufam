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

from absl import app
from absl import flags
import tensorflow as tf

import dataloader
import inatlib
import model_builder

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_integer(
    'num_classes', default=None,
    help=('Number of classes of the model.'))

flags.DEFINE_string(
    'ckpt_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_string(
    'test_files', default=None,
    help=('A file pattern for TFRecord files OR a CSV file containing the list'
          ' of images for test (CSV file must have two columns: filename'
          ' and category id)'))

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory when using a CSV file'))

flags.DEFINE_string(
    'submission_file_path', default=None,
    help=('File name to save predictions on iNat 2021 results format.'))

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('submission_file_path')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('test_files')

BATCH_SIZE = 1

def _load_model():
  model = model_builder.create(model_name=FLAGS.model_name,
                               num_classes=FLAGS.num_classes,
                               input_size=FLAGS.input_size)

  checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")
  model.load_weights(checkpoint_path)

  return model

def build_input_data():
  if FLAGS.test_files.endswith('.csv'):
    if FLAGS.dataset_base_dir is None:
      raise RuntimeError('To use CSV files as input, you must specify'
                         ' --dataset_base_dir')

    input_data = dataloader.CSVInputProcessor(
      csv_file=FLAGS.test_files,
      data_dir=FLAGS.dataset_base_dir,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=FLAGS.input_size,
      num_classes=FLAGS.num_classes,
      provide_instance_id=True,
    )
  else:
    input_data = dataloader.TFRecordWBBoxInputProcessor(
      file_pattern=FLAGS.test_files,
      batch_size=BATCH_SIZE,
      is_training=False,
      output_size=FLAGS.input_size,
      num_classes=FLAGS.num_classes,
      num_instances=0,
      provide_instance_id=True,
    )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def predict_classifier(model, dataset):
  instance_ids = []
  predictions = []
  count = 0

  for batch, metadata in dataset:
    pred = model(batch, training=False)
    _, instanceid = metadata
    instance_ids.append(instanceid[0].numpy().decode('utf8'))
    predictions.append(pred[0].numpy())

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return instance_ids, predictions

def main(_):
  dataset = build_input_data()
  model = _load_model()
  instance_ids, predictions = predict_classifier(model, dataset)
  inatlib.generate_submission(instance_ids,
                              predictions,
                              FLAGS.submission_file_path)

if __name__ == '__main__':
  app.run(main)