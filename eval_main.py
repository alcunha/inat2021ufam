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

r"""Tool to evaluate classifiers.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with prefect reproducibility
"""
import os

from absl import app
from absl import flags
from sklearn.metrics import (accuracy_score, confusion_matrix,
    classification_report)
import tensorflow as tf

import dataloader
import model_builder

os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during prediction.'))

flags.DEFINE_string(
    'ckpt_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_string(
    'test_files', default=None,
    help=('A file pattern for TFRecord files'))

flags.DEFINE_bool(
    'use_coordinates_inputs', default=False,
    help=('Use coordinates as aditional input of the model'))

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

flags.DEFINE_string(
    'results_file', default=None,
    help=('File name where the results will be stored.'))

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('test_files')
flags.mark_flag_as_required('results_file')

def _load_model():
  model = model_builder.create(model_name=FLAGS.model_name,
                            num_classes=FLAGS.num_classes,
                            input_size=FLAGS.input_size,
                            use_coordinates_inputs=FLAGS.use_coordinates_inputs,
                            freeze_layers=True)

  checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")
  model.load_weights(checkpoint_path)

  return model

def build_input_data():
  input_data = dataloader.TFRecordWBBoxInputProcessor(
    file_pattern=FLAGS.test_files,
    batch_size=FLAGS.batch_size,
    is_training=False,
    output_size=FLAGS.input_size,
    num_classes=FLAGS.num_classes,
    num_instances=0,
    provide_instance_id=True,
    provide_coordinates_input=FLAGS.use_coordinates_inputs
  )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor, axis=1).numpy()

def predict_classifier(model, dataset):
  labels = []
  predictions = []
  count = 0

  for batch, metadata in dataset:
    pred = model(batch, training=False)
    label, _ = metadata
    labels += list(_decode_one_hot(label))
    predictions += list(_decode_one_hot(pred))

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return labels, predictions

def main(_):
  dataset = build_input_data()
  model = _load_model()
  labels, predictions = predict_classifier(model, dataset)

  accuracy = accuracy_score(labels, predictions)
  conf_matrix = confusion_matrix(labels, predictions)
  report = classification_report(labels, predictions)

  with open("%s.accuracy" % FLAGS.results_file, "w") as text_file:
    text_file.write("%s" % accuracy)

  with open("%s.conf_matrix" % FLAGS.results_file, "w") as text_file:
    text_file.write("%s" % conf_matrix)

  with open("%s.classification_report" % FLAGS.results_file, "w") as text_file:
    text_file.write("%s" % report)

  print("Accuracy: %s" % accuracy)

if __name__ == '__main__':
  app.run(main)
