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
or infers with reproducibility
"""
import os
import random

from absl import app
from absl import flags
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
    classification_report)
import tensorflow as tf

import dataloader
import geoprior
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

flags.DEFINE_string(
    'geo_prior_ckpt_dir', default=None,
    help=('Location of the checkpoint files for the geo prior model'))

flags.DEFINE_integer(
    'geo_prior_input_size', default=6,
    help=('Input size for the geo prior model'))

flags.DEFINE_bool(
    'use_bn_geo_prior', default=False,
    help=('Include Batch Normalization to the geo prior model'))

flags.DEFINE_integer(
    'embed_dim', default=256,
    help=('Embedding dimension for geo prior model'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('test_files')
flags.mark_flag_as_required('results_file')

def _load_model():
  model = model_builder.create(model_name=FLAGS.model_name,
                            num_classes=FLAGS.num_classes,
                            input_size=FLAGS.input_size,
                            use_coordinates_inputs=FLAGS.use_coordinates_inputs,
                            unfreeze_layers=0)

  checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")
  model.load_weights(checkpoint_path)

  return model

def _load_geo_prior_model():
  if FLAGS.geo_prior_ckpt_dir is not None:
    rand_sample_generator = dataloader.RandSpatioTemporalGenerator()

    geo_prior_model = geoprior.FCNet(
      num_inputs=FLAGS.geo_prior_input_size,
      embed_dim=FLAGS.embed_dim,
      num_classes=FLAGS.num_classes,
      use_bn=FLAGS.use_bn_geo_prior,
      rand_sample_generator=rand_sample_generator)

    checkpoint_path = os.path.join(FLAGS.geo_prior_ckpt_dir, "ckp")
    geo_prior_model.load_weights(checkpoint_path)

    return geo_prior_model
  else:
    return None

def build_input_data():
  include_geo_data = FLAGS.geo_prior_ckpt_dir is not None

  input_data = dataloader.TFRecordWBBoxInputProcessor(
    file_pattern=FLAGS.test_files,
    batch_size=FLAGS.batch_size,
    is_training=False,
    output_size=FLAGS.input_size,
    num_classes=FLAGS.num_classes,
    num_instances=0,
    provide_validity_info_output=include_geo_data,
    provide_coord_date_encoded_input=include_geo_data,
    provide_instance_id=True,
    provide_coordinates_input=FLAGS.use_coordinates_inputs)

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def mix_predictions(cnn_preds, prior_preds, valid):
  valid = tf.expand_dims(valid, axis=-1)
  return cnn_preds*prior_preds*valid + (1 - valid)*cnn_preds

def predict_w_geo_prior(batch, metadata, model, geo_prior_model):
  cnn_input = batch[:-1]
  prior_input = batch[-1]
  label, valid, _ = metadata

  cnn_preds = model(cnn_input, training=False)
  prior_preds = geo_prior_model(prior_input, training=False)
  preds = mix_predictions(cnn_preds, prior_preds, valid)

  return label, preds

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor, axis=1).numpy()

def predict_classifier(model, geo_prior_model, dataset):
  labels = []
  predictions = []
  count = 0

  for batch, metadata in dataset:
    if geo_prior_model is not None:
      label, preds = predict_w_geo_prior(batch,
                                         metadata,
                                         model,
                                         geo_prior_model)
    else:
      preds = model(batch, training=False)
      label, _ = metadata

    labels += list(_decode_one_hot(label))
    predictions += list(_decode_one_hot(preds))

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return labels, predictions

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  dataset = build_input_data()
  model = _load_model()
  geo_prior_model = _load_geo_prior_model()
  labels, predictions = predict_classifier(model, geo_prior_model, dataset)

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
