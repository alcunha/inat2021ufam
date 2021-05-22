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

r"""Tool to generate predictions using classifiers.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import os
import random

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import dataloader
import geoprior
import inatlib
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

flags.DEFINE_bool(
    'use_tta', default=False,
    help=('Use test time augmentation (6-crop: full, left-up, right-down and'
          ' theirs flip version) for image prediction'))

flags.DEFINE_string(
    'test_files', default=None,
    help=('A file pattern for TFRecord files'))

flags.DEFINE_bool(
    'use_coordinates_inputs', default=False,
    help=('Use coordinates as aditional input of the model'))

flags.DEFINE_string(
    'submission_file_path', default=None,
    help=('File name to save predictions on iNat 2021 results format.'))

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

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
flags.mark_flag_as_required('submission_file_path')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('test_files')

def _load_model():
  model = model_builder.create(model_name=FLAGS.model_name,
                            num_classes=FLAGS.num_classes,
                            input_size=FLAGS.input_size,
                            use_coordinates_inputs=FLAGS.use_coordinates_inputs,
                            unfreeze_layers=0)

  checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")
  model.load_weights(checkpoint_path)

  if FLAGS.use_tta:
    inputs = [tf.keras.Input(shape=(FLAGS.input_size, FLAGS.input_size, 3))
              for x in range(6)]
    outputs = [model(img_input, training=False) for img_input in inputs]
    outputs = tf.keras.layers.Average()(outputs)

    tta_model = tf.keras.models.Model(inputs=inputs, outputs=[outputs])
    model = tta_model

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
    batch_drop_remainder=False,
    output_size=FLAGS.input_size,
    num_classes=FLAGS.num_classes,
    num_instances=0,
    provide_validity_info_output=include_geo_data,
    provide_coord_date_encoded_input=include_geo_data,
    provide_instance_id=True,
    use_tta=FLAGS.use_tta,
    provide_coordinates_input=FLAGS.use_coordinates_inputs
  )

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def mix_predictions(cnn_preds, prior_preds, valid):
  valid = tf.expand_dims(valid, axis=-1)
  return cnn_preds*prior_preds*valid + (1 - valid)*cnn_preds

def predict_w_geo_prior(batch, metadata, model, geo_prior_model):
  cnn_input = batch[:-1]
  prior_input = batch[-1]
  _, valid, instanceid = metadata

  cnn_preds = model(cnn_input, training=False)
  prior_preds = geo_prior_model(prior_input, training=False)
  preds = mix_predictions(cnn_preds, prior_preds, valid)

  return instanceid, preds

def predict_classifier(model, geo_prior_model, dataset):
  instance_ids = []
  predictions = []
  count = 0

  for batch, metadata in dataset:
    if geo_prior_model is not None:
      instanceid, preds = predict_w_geo_prior(batch,
                                         metadata,
                                         model,
                                         geo_prior_model)
    else:
      preds = model(batch, training=False)
      _, instanceid = metadata
    instance_ids += list(instanceid.numpy().astype('U13'))
    predictions += list(preds.numpy())

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return instance_ids, predictions

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  dataset = build_input_data()
  model = _load_model()
  geo_prior_model = _load_geo_prior_model()
  instance_ids, predictions = predict_classifier(model,
                                                 geo_prior_model,
                                                 dataset)
  inatlib.generate_submission(instance_ids,
                              predictions,
                              FLAGS.submission_file_path)

if __name__ == '__main__':
  app.run(main)
