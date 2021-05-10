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

r"""Tool to train classifiers.

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
import model_builder
import train_image_classifier
import utils

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'training_files', default=None,
    help=('A file pattern for TFRecord files'))

flags.DEFINE_integer(
    'num_training_instances', default=None,
    help=('Number of training instances'))

flags.DEFINE_string(
    'validation_files', default=None,
    help=('A file pattern for TFRecord files'))

flags.DEFINE_integer(
    'num_validation_instances', default=None,
    help=('Number of validation instances'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_integer(
    'input_size_stage3', default=260,
    help=('Input size of the model on the stage 3 (fix train/test resolution)'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('When 0, no smoothing occurs. When > 0, we apply Label Smoothing to'
          ' the labels during training using this value for parameter e.'))

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during training.'))

flags.DEFINE_integer(
    'randaug_num_layers', default=None,
    help=('Number of operations to be applied by Randaugment'))

flags.DEFINE_integer(
    'randaug_magnitude', default=None,
    help=('Magnitude for operations on Randaugment.'))

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_string(
    'load_checkpoint', default=None,
    help=('Path to weights checkpoint to be loaded into the model'))

flags.DEFINE_string(
    'base_model_weights', default='imagenet',
    help=('Path to h5 weights file to be loaded into the base model during'
          ' model build procedure.'))

flags.DEFINE_integer(
    'num_classes', default=None,
    help=('Number of classes to train the model on. If not passed, it will be'
           'inferred from data'))

flags.DEFINE_float(
    'lr_stage1', default=0.1,
    help=('Initial learning rate for stage 1'))

flags.DEFINE_float(
    'lr_stage2', default=0.1,
    help=('Initial learning rate for stage 2'))

flags.DEFINE_float(
    'lr_stage3', default=0.008,
    help=('Initial learning rate for stage 3'))

flags.DEFINE_float(
    'momentum', default=0,
    help=('Momentum for SGD optimizer'))

flags.DEFINE_bool(
    'use_scaled_lr', default=True,
    help=('Scale the initial learning rate by batch size'))

flags.DEFINE_bool(
    'use_cosine_decay', default=True,
    help=('Apply cosine decay during training'))

flags.DEFINE_float(
    'warmup_epochs', default=0.3,
    help=('Duration of warmp of learning rate in epochs. It can be a'
          ' fractionary value as long will be converted to steps.'))

flags.DEFINE_integer(
    'epochs_stage1', default=4,
    help=('Number of epochs to training during stage 1. Set to 0 do skip this'
          ' stage.'))

flags.DEFINE_integer(
    'epochs_stage2', default=10,
    help=('Number of epochs to training during stage 2. Set to 0 do skip this'
          ' stage.'))

flags.DEFINE_integer(
    'epochs_stage3', default=2,
    help=('Number of epochs to training during stage 3. Set to 0 do skip this'
          ' stage.'))

flags.DEFINE_integer(
    'unfreeze_layers', default=0,
    help=('Number of layers to unfreeze at the end of the image base model '
          ' during stage 3.'))

flags.DEFINE_bool(
    'use_coordinates_inputs', default=False,
    help=('Use coordinates as aditional input of the model'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('training_files')
flags.mark_flag_as_required('num_training_instances')
flags.mark_flag_as_required('model_dir')

def build_tfrecord_input_data(file_pattern,
                              num_instances,
                              input_size,
                              is_training=False,
                              use_eval_preprocess=False):
  if FLAGS.num_classes is None:
    raise RuntimeError('To use TFRecords as input, you must specify'
                       ' --num_classes')

  input_data = dataloader.TFRecordWBBoxInputProcessor(
    file_pattern=file_pattern,
    batch_size=FLAGS.batch_size,
    num_classes=FLAGS.num_classes,
    num_instances=num_instances,
    is_training=is_training,
    use_eval_preprocess=use_eval_preprocess,
    output_size=input_size,
    randaug_num_layers=FLAGS.randaug_num_layers,
    randaug_magnitude=FLAGS.randaug_magnitude,
    provide_coordinates_input=FLAGS.use_coordinates_inputs,
    seed=FLAGS.random_seed,
  )

  return input_data.make_source_dataset()

def get_model(num_classes, input_size, unfreeze_layers):
  model = model_builder.create(
    model_name=FLAGS.model_name,
    num_classes=num_classes,
    input_size=input_size,
    unfreeze_layers=unfreeze_layers,
    use_coordinates_inputs=FLAGS.use_coordinates_inputs,
    base_model_weights=FLAGS.base_model_weights,
    seed=FLAGS.random_seed)

  return model

def train_model(model,
                lr,
                epochs,
                model_dir,
                train_data_and_size,
                val_data_and_size,
                strategy):

  if FLAGS.use_scaled_lr:
    lr = lr * FLAGS.batch_size / 256

  _, train_size = train_data_and_size
  warmup_steps = int(FLAGS.warmup_epochs * (train_size // FLAGS.batch_size))

  hparams = train_image_classifier.get_default_hparams()
  hparams = hparams._replace(
    lr=lr,
    momentum=FLAGS.momentum,
    epochs=epochs,
    warmup_steps=warmup_steps,
    use_cosine_decay=FLAGS.use_cosine_decay,
    batch_size=FLAGS.batch_size,
    model_dir=model_dir,
    label_smoothing=FLAGS.label_smoothing
  )

  history = train_image_classifier.train_model(
    model,
    hparams,
    train_data_and_size,
    val_data_and_size,
    strategy
  )

  return history

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  if utils.xor(FLAGS.randaug_num_layers is None,
               FLAGS.randaug_magnitude is None):
    raise RuntimeError('To apply Randaugment during training you must specify'
                       ' both --randaug_num_layers and --randaug_magnitude')

  set_random_seeds()

  dataset, num_instances, num_classes = build_tfrecord_input_data(
    FLAGS.training_files,
    FLAGS.num_training_instances,
    FLAGS.input_size,
    is_training=True)

  if FLAGS.validation_files is not None:
    if FLAGS.num_validation_instances is None:
      raise RuntimeError('Must specify --num_validation_instances when using'
                        'TFREcords for validation')

    val_dataset, val_num_instances, _ = build_tfrecord_input_data(
      FLAGS.validation_files,
      FLAGS.num_validation_instances,
      FLAGS.input_size,
      is_training=False)
  else:
    val_dataset = None
    val_num_instances = 0

  if FLAGS.num_classes is not None:
    num_classes = FLAGS.num_classes

  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  prev_checkpoint = FLAGS.load_checkpoint

  # Stage 1 - we train only the classifier layer
  if FLAGS.epochs_stage1 > 0:
    with strategy.scope():
      model = get_model(num_classes, FLAGS.input_size, unfreeze_layers=0)
    model.summary()
    if prev_checkpoint is not None:
      checkpoint_path = os.path.join(prev_checkpoint, "ckp")
      model.load_weights(checkpoint_path)
    train_model(model,
                lr=FLAGS.lr_stage1,
                epochs=FLAGS.epochs_stage1,
                model_dir=os.path.join(FLAGS.model_dir, 'stage1'),
                train_data_and_size=(dataset, num_instances),
                val_data_and_size=(val_dataset, val_num_instances),
                strategy=strategy)
    prev_checkpoint = os.path.join(FLAGS.model_dir, 'stage1')

  # Stage 2 - we fine tune all layers
  if FLAGS.epochs_stage2 > 0:
    with strategy.scope():
      model = get_model(num_classes, FLAGS.input_size, unfreeze_layers=-1)
    model.summary()
    if prev_checkpoint is not None:
      checkpoint_path = os.path.join(prev_checkpoint, "ckp")
      model.load_weights(checkpoint_path)
    train_model(model,
                lr=FLAGS.lr_stage2,
                epochs=FLAGS.epochs_stage2,
                model_dir=os.path.join(FLAGS.model_dir, 'stage2'),
                train_data_and_size=(dataset, num_instances),
                val_data_and_size=(val_dataset, val_num_instances),
                strategy=strategy)
    prev_checkpoint = os.path.join(FLAGS.model_dir, 'stage2')

  # Stage 3 - we fine tune the last N layers and use higher input size; we use
  # the evaluation preprocessing of images during training
  if FLAGS.epochs_stage3 > 0:
    dataset, _, _ = build_tfrecord_input_data(
      FLAGS.training_files,
      FLAGS.num_training_instances,
      FLAGS.input_size_stage3,
      is_training=True,
      use_eval_preprocess=True)

    if FLAGS.validation_files is not None:
      val_dataset, _, _ = build_tfrecord_input_data(
        FLAGS.validation_files,
        FLAGS.num_validation_instances,
        FLAGS.input_size_stage3,
        is_training=False,
        use_eval_preprocess=True)
    else:
      val_dataset = None
    with strategy.scope():
      model = get_model(num_classes,
                        FLAGS.input_size_stage3,
                        unfreeze_layers=FLAGS.unfreeze_layers)
    model.summary()
    if prev_checkpoint is not None:
      checkpoint_path = os.path.join(prev_checkpoint, "ckp")
      model.load_weights(checkpoint_path)
    train_model(model,
                lr=FLAGS.lr_stage3,
                epochs=FLAGS.epochs_stage3,
                model_dir=FLAGS.model_dir,
                train_data_and_size=(dataset, num_instances),
                val_data_and_size=(val_dataset, val_num_instances),
                strategy=strategy)

if __name__ == '__main__':
  app.run(main)
