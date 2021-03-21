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

import json
from absl import flags

import tensorflow as tf
import numpy as np
from tf_slim import tfexample_decoder as slim_example_decoder

import preprocessing

flags.DEFINE_integer(
    'num_readers', default=64,
    help=('Number of readers of TFRecord files'))

flags.DEFINE_integer(
    'suffle_buffer_size', default=10000,
    help=('Size of the buffer used to shuffle tfrecords'))

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

def _to_float(value):
  if value is None:
    return 0.0
  else:
    return float(value)

def _generate_coordinates_idx(annotations_file, id_field):
  with tf.io.gfile.GFile(annotations_file, 'r') as json_file:
    json_data = json.load(json_file)

  coordinates_index = {}
  for image in json_data['images']:
    image_id = int(image[id_field])
    coordinates_index[image_id] = np.array([_to_float(image['latitude'])/90,
                                            _to_float(image['longitude'])/180])

  return coordinates_index

class TFRecordWBBoxInputProcessor:
  def __init__(self,
              file_pattern,
              batch_size,
              num_classes,
              num_instances,
              default_empty_label=0,
              is_training=False,
              use_eval_preprocess=False,
              output_size=224,
              resize_with_pad=False,
              randaug_num_layers=None,
              randaug_magnitude=None,
              use_fake_data=False,
              provide_instance_id=False,
              annotations_file=None,
              provide_coordinates_input=False,
              seed=None):
    self.file_pattern = file_pattern
    self.batch_size = batch_size
    self.is_training = is_training
    self.output_size = output_size
    self.resize_with_pad = resize_with_pad
    self.num_classes = num_classes
    self.num_instances = num_instances
    self.default_empty_label = default_empty_label
    self.randaug_num_layers = randaug_num_layers
    self.randaug_magnitude = randaug_magnitude
    self.use_fake_data = use_fake_data
    self.provide_instance_id = provide_instance_id
    self.provide_coordinates_input = provide_coordinates_input
    self.coordinates_index = None
    self.preprocess_for_train = is_training and not use_eval_preprocess
    self.seed = seed

    self.feature_description = {
        'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/filename':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    self.bbox_handler = slim_example_decoder.BoundingBox(
                                             ['ymin', 'xmin', 'ymax', 'xmax'],
                                             'image/object/bbox/')
    if provide_coordinates_input:
      if annotations_file is None:
        raise RuntimeError('To provide coordinates input, you must provide '
                           'annotations_file.')
      else:
        self.coordinates_index = _generate_coordinates_idx(annotations_file,
                                                           'id')

  def make_source_dataset(self):

    filenames = tf.io.gfile.glob(self.file_pattern)
    dataset_files = tf.data.Dataset.list_files(self.file_pattern,
                                               shuffle=self.is_training,
                                               seed=self.seed)

    num_readers = FLAGS.num_readers
    if num_readers > len(filenames):
      num_readers = len(filenames)
      tf.compat.v1.logging.info('num_readers has been reduced to %d to match'
                       ' input file shards.' % num_readers)
    dataset = dataset_files.interleave(
      lambda x: tf.data.TFRecordDataset(x,
                        buffer_size=8 * 1000 * 1000).prefetch(AUTOTUNE),
      cycle_length=num_readers,
      num_parallel_calls=AUTOTUNE)

    if self.is_training:
      dataset = dataset.shuffle(FLAGS.suffle_buffer_size, seed=self.seed)
      dataset = dataset.repeat()

    def _parse_bboxes(features):
      keys_to_tensors = {key: features[key] for key in self.bbox_handler.keys}
      return self.bbox_handler.tensors_to_item(keys_to_tensors)

    def _parse_label(features):
      labels = features['image/object/class/label']
      labels = tf.sparse.to_dense(labels)
      label = tf.cond(
          tf.shape(labels)[0] > 0,
          lambda: labels[0],
          lambda: tf.cast(self.default_empty_label, tf.int64))
      label = tf.one_hot(label, self.num_classes)

      return label

    def _parse_single_example(example_proto):
      features = tf.io.parse_single_example(example_proto,
                                            self.feature_description)
      image = tf.io.decode_jpeg(features['image/encoded'])
      bboxes = _parse_bboxes(features)
      label = _parse_label(features)
      instance_id = features['image/source_id']

      image = preprocessing.preprocess_image(image,
                    output_size=self.output_size,
                    is_training=self.preprocess_for_train,
                    resize_with_pad=self.resize_with_pad,
                    randaug_num_layers=self.randaug_num_layers,
                    randaug_magnitude=self.randaug_magnitude)

      if self.provide_coordinates_input:
        def _get_coords(instance_id):
          instance_id = int(instance_id.numpy().decode('utf8'))
          coords = self.coordinates_index[instance_id]
          return coords

        coordinates = tf.py_function(_get_coords, [instance_id],
                                     Tout=tf.float32)
        inputs = (image, coordinates)
      else:
        inputs = image

      if self.provide_instance_id:
        return inputs, (label, instance_id)

      return inputs, label

    dataset = dataset.map(_parse_single_example, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)

    if self.use_fake_data:
      dataset.take(1).repeat()

    return dataset, self.num_instances, self.num_classes
