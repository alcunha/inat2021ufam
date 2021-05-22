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

import math

from absl import flags
import tensorflow as tf

import preprocessing

flags.DEFINE_integer(
    'num_readers', default=64,
    help=('Number of readers of TFRecord files'))

flags.DEFINE_integer(
    'suffle_buffer_size', default=10000,
    help=('Size of the buffer used to shuffle tfrecords'))

flags.DEFINE_bool(
    'use_coordinates_augment', default=False,
    help=('Apply data augmentation to coordinates data'))

flags.DEFINE_string(
    'loc_encode', default='encode_cos_sin',
    help=('Encoding type for location coordinates'))

flags.DEFINE_string(
    'date_encode', default='encode_cos_sin',
    help=('Encoding type for date'))

flags.DEFINE_bool(
    'use_date_feats', default=True,
    help=('Include date features to the encoded coordinates inputs'))

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

def _drop_coordinates(coordinates):
  should_drop = tf.cast(tf.floor(tf.random.uniform(
                              [], seed=FLAGS.random_seed) + 0.5), tf.bool)
  return tf.cond(should_drop,
                  lambda: coordinates,
                  lambda: tf.zeros(shape=coordinates.shape))

def _encode_feat(feat, encode):
  if encode == 'encode_cos_sin':
    return tf.sin(math.pi*feat), tf.cos(math.pi*feat)
  else:
    raise RuntimeError('%s not implemented' % encode)

  return feat

class TFRecordWBBoxInputProcessor:
  def __init__(self,
              file_pattern,
              batch_size,
              num_classes,
              num_instances,
              default_empty_label=0,
              is_training=False,
              use_eval_preprocess=False,
              use_tta=False,
              output_size=224,
              resize_with_pad=False,
              randaug_num_layers=None,
              randaug_magnitude=None,
              provide_validity_info_output=False,
              provide_coord_date_encoded_input=False,
              use_fake_data=False,
              provide_instance_id=False,
              provide_coordinates_input=False,
              batch_drop_remainder=True,
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
    self.provide_validity_info_output = provide_validity_info_output
    self.provide_instance_id = provide_instance_id
    self.provide_coordinates_input = provide_coordinates_input
    self.provide_coord_date_encoded_input = provide_coord_date_encoded_input
    self.preprocess_for_train = is_training and not use_eval_preprocess
    self.use_tta = use_tta
    self.batch_drop_remainder = batch_drop_remainder
    self.seed = seed

    self.feature_description = {
        'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/latitude':
            tf.io.FixedLenFeature((), tf.float32, default_value=0.0),
        'image/longitude':
            tf.io.FixedLenFeature((), tf.float32, default_value=0.0),
        'image/date':
            tf.io.FixedLenFeature((), tf.float32, default_value=0.0),
        'image/valid':
            tf.io.FixedLenFeature((), tf.float32, default_value=0.0),
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

    def _parse_label(features):
      labels = features['image/object/class/label']
      labels = tf.sparse.to_dense(labels)
      label = tf.cond(
          tf.shape(labels)[0] > 0,
          lambda: labels[0],
          lambda: tf.cast(self.default_empty_label, tf.int64))
      label = tf.one_hot(label, self.num_classes)

      return label

    def _image_tta(image):
      rescale = preprocessing.preprocess_image(image,
                      output_size=self.output_size,
                      is_training=False,
                      resize_with_pad=self.resize_with_pad)
      rescale_flip = tf.image.flip_left_right(rescale)
      leftup = preprocessing.preprocess_image(image,
                      output_size=self.output_size,
                      is_training=False,
                      resize_with_pad=self.resize_with_pad,
                      tta='leftup')
      leftup_flip = tf.image.flip_left_right(leftup)
      rightdown = preprocessing.preprocess_image(image,
                      output_size=self.output_size,
                      is_training=False,
                      resize_with_pad=self.resize_with_pad,
                      tta='rightdown')
      rightdown_flip =  tf.image.flip_left_right(rightdown)
      return (rescale, rescale_flip, leftup, leftup_flip, rightdown, \
               rightdown_flip)

    def _parse_single_example(example_proto):
      features = tf.io.parse_single_example(example_proto,
                                            self.feature_description)
      image = tf.io.decode_jpeg(features['image/encoded'])
      label = _parse_label(features)
      instance_id = features['image/source_id']
      latitude = features['image/latitude']
      longitude = features['image/longitude']
      date = features['image/date']
      valid = features['image/valid']

      if self.use_tta:
        image = _image_tta(image)
      else:
        image = preprocessing.preprocess_image(image,
                      output_size=self.output_size,
                      is_training=self.preprocess_for_train,
                      resize_with_pad=self.resize_with_pad,
                      randaug_num_layers=self.randaug_num_layers,
                      randaug_magnitude=self.randaug_magnitude)

      coordinates = tf.stack([latitude, longitude], 0)
      if self.is_training and FLAGS.use_coordinates_augment:
        coordinates = _drop_coordinates(coordinates)

      if self.provide_coord_date_encoded_input:
        lat = _encode_feat(latitude, FLAGS.loc_encode)
        lon = _encode_feat(longitude, FLAGS.loc_encode)
        if FLAGS.use_date_feats:
          date = date*2.0 - 1.0
          date = _encode_feat(date, FLAGS.date_encode)
          coord_date_encoded = tf.concat([lon, lat, date], axis=0)
        else:
          coord_date_encoded = tf.concat([lon, lat], axis=0)
        inputs = (image, coordinates, coord_date_encoded) \
                  if self.provide_coordinates_input \
                  else (image, coord_date_encoded)
      else:
        inputs = (image, coordinates) if self.provide_coordinates_input \
                                      else image

      if self.provide_validity_info_output:
        outputs = (label, valid, instance_id) if self.provide_instance_id \
                                              else (label, valid)
      else:
        outputs = (label, instance_id) if self.provide_instance_id else label

      return inputs, outputs
    dataset = dataset.map(_parse_single_example, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.batch_drop_remainder)

    if self.use_fake_data:
      dataset.take(1).repeat()

    return dataset, self.num_instances, self.num_classes


class RandSpatioTemporalGenerator:
  def __init__(self, rand_type='spherical'):
    self.rand_type = rand_type

  def _encode_feat(self, feat, encode):
    if encode == 'encode_cos_sin':
      feats = tf.concat([
        tf.sin(math.pi*feat),
        tf.cos(math.pi*feat)], axis=1)
    else:
      raise RuntimeError('%s not implemented' % encode)

    return feats

  def get_rand_samples(self, batch_size):
    if self.rand_type == 'spherical':
      rand_feats = tf.random.uniform(shape=(batch_size, 3),
                                    dtype=tf.float32)
      theta1 = 2.0*math.pi*rand_feats[:,0]
      theta2 = tf.acos(2.0*rand_feats[:,1] - 1.0)
      lat = 1.0 - 2.0*theta2/math.pi
      lon = (theta1/math.pi) - 1.0
      time = rand_feats[:,2]*2.0 - 1.0

      lon = tf.expand_dims(lon, axis=-1)
      lat = tf.expand_dims(lat, axis=-1)
      time = tf.expand_dims(time, axis=-1)
    else:
      raise RuntimeError('%s rand type not implemented' % self.rand_type)

    lon = self._encode_feat(lon, FLAGS.loc_encode)
    lat = self._encode_feat(lat, FLAGS.loc_encode)
    time = self._encode_feat(time, FLAGS.date_encode)

    if FLAGS.use_date_feats:
      return tf.concat([lon, lat, time], axis=1)
    else:
      return tf.concat([lon, lat], axis=1)
