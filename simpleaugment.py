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

import tensorflow as tf
import tensorflow_addons as tfa

import utils

def distort_color(image, seed=None):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
  image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
  image = tf.image.random_hue(image, max_delta=0.2, seed=seed)

  return tf.clip_by_value(image, 0.0, 1.0)

def random_rotation(image, deg=20, seed=None):
  rotation_theta = utils.deg2rad(deg)

  random_deg = tf.random.uniform(
    shape=[1],
    minval=-rotation_theta,
    maxval=rotation_theta,
    seed=seed)

  image = tfa.image.rotate(image, random_deg, interpolation='BILINEAR')

  return image

def distort_image_with_simpleaugment(image, seed=None):

  tf.compat.v1.logging.info('Using SimpleAug.')

  image = distort_color(image, seed=seed)
  image = random_rotation(image, seed=seed)

  return image
