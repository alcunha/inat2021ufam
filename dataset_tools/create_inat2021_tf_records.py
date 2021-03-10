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
import random
import hashlib
import json
import contextlib2

from absl import app
from absl import flags
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

FLAGS = flags.FLAGS

flags.DEFINE_string(
      'annotations_file', default=None,
      help=('Json file containing annotations in COCO format')
)

flags.DEFINE_string(
    'dataset_base_dir', default=None,
    help=('Path to images dataset base directory.'))

flags.DEFINE_string(
    'output_dir', default=None,
    help=('Path to save tfrecords to.')
)

flags.DEFINE_integer(
    'images_per_shard', default=1100,
    help=('Number of images per shard')
)

flags.DEFINE_bool(
    'shufle_images', default=True,
    help=('Shufle images before to write to tfrecords')
)

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments')
  )

flags.mark_flag_as_required('annotations_file')
flags.mark_flag_as_required('dataset_base_dir')
flags.mark_flag_as_required('output_dir')

def create_tf_example(image,
                      dataset_base_dir,
                      annotations,
                      category_index):
  filename = image['file_name'].split('/')[-1]
  image_id = image['id']

  image_path = os.path.join(dataset_base_dir, image['file_name'])
  if not tf.io.gfile.exists(image_path):
    return None

  with tf.io.gfile.GFile(image_path, 'rb') as image_file:
    encoded_image_data = image_file.read()
  key = hashlib.sha256(encoded_image_data).hexdigest()

  height = image['height']
  width = image['width']

  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []
  for annotation in annotations:
    category_id = annotation['category_id']
    classes_text.append(category_index[category_id]['name'].encode('utf8'))
    classes.append(category_id)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))

  return tf_example

def create_tf_record_from_images_list(images,
                                      annotations_index,
                                      dataset_base_dir,
                                      category_index,
                                      output_path):
  num_shards = 1 + (len(images) // FLAGS.images_per_shard)
  total_image_skipped = 0

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)

    for index, image in enumerate(images):
      image_id = image['id']
      if image_id not in annotations_index:
        annotations_index[image_id] = []
      tf_example = create_tf_example(image,
                                     dataset_base_dir,
                                     annotations_index[image_id],
                                     category_index)

      if tf_example is not None:
        output_shard_index = index % num_shards
        output_tfrecords[output_shard_index].write(
            tf_example.SerializeToString())
      else:
        total_image_skipped += 1

    tf.compat.v1.logging.info('%d images not found.', total_image_skipped)

def _get_annotations_index(annotations):
  annotations_index = {}
  for annotation in annotations:
    image_id = annotation['image_id']
    if image_id not in annotations_index:
      annotations_index[image_id] = []
    annotations_index[image_id].append(annotation)

  return annotations_index

def _create_inat_tf_record(annotations_file,
                           dataset_base_dir,
                           output_path):

  with tf.io.gfile.GFile(annotations_file, 'r') as json_file:
    json_data = json.load(json_file)

  images = json_data['images']
  annot_index = _get_annotations_index(json_data['annotations'])
  category_index = label_map_util.create_category_index(json_data['categories'])

  split = annotations_file.split('/')[-1][:-len('.json')]
  tfrecord_path = os.path.join(output_path, 'inat_%s.record' % split)

  if FLAGS.shufle_images:
    random.shuffle(images)

  create_tf_record_from_images_list(images,
                                    annot_index,
                                    dataset_base_dir,
                                    category_index,
                                    tfrecord_path)

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  _create_inat_tf_record(FLAGS.annotations_file,
                         FLAGS.dataset_base_dir,
                         FLAGS.output_dir)

if __name__ == '__main__':
  app.run(main)
