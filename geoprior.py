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

def _create_res_layer(inputs, embed_dim, use_bn=False):
  x = tf.keras.layers.Dense(embed_dim)(inputs)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dropout(rate=0.5)(x)
  x = tf.keras.layers.Dense(embed_dim)(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  outputs = tf.keras.layers.add([inputs, x])

  return outputs

def _create_loc_encoder(inputs, embed_dim, num_res_blocks, use_bn=False):
  x = tf.keras.layers.Dense(embed_dim)(inputs)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  for _ in range(num_res_blocks):
    x = _create_res_layer(x, embed_dim)

  return x

def _create_FCNet(num_inputs,
                  num_classes,
                  embed_dim,
                  num_res_blocks=4,
                  use_bn=False):
  inputs = tf.keras.Input(shape=(num_inputs,))
  loc_embed = _create_loc_encoder(inputs, embed_dim, num_res_blocks, use_bn)
  class_embed = tf.keras.layers.Dense(num_classes,
                                      activation='sigmoid',
                                      use_bias=False)(loc_embed)

  model = tf.keras.models.Model(inputs=inputs, outputs=class_embed)

  return model

class FCNet(tf.keras.Model):
  def __init__(self, num_inputs, embed_dim, num_classes, rand_sample_generator,
                num_users=0, num_res_blocks=4, use_bn=False):
    super(FCNet, self).__init__()
    if num_users > 1:
      raise RuntimeError('Users branch not implemented')
    self.model = _create_FCNet(num_inputs, num_classes, embed_dim,
                               num_res_blocks=num_res_blocks, use_bn=use_bn)
    self.rand_sample_generator = rand_sample_generator

  def call(self, inputs):
    return self.model(inputs)

  def train_step(self, data):
    x, y = data
    batch_size = tf.shape(x)[0]

    rand_samples = self.rand_sample_generator.get_rand_samples(batch_size)

    # The localization loss on the paper for the random points is equivalent to
    # the Binary Cross Entropy considering all labels as zero
    rand_labels = tf.zeros(shape=y.shape)

    combined_inputs = tf.concat([x, rand_samples], axis=0)
    y_true = tf.concat([y, rand_labels], axis=0)

    with tf.GradientTape() as tape:
      y_pred = self(combined_inputs, training=True)
      loss = self.compiled_loss(y_true,
                                y_pred,
                                regularization_losses=self.losses,)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y, y_pred)

    return {m.name: m.result() for m in self.metrics}
