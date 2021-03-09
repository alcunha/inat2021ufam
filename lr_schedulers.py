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

import tensorflow as tf

def lr_cosine_decay(initial_learning_rate,
                    current_step,
                    decay_steps,
                    alpha=0.0):

  if current_step > decay_steps:
    current_step = decay_steps

  cosine_decay = 0.5 * (1 + tf.math.cos(
                                  math.pi * current_step / float(decay_steps)))
  decayed = (1 - alpha) * cosine_decay + alpha

  return initial_learning_rate * decayed

def lr_linear_warmup(initial_learning_rate, current_step, warmup_steps):

  return current_step * initial_learning_rate / float(warmup_steps)

class CosineDecayWithLinearWarmUpScheduler(tf.keras.callbacks.Callback):

  def __init__(self,
               initial_learning_rate,
               decay_steps,
               warmup_steps=0,
               alpha=0.0):

    super(CosineDecayWithLinearWarmUpScheduler, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.warmup_steps = warmup_steps
    self.alpha = alpha
    self.steps = 0
    self.learning_rates = []

  def on_train_batch_begin(self, batch, logs=None):
    if not hasattr(self.model.optimizer, "lr"):
      raise ValueError('Optimizer must have a "lr" attribute.')

    self.steps = self.steps + 1

    if self.steps < self.warmup_steps:
      lr = lr_linear_warmup(
              self.initial_learning_rate,
              self.steps,
              self.warmup_steps)
    elif self.initial_learning_rate == self.alpha:
      lr = self.initial_learning_rate
    else:
      lr = lr_cosine_decay(
              self.initial_learning_rate,
              self.steps - self.warmup_steps,
              self.decay_steps,
              self.alpha)

    tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    self.learning_rates.append(lr)
