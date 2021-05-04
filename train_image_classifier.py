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
import collections

import tensorflow as tf

import lr_schedulers

HParams = collections.namedtuple("HParams", [
    'lr', 'use_cosine_decay', 'warmup_steps', 'epochs', 'batch_size',
    'momentum', 'label_smoothing', 'use_logits', 'model_dir'
  ])

def get_default_hparams():
  return HParams(
    lr=0.01,
    use_cosine_decay=False,
    warmup_steps=500,
    epochs=10,
    batch_size=32,
    momentum=0.0,
    label_smoothing=0.0,
    use_logits=False,
    model_dir='/tmp/models/'
  )

def generate_optimizer(hparams):
  optimizer = tf.keras.optimizers.SGD(lr=hparams.lr, momentum=hparams.momentum)

  return optimizer

def generate_loss_fn(hparams):
  loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=hparams.use_logits,
    label_smoothing=hparams.label_smoothing
    )

  return loss_fn

def generate_lr_scheduler(hparams, steps_per_epoch):
  if hparams.use_cosine_decay:
    alpha = 0.0
  else:
    alpha = hparams.lr

  scheduler = lr_schedulers.CosineDecayWithLinearWarmUpScheduler(
    initial_learning_rate=hparams.lr,
    decay_steps=hparams.epochs*steps_per_epoch,
    warmup_steps=hparams.warmup_steps,
    alpha=alpha
  )

  return scheduler

def train_model(model,
                hparams,
                train_data_and_size,
                val_data_and_size,
                strategy):

  train_data, train_size = train_data_and_size
  val_data, val_size = val_data_and_size

  steps_per_epoch = train_size // hparams.batch_size
  validation_steps = val_size // hparams.batch_size

  summary_dir = os.path.join(hparams.model_dir, "summaries")
  summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)

  checkpoint_filepath = os.path.join(hparams.model_dir, "ckp")
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      save_freq='epoch')

  callbacks = [summary_callback, checkpoint_callback]

  if hparams.use_cosine_decay or hparams.warmup_steps > 0:
    callbacks.append(generate_lr_scheduler(hparams, steps_per_epoch))

  with strategy.scope():
    optimizer = generate_optimizer(hparams)
    loss_fn = generate_loss_fn(hparams)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

  return model.fit(
    train_data,
    epochs=hparams.epochs,
    callbacks=callbacks,
    validation_data=val_data,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
  )
