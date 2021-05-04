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

import collections

import tensorflow as tf

from utils import is_number

ModelSpecs = collections.namedtuple("ModelSpecs", [
    'name', 'func', 'input_size', 'classes', 'activation'
  ])

def get_default_specs():
  return ModelSpecs(
    name='efficientnet-b0',
    func=tf.keras.applications.EfficientNetB0,
    input_size=224,
    classes=2,
    activation='softmax'
  )

efficientnet_b0_spec = get_default_specs()._replace(
  name='efficientnet-b0',
  func=tf.keras.applications.EfficientNetB0,
  input_size=224
)

efficientnet_b2_spec = get_default_specs()._replace(
  name='efficientnet-b2',
  func=tf.keras.applications.EfficientNetB2,
  input_size=260
)

efficientnet_b3_spec = get_default_specs()._replace(
  name='efficientnet-b3',
  func=tf.keras.applications.EfficientNetB3,
  input_size=300
)

efficientnet_b4_spec = get_default_specs()._replace(
  name='efficientnet-b4',
  func=tf.keras.applications.EfficientNetB4,
  input_size=380
)

mobilenetv2_spec = get_default_specs()._replace(
  name='mobilenetv2',
  func=tf.keras.applications.MobileNetV2,
  input_size=224
)

MODELS_SPECS = {
  'efficientnet-b0': efficientnet_b0_spec,
  'efficientnet-b2': efficientnet_b2_spec,
  'efficientnet-b3': efficientnet_b3_spec,
  'efficientnet-b4': efficientnet_b4_spec,
  'mobilenetv2': mobilenetv2_spec,
}

def _get_mobilenet_params(model_name):
  alpha = 1.0
  params_list = model_name.split('_')

  if len(params_list) > 1:
    for param in params_list[1:]:
      if param.startswith('a') and is_number(param[1:]):
        alpha = float(param[1:])

  return alpha

def _get_keras_base_model(specs, model_name, weights='imagenet'):
  base_model = None

  if specs.name == 'mobilenetv2':
    alpha = _get_mobilenet_params(model_name)
    base_model = specs.func(
      input_shape=(specs.input_size, specs.input_size, 3),
      alpha=alpha,
      include_top=False,
      weights=weights
    )
  elif specs.name.startswith('efficientnet'):
    base_model = specs.func(
      input_shape=(specs.input_size, specs.input_size, 3),
      include_top=False,
      weights=weights
    )
  else:
    raise RuntimeError('Model %s not implemented' % specs.name)

  return base_model

def _get_coordinates_base_model(seed=None):
  coordinates_model = tf.keras.Sequential([
    tf.keras.layers.Dense(4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
  ])

  return coordinates_model

def _create_model_from_specs(specs, model_name, use_coordinates_inputs,
                             unfreeze_layers=0, return_base_model=False,
                             base_model_weights='imagenet', seed=None):
  training = unfreeze_layers == -1

  image_input = tf.keras.Input(shape=(specs.input_size, specs.input_size, 3))
  base_model = _get_keras_base_model(specs, model_name, base_model_weights)
  base_model.trainable = training
  if unfreeze_layers > 0:
    for layer in base_model.layers[-unfreeze_layers:]:
      layer.trainable = True

  x = base_model(image_input, training=training)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)

  if use_coordinates_inputs:
    coordinates_input = tf.keras.Input(shape=(2,))
    coordinates_model = _get_coordinates_base_model(seed)
    coordinates_model.trainable = training
    y = coordinates_model(coordinates_input, training=training)
    x = tf.keras.layers.concatenate([x, y])
    inputs = [image_input, coordinates_input]
  else:
    inputs = [image_input]

  outputs = tf.keras.layers.Dense(
      specs.classes,
      activation=specs.activation,
      kernel_initializer=tf.keras.initializers.glorot_uniform(seed))(x)
  model = tf.keras.models.Model(inputs=inputs, outputs=[outputs])

  if return_base_model:
    return model, base_model

  return model

def create(model_name,
           num_classes,
           input_size=None,
           classifier_activation="softmax",
           unfreeze_layers=-1,
           use_coordinates_inputs=False,
           return_base_model=False,
           base_model_weights='imagenet',
           seed=None):

  model_name_base = model_name.split('_')[0]

  if model_name_base not in MODELS_SPECS.keys():
    raise RuntimeError('Model %s not implemented' % model_name_base)

  specs = MODELS_SPECS[model_name_base]
  specs = specs._replace(
    classes=num_classes,
    activation=classifier_activation,
  )
  if input_size is not None:
    specs = specs._replace(input_size=input_size)

  return _create_model_from_specs(specs, model_name, use_coordinates_inputs,
                                  unfreeze_layers, return_base_model,
                                  base_model_weights, seed)
