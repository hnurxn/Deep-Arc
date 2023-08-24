# coding=utf-8
import functools
from itertools import combinations
import json
import os
import pickle
import random
import re

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from cifar_train1 import load_test_data, preprocess_data
from efficient_CKA import *

tf.enable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

FLAGS = flags.FLAGS
flags.DEFINE_integer('cka_batch', 256, 'Batch size used to approximate CKA')
flags.DEFINE_integer('cka_iter', 10,
                     'Number of iterations to run minibatch CKA approximation')

flags.DEFINE_string('experiment_dir', None,
                    'Path to where the trained model is saved')



def normalize_activations(act):
  act = act.reshape(act.shape[0], -1)
  act_norm = np.linalg.norm(act, axis=1)
  act /= act_norm[:, None]
  return act


def get_activations(images, model, normalize_act=False):
  input_layer = model.input
  layer_outputs = [layer.output for layer in model.layers]
  get_layer_outputs = K.function(input_layer, layer_outputs)
  activations = get_layer_outputs(images)
  if normalize_act:
    activations = [normalize_activations(act) for act in activations]
  return activations


def convert_bn_to_train_mode(model):
  bn_layers = [
      i for i, layer in enumerate(model.layers)
      if 'batch_normalization' in layer.name
  ]
  model_config = model.get_config()
  for i in bn_layers:
    model_config['layers'][i]['inbound_nodes'][0][0][-1]['training'] = True
  new_model = model.from_config(model_config)
  for i, layer in enumerate(new_model.layers):
    layer.set_weights(model.layers[i].get_weights())
  return new_model


def compute_cka_internal(model_dir,
                         data_path=None,
                         dataset_name='cifar10',
                         use_batch=True,
                         use_train_mode=False,
                         normalize_act=False):
  if dataset_name == 'cifar10':
    if use_train_mode:
      filename = 'cka_within_model_%d_bn_train_mode.pkl' % FLAGS.cka_batch
    else:
      filename = 'cka_within_model_%d.pkl' % FLAGS.cka_batch
  else:
    suffix = dataset_name.split('_')[-1]
    if use_train_mode:
      filename = 'cka_within_model_%d_%s_bn_train_mode.pkl' % (FLAGS.cka_batch,
                                                               suffix)
    else:
      filename = 'cka_within_model_%d_%s.pkl' % (FLAGS.cka_batch, suffix)
  if normalize_act:
    filename = filename.replace('.pkl', '_normalize_activations.pkl')
  print('------------',model_dir)
  out_dir = os.path.join(model_dir, filename)
  if tf.io.gfile.exists(out_dir):
    return

  model = tf.keras.models.load_model(model_dir)
  if use_train_mode:
    model = convert_bn_to_train_mode(model)

  n_layers = len(model.layers)
  cka = MinibatchCKA(n_layers)
  if use_batch:
    for _ in range(FLAGS.cka_iter):
      dataset = load_test_data(
          FLAGS.cka_batch,
          shuffle=True,
          dataset_name=dataset_name,
          targets=FLAGS.targets,
          n_data=10000)
      for images, _ in dataset:
        cka.update_state(get_activations(images, model, normalize_act))
  else:
    dataset = load_test_data(
        FLAGS.cka_batch, data_path=data_path, dataset_name=dataset_name)
    all_images = tf.concat([x[0] for x in dataset], 0)
    cka.update_state(get_activations(all_images, model))
  heatmap = cka.result().numpy()
  logging.info(out_dir)
  with tf.io.gfile.GFile(out_dir, 'wb') as f:
    pickle.dump(heatmap, f)






def main(argv):
  tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.gpu)])

  compute_cka_internal(FLAGS.experiment_dir)
if __name__ == '__main__':
  app.run(main)
