# coding=utf-8
from absl import app
from absl import flags

import functools
import os
import numpy as np
from numpy.random import randint

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from resnet_cifar import ResNet_CIFAR

tf.enable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train for')
flags.DEFINE_float('weight_decay', 0.005, 'L2 regularization')
flags.DEFINE_integer('depth', 56, 'No. of layers to use in the ResNet model')
flags.DEFINE_integer(
    'width_multiplier', 1,
    'How much to scale the width of the standard ResNet model by')
flags.DEFINE_integer(
    'copy', 0,
    'If the same model configuration has been run before, train another copy with a different random initialization'
)

flags.DEFINE_string('base_dir', None,
                    'Where the trained model will be saved')

flags.DEFINE_string('dataset_name', 'cifar10',
                    'Name of dataset used (CIFAR-10 of CIFAR-100)')

flags.DEFINE_boolean('use_residual', True,
                     'Whether to include residual connections in the model')

flags.DEFINE_boolean('randomize_labels', False,
                     'Whether to randomize labels during training')

flags.DEFINE_boolean(
    'partial_init', False,
    'Whether to initialize only the first few layers with pretrained weights')

flags.DEFINE_integer('epoch_save_freq', 100, 'Frequency at which ckpts are saved')
       
flags.DEFINE_integer(
    'gpu', 4096,
    'the gpu memory limit')
flags.DEFINE_integer('div',100,'division ratio')
flags.DEFINE_integer('targets',10,'category number')

def find_stack_markers(model):
  """Finds the layers where a new stack starts."""
  stack_markers = []
  old_shape = None
  for i, layer in enumerate(model.layers):
    if i == 0:
      continue
    if 'conv' in layer.name:
      conv_weights_shape = layer.get_weights()[0].shape
      if conv_weights_shape[-1] != conv_weights_shape[-2] and conv_weights_shape[
          0] != 1 and conv_weights_shape[-2] % 16 == 0:
        stack_markers.append(i)
  assert (len(stack_markers) == 2)
  return stack_markers


def random_apply(transform_fn, image, p):
  """Randomly apply with probability p a transformation to an image"""
  if tf.random.uniform([]) < p:
    return transform_fn(image)
  else:
    return image

#data augmentation
def preprocess_data(image, label, is_training):
  """CIFAR data preprocessing"""
  image = tf.image.convert_image_dtype(image, tf.float32)

  if is_training:
    crop_padding = 4
    image = tf.pad(image, [[crop_padding, crop_padding],
                           [crop_padding, crop_padding], [0, 0]], 'REFLECT')
    
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)

  else:
    image = tf.image.resize_with_crop_or_pad(image, 32, 32) 
  return image, label



def load_train_data(batch_size,
                    data_path='',
                    div = 100,
                    targets = 10,
                    dataset_name='cifar10',
                    n_data=50000,
                    randomize_labels=False,
                    as_supervised=True):
  """Load CIFAR training data"""
  if div==100:
    start = 0
  else:
    start = randint(0,100-div)
  train_dataset = tfds.load('cifar10',split='train[{}%:{}%]'.format(start,start+div),as_supervised=True)
  
  

  all_labels = []
  all_images = []
  for images, labels in train_dataset:  

    if labels.numpy() >= targets:
      continue
    else:
        all_images.append(images.numpy()[np.newaxis, :, :, :])
        all_labels.append(labels.numpy())
  all_images = np.vstack(all_images)
  all_images = all_images/255.0
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (tf.convert_to_tensor(all_images, dtype=tf.float32),
       tf.convert_to_tensor(all_labels, dtype=tf.int64)))  
  train_dataset = train_dataset.shuffle(buffer_size=n_data*div*targets//1000)

  train_dataset = train_dataset.map(
        functools.partial(preprocess_data, is_training=True))

  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset


def load_test_data(batch_size,
                   shuffle=False,
                   data_path='',
                   dataset_name='cifar10',
                   targets = 10,
                   n_data=10000,
                   as_supervised=True):
  """Load CIFAR test data"""

  test_dataset = tfds.load(
        name=dataset_name, split='test', as_supervised=as_supervised)
  all_labels = []
  all_images = []
  for images, labels in test_dataset:  

    if labels.numpy() >= targets:
      continue
    else:
        all_images.append(images.numpy()[np.newaxis, :, :, :])
        all_labels.append(labels.numpy())
  all_images = np.vstack(all_images)
  all_images = all_images/255.0
  test_dataset = tf.data.Dataset.from_tensor_slices(
      (tf.convert_to_tensor(all_images, dtype=tf.float32),
       tf.convert_to_tensor(all_labels, dtype=tf.int64)))    
  test_dataset = test_dataset.map(
          functools.partial(preprocess_data, is_training=False))

  if shuffle:
    test_dataset = test_dataset.shuffle(buffer_size=n_data)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  return test_dataset


def main(argv):
  tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.gpu)])
  n_data = 50000
  train_dataset = load_train_data(
      FLAGS.batch_size,
      dataset_name=FLAGS.dataset_name,
      n_data=n_data,
      div=FLAGS.div,
      targets=FLAGS.targets,
      randomize_labels=FLAGS.randomize_labels,
      as_supervised=not FLAGS.save_image)

  test_dataset = load_test_data(
      FLAGS.batch_size, dataset_name=FLAGS.dataset_name,targets=FLAGS.targets, n_data=10000)
  
  print(n_data)
  print(FLAGS.batch_size)
  steps_per_epoch = n_data // FLAGS.batch_size - 2
  print(steps_per_epoch)
  optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate, momentum=0.9)
  schedule = tf.keras.experimental.CosineDecay(FLAGS.learning_rate,
                                               FLAGS.epochs)
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

  print(FLAGS.use_residual)
  model = ResNet_CIFAR(
        FLAGS.depth, #depth
        FLAGS.width_multiplier, #width
        FLAGS.weight_decay, #l2-regularization
        num_classes=FLAGS.targets,# num_class
        use_residual=FLAGS.use_residual)

  model.compile(
        optimizer,
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])


  if FLAGS.copy >= 0:
    experiment_dir = '%s/cifar-depth-%d-width-%d-bs-%d-lr-%f-reg-%f-div-%d-targets-%d-copy-%d/' % \
      (FLAGS.base_dir, FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay,FLAGS.div, FLAGS.targets, FLAGS.copy)

  if FLAGS.epoch_save_freq > 0:
    tf.keras.models.save_model(
        model, experiment_dir, overwrite=True,
        include_optimizer=False)  
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=experiment_dir + 'weights.{epoch:02d}.ckpt',
        monitor='val_acc',
        verbose=1,
        save_best_only=False,
        save_freq='epoch',
        period=FLAGS.epoch_save_freq)
  else:
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=experiment_dir,
        monitor='val_acc',
        verbose=1,
        save_best_only=True)
  for l in model.layers:
      print(l.name)

  hist = model.fit(
      train_dataset,
      batch_size=FLAGS.batch_size,
      epochs=FLAGS.epochs,
      validation_data=test_dataset,
      verbose=1,
      callbacks=[checkpoint, lr_scheduler])


if __name__ == '__main__':
  app.run(main)
