import tensorflow.compat.v2 as tf

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 weight_decay=0):

  conv = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
  if batch_normalization:
    bn_layer = tf.keras.layers.BatchNormalization()
  else:
    bn_layer = lambda x: x

  x = inputs
  if conv_first:
    x = conv(x)
    x = bn_layer(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
  else:
    x = bn_layer(x)
    if activation is not None:
      x = tf.keras.layers.Activation(activation)(x)
    x = conv(x)
  return x

class CustomModel(tf.keras.Model):

  def __init__(self, inputs, outputs):
    super(CustomModel, self).__init__(inputs=inputs, outputs=outputs)
    self.all_ids = []

  def train_step(self, data):

    id, x, y = data
    self.all_ids.append(id.numpy())
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)

      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}


def ResNet_CIFAR(depth,
                 width_multiplier,
                 weight_decay,
                 num_classes,
                 input_shape=(32, 32, 3),
                 use_residual=True):

  num_filters = int(round(16 * width_multiplier))
  num_res_blocks = int((depth - 2) / 6)
  inputs = tf.keras.Input(shape=input_shape)
  x = resnet_layer(
      inputs=inputs,
      num_filters=num_filters,
      weight_decay=weight_decay)

  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1

      if stack > 0 and res_block == 0:  
        strides = 2  

      y = resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides,
          weight_decay=weight_decay)
      y = resnet_layer(
          inputs=y,
          num_filters=num_filters,
          activation=None,
          weight_decay=weight_decay)

      if stack > 0 and res_block == 0:
        x = resnet_layer(
            inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            activation=None,
            batch_normalization=False,
            weight_decay=weight_decay)
      if use_residual:
        x = tf.keras.layers.add([x, y])
      else:
        x = y

      x = tf.keras.layers.Activation('relu')(x)

    num_filters *= 2

  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  y = tf.keras.layers.Flatten()(x)

  outputs = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(
          y)


  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model
