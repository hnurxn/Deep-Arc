'''
input: cka_dir 、threshold
output: module results divided by layer and block respectively
[Specific division] Number of modules and number of layers
'''
import tensorflow as tf
import argparse
import os
import pickle
import numpy as np
from plot import *
parser = argparse.ArgumentParser(description='Process modulary.')
parser.add_argument('--base_dir', type=str,
                    help= 'Where the trained model will be saved')
parser.add_argument('--gpu', type=int,
                    help= 'gpu limit')
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu)])


cka_dir = os.path.join(args.base_dir, 'cka_within_model_256.pkl')

best_model = tf.keras.models.load_model(args.base_dir)
layers = []
for i,l in enumerate(best_model.layers):
    if 'add' in l.name:
        layers.append(i)
print(layers)  
cka = pickle.load(tf.io.gfile.GFile(cka_dir, 'rb'))

cka1 = cka[layers]
cka1 = cka1[:,layers]
print(cka.shape,cka1.shape)

out_dir = os.path.join(args.base_dir, 'cka_within_model_256_b.pkl')
with tf.io.gfile.GFile(out_dir, 'wb') as f:
  pickle.dump(cka1, f)

plot_dir = os.path.join(args.base_dir, 'layer')
plot_ckalist_resume([cka],plot_dir)
plot_dir = os.path.join(args.base_dir, 'block')
plot_ckalist_resume([cka1],plot_dir)
