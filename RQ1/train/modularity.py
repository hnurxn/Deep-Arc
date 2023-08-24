'''
input: cka_dir 、threshold
output: Module results divided by layer and block respectively
[Specific division] Number of modules and number of layers
'''
import tensorflow as tf
import argparse
import string
import os
import pickle
import numpy as np
from plot import *
parser = argparse.ArgumentParser(description='Process modulary.')
parser.add_argument('--base_dir', type=str,
                    help= 'Where the trained model will be saved')
parser.add_argument('--threshold', type=float,
                    help= 'the threshold of modularity')
args = parser.parse_args()

cka_dir = os.path.join(args.base_dir, 'cka_within_model_256.pkl')
cka_dir1 = os.path.join(args.base_dir, 'cka_within_model_256_b.pkl')

 
cka = pickle.load(tf.io.gfile.GFile(cka_dir, 'rb'))
cka1 = pickle.load(tf.io.gfile.GFile(cka_dir1, 'rb'))


mask1 = []
start = 0
for i in range(1,len(cka)):
    end = i
    if cka[end-1,end] > args.threshold:
        continue
    else:
        mask1.append([start,end-1])
        start = i
mask1.append([start,len(cka)])

mask2 = []
start = 0
for i in range(1,len(cka1)):
    end = i
    if cka1[end-1,end] > args.threshold:
        continue
    else:
        mask2.append([start,end-1])
        start = i
mask2.append([start,len(cka1)])

# save the modular result
out_dir = os.path.join(args.base_dir, '{}_modules.pkl'.format(args.threshold))
# The division result of the layer and the total number of layers The division result of the block and the total number of blocks
ans = [mask1,len(cka),mask2,len(cka1)]
with tf.io.gfile.GFile(out_dir, 'wb') as f:
        pickle.dump(ans, f)
