import torch
import random
import tensorflow.compat.v2 as tf
import argparse
from tensorflow.keras import backend as K
import os
import numpy as np
from scipy.special import logsumexp, softmax
from art.utils import  load_cifar10
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
tf.enable_v2_behavior()
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', type=str, default=' ')
parser.add_argument('--layer_indexes', type=int,nargs='+')

parser=parser.parse_args()
def get_all_layer_outputs_fn3(model,i,j):
  '''Builds and returns function that returns the output of every (intermediate) layer'''

  return K.function([model.layers[i+1].input],
                                  [model.layers[j].output])
def compress(best_model,layers,inter):
    acc_test = 0
    for j in range(0,len(inter)+1):
        #print('-------------','remove {} layers'.format(j))
        predictions = get_all_layer_outputs_fn3(best_model,-1,layers[0]+1)(x_test)[0]
        temp = 0
        for i in list(inter[:j]):
            if not temp ==i-1: # two adjacent layers [deleted together]
                predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,layers[i-1]+1)(predictions)[0]
            temp = i
        predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,len(best_model.layers)-1)(predictions)[0]
    acc_test = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / 10000
    return acc_test
    #print(acc_test)

if __name__ == '__main__':
    best_model = tf.keras.models.load_model(parser.experiment_dir)
    layers = torch.load('../checkpoint/resnet110_block.pt')
    '''
    input: the indexes of layers to be add or remove
    '''
  
    print(parser.layer_indexes)
    n = len(parser.layer_indexes)
    for i in range(1,n+1):
        best = 0
        ans = 0
        for j in range(5):
            iter = random.sample(parser.layer_indexes,i)
            iter.sort()
            ans = compress(best_model,layers,iter)
            if ans>best:
                best = ans
        print('After deleting {} blocks, the guaranteed accuracy is {} '.format(i,best))
    
