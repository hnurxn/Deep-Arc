import torch
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
parser.add_argument('--opt', type=str, default='removal')#removal or addition
parser.add_argument('--experiment_dir', type=str, default=' ')
parser.add_argument('--layer_indexes', type=int,nargs='+')

parser=parser.parse_args()
def get_all_layer_outputs_fn3(model,i,j):
  '''Builds and returns function that returns the output of every (intermediate) layer'''

  return K.function([model.layers[i+1].input],
                                  [model.layers[j].output])
def removal_or_add(best_model,layers,inter,opt):
    acc = []
    if opt == 'removal':
        for j in range(0,len(inter)+1):
            print('-------------','remove {} layers'.format(j))
            predictions = get_all_layer_outputs_fn3(best_model,-1,layers[0]+1)(x_test)[0]
            temp = 0
            #for i in list(reversed(inter[:j])):
            for i in list(inter[:j]):
                if not temp ==i-1: # two adjacent layers [deleted together]
                    predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,layers[i-1]+1)(predictions)[0]
                #predictions = get_all_layer_outputs_fn3(best_model,layers[i]+1,layers[i]+1)(predictions)[0]
                temp = i
            predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,len(best_model.layers)-1)(predictions)[0]
            acc_test = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / 10000
            print(acc_test)
            acc.append(acc_test) 
    else:
        for j in range(0,len(inter)+1):
            print('-------------','add {} layers'.format(j))
            predictions = get_all_layer_outputs_fn3(best_model,-1,layers[0]+1)(x_test)[0]
            temp = 0
            #for i in list(reversed(inter[:j])):
            for i in list(inter[:j]):
                predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,layers[i+1]+1)(predictions)[0]
                #predictions = get_all_layer_outputs_fn3(best_model,layers[i]+1,layers[i]+1)(predictions)[0]
                temp = i
            predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,len(best_model.layers)-1)(predictions)[0]
            acc_test = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / 10000
            print(acc_test)
            acc.append(acc_test)         
    print(acc)  
if __name__ == '__main__':
    best_model = tf.keras.models.load_model(parser.experiment_dir)
    layers = torch.load('../checkpoint/resnet110_block.pt')
    '''
    input: the indexes of layers to be add or remove
    '''
    
    print(parser.layer_indexes)

    #inter = [49,48,47,46]
    inter = parser.layer_indexes
    removal_or_add(best_model,layers,inter,parser.opt)
    # for j in range(0,len(inter)+1):
    #     print('-------------','remove {} layers'.format(j))
    #     predictions = get_all_layer_outputs_fn3(best_model,-1,layers[0]+1)(x_test)[0]
    #     temp = 0
    #     #for i in list(reversed(inter[:j])):
    #     for i in list(inter[:j]):
    #         if not temp ==i-1:
    #             predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,layers[i-1]+1)(predictions)[0]
    #         #predictions = get_all_layer_outputs_fn3(best_model,layers[i]+1,layers[i]+1)(predictions)[0]
    #         temp = i
    #     predictions = get_all_layer_outputs_fn3(best_model,layers[temp]+1,len(best_model.layers)-1)(predictions)[0]
    #     acc_test = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / 10000
    #     print(acc_test)
    #     acc.append(acc_test)

