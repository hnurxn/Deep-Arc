from math import sqrt,ceil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import os
def plot_ckalist_resume(cka_list,save_name):
    n = len(cka_list)
    y = ceil(sqrt(n))
    if n == sqrt(n)*sqrt(n):
        x = y
    elif (y-1) * y < n:
        x = y
    else:
        x = y - 1
    print("x | y :",x,y)    
    fig = plt.figure(figsize=(y*4,x*4),frameon=False)

    sc = None
    for i,cka in enumerate(cka_list):
        ax = fig.add_subplot(x,y,i+1)
        ll = cka.shape[0]
        sc = ax.imshow(cka, cmap='magma', vmin=0.0,vmax=1.0)
        tick = [i for i in range(0,ll,int(ll/5))]
        ax.set_xticks(tick) 
        tick.reverse()
        ax.set_yticks([]) 
        ax.axes.invert_yaxis()
    
    l = 0.92
    b = 0.35
    w = 0.015
    h = 0.35

    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect) 

    plt.colorbar(sc,cax = cbar_ax)

    plt.savefig('{}.png'.format(save_name),dpi=700)   