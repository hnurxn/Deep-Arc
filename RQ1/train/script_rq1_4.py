import os
import _thread
import time
def shell(order):
    time.sleep(2) 
    os.system(order)

divs = [10,25,50,75,100]

targets = [2,4,6,8,10]
deeps = [14,44,86,110,152]
gpus = [1024,2048,4096,5120,8192]
threshold = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
commend = []
for threshold in threshold[::-1]:
    for seed in range(8,11):
        for div in divs:
            for target in targets:
                for deep,gpu in zip(deeps,gpus):

                    dir = './checkpoint/cifar-depth-{}-width-1-bs-128-lr-0.010000-reg-0.005000-div-{}-targets-{}-copy-{}/weights.300.ckpt/'.format(deep,div,target,seed)               
                    des_dir = dir + '{}_modules.pkl'.format(threshold)

                    commend = 'python modularity.py --base_dir {} --threshold {}'.format(dir,threshold)

                    if os.path.exists(des_dir):
                        continue
                    else:
                        commend = 'python modularity.py --base_dir {} --threshold {}'.format(dir,threshold)
                        _thread.start_new_thread(shell,(commend,))
                        time.sleep(2)                 
  
