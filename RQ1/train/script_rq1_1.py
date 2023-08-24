import os
import _thread
import time
import pynvml
def find_gpu(req):
    pynvml.nvmlInit()
    for i in [0,1,2,3]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)    
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(meminfo.free / 1024**2)
        if (meminfo.free / 1024**2) > req+300:
            return i
    return 5
def shell(order):
    time.sleep(3) 
    os.system(order)

divs = [10,25,50,75,100]

targets = [2,4,6,8,10]
deeps = [14,44,86,110,152]
gpus = [1024,2048,4096,5120,8192]
 
commend = []
for seed in range(8,11):
    for div in divs:
        for target in targets:
            for deep,gpu in zip(deeps,gpus):
                dir = 'cifar-depth-{}-width-1-bs-128-lr-0.010000-reg-0.005000-div-{}-targets-{}-copy-{}/weights.300.ckpt'.format(deep,div,target,seed)               


                g = find_gpu(gpu)
                while g==5:
                    print('--------------------------------------')
                    time.sleep(60)
                    g = find_gpu(gpu)
                if os.path.exists('./checkpoint/'+dir):
                    continue
                else:
            
                    commend = 'CUDA_VISIBLE_DEVICES={} python cifar_train1.py --gpu {} --base_dir checkpoint/ --div {} --targets {} --depth {} --copy {}'.format(g,gpu,div,target,deep,seed)
                    _thread.start_new_thread(shell,(commend,))
                    time.sleep(30)                 
    
