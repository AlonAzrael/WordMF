

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import numpy as np
from numpy import random
import time as t

from jinja2 import Template


def gen_glove_cuda_kernel():
    
    learning_rate = 0.05
    max_loss = 10.0
    
    glove_cuda_kernel_template = """
    #include <stdio.h>

    __global__ void glove_kernel(
        // data part
        int* shuffle_indices,
        int* rows, 
        int* cols, 
        float* datas, 
        int datas_size
        // nn part
        /*
        float* word_vectors, 
        float* wordvec_sum_gradients, 
        float* word_biases, 
        float* wordbias_sum_gradients, 
        */
        ){

    // float learning_rate = {{learning_rate}};
    // float max_loss = {{max_loss}};

    // distribute index
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k < datas_size) {
        int shuffle_index = shuffle_indices[k];
        int word_a = rows[shuffle_index];
        int word_b = cols[shuffle_index];
        float val = datas[shuffle_index];

        printf("I am %dth thread in threadIdx.x:%d.threadIdx.y:%d  blockIdx.:%d blockIdx.y:%d blockDim.x:%d blockDim.y:%d\\n",(threadIdx.x+threadIdx.y*blockDim.x+(blockIdx.x*blockDim.x*blockDim.y)+(blockIdx.y*blockDim.x*blockDim.y)),threadIdx.x, threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
        printf("val:%f", val);
    } 
    else {
        return;
    }
    
    }
    """
    glove_cuda_kernel_s = Template(glove_cuda_kernel_template).render(learning_rate=learning_rate, max_loss=max_loss)

    return glove_cuda_kernel_s


def run_glove_gpu():
    glove_cuda_kernel_s = gen_glove_cuda_kernel()
    
    mod = SourceModule(glove_cuda_kernel_s)
    glove_kernel = mod.get_function("glove_kernel")

    datas_size = 100
    shuffle_indices = np.arange(datas_size, dtype=np.int32)
    rows = np.arange(datas_size, dtype=np.int32)
    cols = np.arange(datas_size, dtype=np.int32)
    # datas = random.random(datas_size, dtype=np.float32)
    datas = np.arange(datas_size, dtype=np.float32)
    
    gshuffle_indices = gpuarray.to_gpu(shuffle_indices)
    grows = gpuarray.to_gpu(rows)
    gcols = gpuarray.to_gpu(cols)
    gdatas = gpuarray.to_gpu(datas)

    glove_kernel(gshuffle_indices, grows, gcols, gdatas, np.int32(datas_size), block=(4,4,1), grid=(2,2,1))





def test_cuda_run():
    
    a = np.random.randn(4,4)
    a = a.astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)


def random_numbers():
    x = np.random.random(int(10e+6)).astype(np.float32)
    start = t.time()
    valid = np.logical_and(-1 < x , x < +1)
    print "Found values", np.sum(valid), "elapsed time:", t.time() - start


if __name__ == '__main__':
    # test_cuda_run()
    # random_numbers()

    run_glove_gpu()

