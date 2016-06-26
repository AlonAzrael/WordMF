

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import numpy as np
from numpy import random
import time as t
from time import time

from jinja2 import Template


def gen_glove_cuda_kernel(block_dim, grid_dim, n_features=10, learning_rate=0.05, max_loss=10, k_loss=2, gpu_thread_batch_size=1):
    
    glove_cuda_kernel_template = """
    #include <stdio.h>
    #include <math.h>

    #define BLOCK_SIZE {{block_size}}
    #define GRID_DIMX {{grid_dimx}}
    #define BLOCK_DIMX {{block_dimx}}
    #define N_FEATURES {{n_features}}
    #define K_LOSS {{k_loss}}
    #define GPU_THREAD_BATCH_SIZE {{gpu_thread_batch_size}}

    __global__ void glove_kernel(
        // data part
        int* rows, 
        int* cols, 
        float* datas, int datas_size,
        int* shuffle_indices,
        // nn part
        /*
        float* word_biases, float* wordbias_sum_gradients, 
        */
        float* word_vectors, float* wordvec_sum_gradients
        ){

    float init_learning_rate = {{learning_rate}};
    float max_loss = {{max_loss}};

    // distribute index
    // int k = threadIdx.x + blockIdx.x * blockDim.x;
    int start_i = blockIdx.y*GRID_DIMX*BLOCK_SIZE+blockIdx.x*BLOCK_SIZE+threadIdx.y*BLOCK_DIMX+threadIdx.x;
    start_i *= GPU_THREAD_BATCH_SIZE;

    if (start_i < datas_size) {

        int stop_i = start_i + GPU_THREAD_BATCH_SIZE;
        if (stop_i>datas_size) 
            stop_i = datas_size;

        for (int k=start_i;k<stop_i; k++) {

            // printf("k:%d\\n", k);
            // printf("shuffle_index:%d\\n", shuffle_index);

            int shuffle_index = shuffle_indices[k];
            int word_a = rows[shuffle_index];
            int word_b = cols[shuffle_index];
            float val = datas[shuffle_index];
            
            float prediction = 0.0;
            float loss = 0.0;
            float learning_rate = init_learning_rate;
            float gradient = 0.0;
            
            // === start one update ===
            int ia = word_a*N_FEATURES;
            int ib = word_b*N_FEATURES;

            for (int i=0;i<N_FEATURES;i++) {
                prediction = prediction + word_vectors[ia+i] * word_vectors[ib+i];
            }

            loss = (prediction - val) * K_LOSS;
            if (loss < -max_loss) 
                loss = -max_loss;
            else if (loss > max_loss)
                loss = max_loss;

            for (int i=0;i<N_FEATURES;i++) {
                
                // update word_a
                learning_rate = init_learning_rate / sqrt(wordvec_sum_gradients[ia+i]);
                gradient = loss * word_vectors[ib+i];
                word_vectors[ia+i] = (word_vectors[ia+i] - learning_rate*gradient);
                wordvec_sum_gradients[ia+i] += gradient*gradient;

                // update word_b
                learning_rate = init_learning_rate / sqrt(wordvec_sum_gradients[ib+i]);
                gradient = loss * word_vectors[ia+i];
                word_vectors[ib+i] = (word_vectors[ib+i] - learning_rate*gradient);
                wordvec_sum_gradients[ib+i] += gradient*gradient;

            }
            // === end one update ===

        }

    } 
    else {
        return;
    }
    
    }
    """
    glove_cuda_kernel_s = Template(glove_cuda_kernel_template).render(
        
        learning_rate=learning_rate, 
        max_loss=max_loss, 
        n_features=n_features,
        k_loss=k_loss,
        gpu_thread_batch_size=gpu_thread_batch_size,

        block_dimx=block_dim[0],
        block_dimy=block_dim[1],
        block_size=block_dim[0]*block_dim[1],
        grid_dimx=grid_dim[0],
        grid_dimy=grid_dim[1],
    )

    return glove_cuda_kernel_s


# main API
def fit_vectors(
        word_vectors, 
        wordvec_sum_gradients, 
        word_biases, 
        wordbias_sum_gradients, 
        rows, 
        cols, 
        datas, 
        shuffle_indices, 
        n_epochs=10, # must have this
        learning_rate=0.05, 
        max_count=100,
        alpha=0.75,
        max_loss=10,
        no_threads=4, # useless
        iter_counter=2,
        k_loss=2,
    ):

    datas_size = len(datas)
    n_features = word_vectors.shape[1]

    # gpu init

    # block and grid configuration should be tuned by hardware
    block_dim = [32,32]
    grid_dim = [2,2]
    gpu_batch_size = block_dim[0]*block_dim[1] * grid_dim[0]*grid_dim[1]

    # datas_size should be really large
    if datas_size > gpu_batch_size:
        gpu_thread_batch_size = 1.0*datas_size/gpu_batch_size
        if "." in str(gpu_thread_batch_size):
            gpu_thread_batch_size = int(gpu_thread_batch_size)+1
        else:
            gpu_thread_batch_size = int(gpu_thread_batch_size)
    else:
        gpu_thread_batch_size = 1

    glove_cuda_kernel_s = gen_glove_cuda_kernel(
        block_dim, 
        grid_dim, 
        n_features=n_features, 
        learning_rate=learning_rate,
        max_loss=max_loss,
        k_loss=k_loss,
        gpu_thread_batch_size=gpu_thread_batch_size)
    print "cuda_kernel_source:"
    print glove_cuda_kernel_s
    
    mod = SourceModule(glove_cuda_kernel_s)
    glove_kernel = mod.get_function("glove_kernel")
    
    # params to gpu

    grows = gpuarray.to_gpu(rows)
    gcols = gpuarray.to_gpu(cols)
    gdatas = gpuarray.to_gpu(datas)

    gword_vectors = gpuarray.to_gpu(word_vectors)
    gwordvec_sum_gradients = gpuarray.to_gpu(wordvec_sum_gradients)
    
    gshuffle_indices = gpuarray.to_gpu(shuffle_indices)

    start_time = time()

    for i in xrange(n_epochs):
        shuffle_indices = random.permutation(datas_size).astype("int32")
        gshuffle_indices.set(shuffle_indices)

        glove_kernel(
            grows, gcols, gdatas, np.int32(datas_size),
            gshuffle_indices,
            gword_vectors, gwordvec_sum_gradients,
            block=tuple(block_dim+[1]), grid=tuple(grid_dim+[1]),
        )

    word_vectors_out = gword_vectors.get()

    end_time = time()
    print "run_glove_gpu elapsed time:", end_time - start_time

    return word_vectors_out, word_biases


def run_glove_gpu(n_epochs=50, n_features=10, datas_size=10000, n_words=10000, check_error=False):
    # for demo usage
    
    # data init
    
    rows = np.arange(datas_size, dtype=np.int32)
    cols = np.arange(datas_size, dtype=np.int32)
    # datas = random.random(datas_size, dtype=np.float32)
    datas = np.arange(datas_size, dtype=np.float32)

    # model init
    
    word_vectors = (random.random((n_words, n_features), ).astype('float32') - 0.5) / n_features
    wordvec_sum_gradients = np.ones_like(word_vectors)
    
    word_biases = np.zeros(n_words, dtype=np.float32)
    wordbias_sum_gradients = np.ones_like(word_biases)

    # glove_kernel.prepare("P")
    # glove_kernel.prepared_call(
    #     tuple(grid_dim+[1]), tuple(block_dim+[1]),
    #     grows, gcols, gdatas, np.int32(datas_size),
    #     gshuffle_indices,
    #     gword_vectors, gwordvec_sum_gradients, 
    # )

    shuffle_indices = random.permutation(datas_size).astype("int32")

    word_vectors_out, word_biases = fit_vectors(word_vectors, wordvec_sum_gradients, word_biases, wordbias_sum_gradients, rows, cols, datas, shuffle_indices, n_epochs=n_epochs)
    
    if check_error:
        error = 0.0
        for i,ri in enumerate(rows):
            ci = cols[i]
            x = datas[i]

            error += abs(np.dot(word_vectors_out[ri],word_vectors_out[ci]) - x)

        print "error:", error / datas_size



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

    run_glove_gpu(check_error=True)

