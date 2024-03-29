
# Here's benchmark for numpy.

"""
Sections
1. Matmul (2D * 2D)
2. Broadcasting
3. Slicing Tensor
4. Complicated Exps (denselayer)
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from statistics import mean
import json

print(np.show_config())

# Parameters
BACKEND_NAME = "MKL"
N = 200

BATCH_SIZE = 10000

MATMUL_SIZE = [16, 32, 64, 128, 256, 512, 1024, 2048]
BROADCASTING_SHAPE = [[[10, 10, 1], [1, 10, 10]],
                      [[100, 100, 1], [1, 100, 100]],
                      [[200, 200, 1], [1, 200, 200]],
                      [[300, 300, 1], [1, 300, 300]]]
                      
SLICE_SIZE = [512, 1024, 2048, 4096, 8192] 
NN_SIZE   =  [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

RESULT_DIR = './benchmark/results/numpy_result.json'
RESULTS = []

# Files
MATMUL_RESULT_DIR = './benchmark/results/matmul_numpy.png'
BROADCASTING_RESULT_DIR = './benchmark/results/broadcasting_numpy.png'
SLICING_RESULT_DIR = './benchmark/results/slicing_numpy.png'
NN_RESULT_DIR = './benchmark/results/dense_numpy.png'

# Counters
matmul_try_n = 1
broadcasting_try_n = 1
slicing_try_n = 1
nn_try_n = 1

class DenseLayer():
    def __init__(self, in_features, out_features):
        self.weight = 0.01 * np.random.randn(in_features, out_features).astype('float32')
        self.bias   = np.zeros([1, out_features]).astype('float32')
    def forward(self, x):
        return np.maximum(0, np.matmul(x, self.weight) + self.bias)

    
def matmul_2D(K=1000):
    global matmul_try_n
    print(f"[{matmul_try_n}/{len(MATMUL_SIZE)}] Testing on {K}*{K} Matrix for {N} times...")
    matmul_try_n += 1
    x = np.random.randn(K, K).astype('float32')
    t1 = time()
    for i in range(N):
        np.matmul(x, x)
    t2 = time()
    return (t2 - t1) / N

def broadcasting_2D(K):
    global broadcasting_try_n
    print(f"[{broadcasting_try_n}/{len(BROADCASTING_SHAPE)}] Testing on Matrix.size()[1]={K[0][1]} for {N} times...")
    broadcasting_try_n += 1
    a = np.random.randn(K[0][0], K[0][1], K[0][2]).astype('float32')
    b = np.random.randn(K[1][0], K[1][1], K[1][2]).astype('float32')

    t1 = time()
    for i in range(N):
        np.add(a, b)
    t2 = time()
    return (t2 - t1) / N

def nn_bench(K):
    global nn_try_n
    print(f"[{nn_try_n}/{len(NN_SIZE)}] Testing on {BATCH_SIZE}*{K} Matrix for {N} times...")
    nn_try_n += 1
    x = np.random.randn(BATCH_SIZE, K).astype('float32')
    model = DenseLayer(K, 10)

    t1 = time()
    for i in range(N):
        model.forward(x)
    t2 = time()
    return (t2 - t1) / N
    
def slicing_bench(K):
    global slicing_try_n
    print(f"[{slicing_try_n}/{len(SLICE_SIZE)}] Testing on {K}*{K} Matrix for {N} times...")
    slicing_try_n += 1
    a = np.random.randn(K, K).astype('float32')

    t1 = time()
    for i in range(N):
        _ = a[200:400, :].copy()
    t2 = time()
    return (t2 - t1) / N

if __name__ == "__main__":
    
    print("ℹ️ Running matmul_2D...")
    print("")
    matmul_result = []
    for case in MATMUL_SIZE:
        result = matmul_2D(K=case)
        matmul_result.append(result)
    plt.plot(MATMUL_SIZE, matmul_result)
    plt.title(f"matmul (numpy + {BACKEND_NAME}) (N={N})")
    plt.xlabel("Matrix Size")
    plt.ylabel("time (second)")
    plt.savefig(MATMUL_RESULT_DIR)
    print(f"⭕️ The result is correctly saved at {MATMUL_RESULT_DIR}")



    print("Running broadcasting...")
    print("")
    plt.cla()
    broadcasting_result = []
    for case in BROADCASTING_SHAPE:
        result = broadcasting_2D(case)
        broadcasting_result.append(result)
    plt.plot([K[0][1] for K in BROADCASTING_SHAPE], broadcasting_result)
    plt.title(f"broadcasting (numpy + {BACKEND_NAME}) (N={N})")
    plt.xlabel("Matrix Size")
    plt.ylabel("time (second)")
    plt.savefig(BROADCASTING_RESULT_DIR)
    # Todo Output to CSV and merge graphs
    print(f"⭕️ The result is correctly saved at {BROADCASTING_RESULT_DIR}")


    
    print("Running slicing...")
    print("")
    plt.cla()
    slicing_result = []
    for case in SLICE_SIZE:
        result = slicing_bench(case)
        slicing_result.append(result)
    plt.plot(SLICE_SIZE, slicing_result)
    plt.title(f"slicing (numpy + {BACKEND_NAME}) (N={N})")
    plt.xlabel("Matrix Size")
    plt.ylabel("time (second)")
    plt.savefig(SLICING_RESULT_DIR)
    print(f"⭕️ The result is correctly saved at {SLICING_RESULT_DIR}")
    

    print("Running Dense...")
    print("")
    plt.cla()
    dense_result = []
    for case in NN_SIZE:
        result = nn_bench(case)
        dense_result.append(result)
    plt.plot(NN_SIZE, dense_result)
    plt.title(f"DenseLayer(ReLU) (numpy + {BACKEND_NAME}) (N={N}, BATCH_SIZE={BATCH_SIZE})")
    plt.xlabel("Matrix Size")
    plt.ylabel("time (second)")
    plt.savefig(NN_RESULT_DIR)
    print(f"⭕️ The result is correctly saved at {NN_RESULT_DIR}")

    
    RESULTS = [{"backend":f"numpy version: {np.__version__}, works on {BACKEND_NAME}"},
               {"dtype":"single-float(float32)"},
               {"matmul":{"desc":f"{BACKEND_NAME}, N={N}",
                          "x-seq":MATMUL_SIZE,
                          "y-seq":matmul_result}},
               {"broadcasting":{"desc":f"{BACKEND_NAME}, N={N}",
                                "x-seq":[K[0][1] for K in BROADCASTING_SHAPE],
                                "y-seq":broadcasting_result}},
               {"slice":{"desc":f"{BACKEND_NAME}, N={N}",
                          "x-seq":SLICE_SIZE,
                          "y-seq":slicing_result}},
               {"DenseLayer":{"desc":f"{BACKEND_NAME}, N={N}",
                              "x-seq":NN_SIZE,
                              "y-seq":dense_result}}]

    with open(RESULT_DIR, mode='w') as f:
        f.write(json.dumps(RESULTS))
    
