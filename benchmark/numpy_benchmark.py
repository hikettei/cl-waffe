
# Here's benchmark for numpy.

"""
Sections
1. Matmul (2D * 2D)
2. Broadcasting
3. Slicing Tensor
4. Complicated Exps (e.g.: sigmoid)
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from statistics import mean

print(np.show_config())

# Parameters
BACKEND_NAME = "MKL"
N = 100
MATMUL_SIZE = [16, 32, 64, 256, 512, 1024, 2048]#, 4096, 8192]


# Files
MATMUL_RESULT_DIR = './benchmark/results/matmul_numpy.png'

# Counters
matmul_try_n = 1

def matmul_2D(K=1000):
    global matmul_try_n
    print(f"[{matmul_try_n}/{len(MATMUL_SIZE)}] Testing on {K}*{K} Matrix for {N} times...")
    matmul_try_n += 1
    x = np.random.randn(K, K)
    def run_test():
        t1 = time()
        np.matmul(x, x)
        t2 = time()
        return t2 - t1
    return [run_test() for i in range(N)]


if __name__ == "__main__":
    print("ℹ️ Running matmul_2D...")
    print("")
    matmul_result = []
    for case in MATMUL_SIZE:
        result = matmul_2D(K=case)
        matmul_result.append(mean(result))
    plt.plot(MATMUL_SIZE, matmul_result)
    plt.title(f"matmul (numpy + {BACKEND_NAME}) (N={N})")
    plt.xlabel("Matrix Size")
    plt.ylabel("time (second)")
    plt.savefig(MATMUL_RESULT_DIR)
    # Todo Output to CSV and merge graphs
    print(f"⭕️ The result is correctly saved at {MATMUL_RESULT_DIR}")
    print("Running broadcasting 4D...")
    
