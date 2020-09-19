# How to optimize DGEMM on x86 CPU platforms

General matrix/matrix multiplication (GEMM) is a core routine of many popular algorithms. On modern computing platforms with hierarchical memory architecture, it is typically possible that we can reach near-optimal performance for GEMM. For example, on most x86 CPUs, Intel MKL, as well as other well-known BLAS implementations including OpenBLAS and BLIS, can provide >90% of the peak performance for GEMM. On the GPU side, cuBLAS, provided by NVIDIA, can also provide near-optimal performance for GEMM. Though optimizing serial implementation of GEMM on x86 platforms is never a new topic, a tutorial discussing optimizing GEMM on x86 platforms with AVX512 instructions is missing among existing learning resources online. Additionally, with the increasing on data width compared between AVX512 and its predecessors AVX2, AVX, SSE4 and etc, the gap between the peak computational capability and the memory bandwidth continues growing. This simultaneously gives rise of the requirement on programmers to design more delicate prefetching schemes in order to hide the memory latency. Comparing with existed turials, ours is the first one which not only touches the implementation leaveraging AVX512 instructions, and provides step-wise optimization with prefetching strategies as well. The DGEMM implementation eventually reaches comparable performance to Intel MKL.

# Hardware platforms and software configurations

* We require a CPU with the CPU flag ```avx512f``` to run all test cases in this tutorial. This can be checked on terminal using the command: ```lscpu | grep "avx512f"```. 
* The experimental data shown are collected on an Intel Xeon W-2255 CPU (2 AVX512 units, base frequency 3.7 GHz, turbo boost frequency running AVX512 instructions on a single core: 4.0 GHz). This workstation is equipped with 4X8GB=32GB DRAM at 2933 GHz. The theoretical peak performance on a single core is: 2(FMA)*2(AVX512 Units)*512(data with)/64(bit of a fp64 number)*4 GHz = 128 GFLOPS.
* We compiled the program with ```gcc 7.3.0``` under Ubuntu 18.04.5 LTS.
* Intel MKL version: oneMKL 2021.1 beta.

# How to run
Just three steps.
* We first modify the path of MKL in ```Makefile```.
* Second, type in ```make``` to compile. A binary executable ```dgemm_x86``` will be generated.
* Third, run the binary using ```./dgemm_x86 [kernel_number]```, where ```kernel_number``` selects the kernel for benchmark. ```0``` represents Intel MKL and ```1-19``` represent 19 kernels demonstrating the optimizing strategies.

# Step-wise Optimizations

Here we takes the column-major implemetation for DGEMM.

## Kernel 1
[source code](https://github.com/yzhaiustc/GEMM/blob/master/include/kernel1.h)

Kernel 1 is the most naive implementation of DGEMM.

## Kernel 2
[source code](https://github.com/yzhaiustc/GEMM/blob/master/include/kernel2.h)

Observing the innermost loop in [kernel1](https://github.com/yzhaiustc/GEMM/blob/master/include/kernel1.h), ```C(i,j)``` is irrelevant to the innermost loop index ```k```, therefore, one can load it into the register before entering the k-loop to avoid unnecessary memory access.