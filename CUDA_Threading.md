## *Picking up the Threads*

Now that you’ve run a kernel with one thread that does some computation, how do you make it parallel? The key is in CUDA’s `<<<1, 1>>>` syntax. This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU. There are two parameters here, but let’s start by changing the second one: the number of threads in a thread block. CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.

`add<<<1, 256>>>(N, x, y)`

If I run the code with only this change, it will do the computation once per thread, rather than spreading the computation across the parallel threads. To do it properly, I need to modify the kernel. CUDA C++ provides keywords that let kernels get the indices of the running threads. Specifically, threadIdx.x contains the index of the current thread within its block, and blockDim.x contains the number of threads in the block. I’ll just modify the loop to stride through the array with parallel threads.

```
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
```

The add function hasn’t changed that much. In fact, setting index to 0 and stride to 1 makes it semantically identical to the first version.

Save the file as add_block.cu and compile and run it in nvprof again. For the remainder of the post I’ll just show the relevant line from the output.

```
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  2.7107ms         1  2.7107ms  2.7107ms  2.7107ms  add(int, float*, float*)
```

## *Out of the Blocks*

CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. Each SM can run multiple concurrent thread blocks. As an example, a Tesla P100 GPU based on the [Pascal GPU Architecture](https://developer.nvidia.com/blog/inside-pascal/) has 56 SMs, each capable of supporting up to 2048 active threads. To take full advantage of all these threads, I should launch the kernel with multiple thread blocks.

By now you may have guessed that the first parameter of the execution configuration specifies the number of thread blocks. Together, the blocks of parallel threads make up what is known as the grid. Since I have `N` elements to process, and 256 threads per block, I just need to calculate the number of blocks to get at least `N` threads. I simply divide `N` by the block size (being careful to round up in case `N` is not a multiple of `blockSize`).

```
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```

I also need to update the kernel code to take into account the entire grid of thread blocks. CUDA provides `gridDim.x`, which contains the number of blocks in the grid, and `blockIdx.x`, which contains the index of the current thread block in the grid. Figure 1 illustrates the the approach to indexing into an array (one-dimensional) in CUDA using `blockDim.x`,  , and `threadIdx.x`. The idea is that each thread gets its index by computing the offset to the beginning of its block (the block index times the block size:    * `blockDim.x`) and adding the thread’s index within the block (`threadIdx.x`). The code `blockIdx.x` * `blockDim.x` + `threadIdx.x` is idiomatic CUDA.

```
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```

The updated kernel also sets stride to the total number of threads in the grid (`blockDim.x` * `gridDim.x`). This type of loop in a CUDA kernel is often called a grid-stride loop.

Save the file as `add_grid.cu` and compile and run it in `nvprof` again.

```
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  94.015us         1  94.015us  94.015us  94.015us  add(int, float*, float*)
```

