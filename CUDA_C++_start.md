# CUDA programming with C++

This MarkDown file is copy from [Mark Harris's *"An Even Easier Introduction to CUDA"*](https://developer.nvidia.com/blog/even-easier-introduction-cuda/#:~:text=CUDA%20C%2B%2B%20is%20just%20one,parallel%20threads%20running%20on%20GPUs.)

## *Simple program in C++*

***C++ program***

```
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}
```
As expected, it prints that there was no error in the summation and then exits. Now I want to get this computation running (in parallel) on the many cores of a GPU. It’s actually pretty easy to take the first steps.

First, I just have to turn our add function into a function that the GPU can run, called a kernel in CUDA. To do this, all I have to do is add the specifier __global__ to the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code

- Specifier `__global__` which tells the CUDA C++ compiler to run this function on GPU and can be called from CPU code.

```
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}
```

- `__global__` functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.

## *Memory Allocation in CUDA*

To compute on the GPU, I need to allocate memory accessible by the GPU. Unified Memory in CUDA makes this easy by providing a single memory space accessible by all GPUs and CPUs in your system. To allocate data in unified memory, call `cudaMallocManaged()`, which returns a pointer that you can access from host (CPU) code or device (GPU) code. To free the data, just pass the pointer to `cudaFree()`.


I just need to replace the calls to new in the code above with calls to `cudaMallocManaged()`, and replace calls to `delete []` with calls to `cudaFree`.

```
 // Allocate Unified Memory -- accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  ...

  // Free memory
  cudaFree(x);
  cudaFree(y);
```

Finally, I need to launch the add() kernel, which invokes it on the GPU. CUDA kernel launches are specified using the triple angle bracket syntax <<< >>>. I just have to add it to the call to add before the parameter list.

`add<<<1, 1>>>(N, x, y)`

Easy! I’ll get into the details of what goes inside the angle brackets soon; for now all you need to know is that this line launches one GPU thread to run add().

Just one more thing: I need the CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread). To do this I just call `cudaDeviceSynchronize()` before doing the final error checking on the CPU.

Here’s the complete code:

```
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
```

This is only a first step, because as written, this kernel is only correct for a single thread, since every thread that runs it will perform the add on the whole array. Moreover, there is a [race condition](https://en.wikipedia.org/wiki/Race_condition) since multiple parallel threads would both read and write the same locations.

Note: on Windows, you need to make sure you set Platform to x64 in the Configuration Properties for your project in Microsoft Visual Studio.


## *Profile it!*

 think the simplest way to find out how long the kernel takes to run is to run it with `nvprof`, the command line GPU profiler that comes with the CUDA Toolkit. Just type `nvprof ./add_cuda` or `nvprof --unified-memory-profiling off ./add_cuda` on the command line:

 ```
==15975== NVPROF is profiling process 15975, command: ./add_cuda
Max error: 0==15975== NVPROF is profiling process 15975, command: ./add_cuda
Max error: 0
==15975== Profiling application: ./add_cuda
==15975== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  86.205ms         1  86.205ms  86.205ms  86.205ms  add(int, float*, float*)
      API calls:   52.94%  361.97ms         1  361.97ms  361.97ms  361.97ms  cudaProfilerStart
                   34.20%  233.88ms         1  233.88ms  233.88ms  233.88ms  cudaDeviceReset
                   12.61%  86.215ms         1  86.215ms  86.215ms  86.215ms  cudaDeviceSynchronize
                    0.14%  983.87us         2  491.93us  351.12us  632.75us  cudaFree
                    0.05%  337.51us         1  337.51us  337.51us  337.51us  cuDeviceTotalMem
                    0.04%  246.22us        97  2.5380us     321ns  102.92us  cuDeviceGetAttribute
                    0.01%  54.763us         1  54.763us  54.763us  54.763us  cuDeviceGetName
                    0.01%  53.450us         2  26.725us  8.5660us  44.884us  cudaMallocManaged
                    0.01%  43.301us         1  43.301us  43.301us  43.301us  cudaLaunchKernel
                    0.00%  2.5250us         1  2.5250us  2.5250us  2.5250us  cuDeviceGetPCIBusId
                    0.00%  2.2040us         2  1.1020us     390ns  1.8140us  cuDeviceGetCount
                    0.00%  1.5730us         2     786ns     371ns  1.2020us  cuDeviceGet
                    0.00%  1.0020us         1  1.0020us  1.0020us  1.0020us  cudaProfilerStop
                    0.00%     551ns         1     551ns     551ns     551ns  cuDeviceGetUuid83.87us         2  491.93us  351.12us  632.75us  cudaFree
                    0.05%  337.51us         1  337.51us  337.51us  337.51us  cuDeviceTotalMem
                    0.04%  246.22us        97  2.5380us     321ns  102.92us  cuDeviceGetAttribute
                    0.01%  54.763us         1  54.763us  54.763us  54.763us  cuDeviceGetName
                    0.01%  53.450us         2  26.725us  8.5660us  44.884us  cudaMallocManaged
                    0.01%  43.301us         1  43.301us  43.301us  43.301us  cudaLaunchKernel
                    0.00%  2.5250us         1  2.5250us  2.5250us  2.5250us  cuDeviceGetPCIBusId
                    0.00%  2.2040us         2  1.1020us     390ns  1.8140us  cuDeviceGetCount
                    0.00%  1.5730us         2     786ns     371ns  1.2020us  cuDeviceGet
                    0.00%  1.0020us         1  1.0020us  1.0020us  1.0020us  cudaProfilerStop
                    0.00%     551ns         1     551ns     551ns     551ns  cuDeviceGetUuid
 ```

 [Picking up the Threads](/CUDA_GPU_Threading.md)
