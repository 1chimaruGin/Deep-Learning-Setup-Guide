# Deep-Learning-Setup-Guide

> Install and usage guide for amazing deep-learning tools.

A practical, copy-paste-friendly guide for setting up a GPU deep-learning
environment on **Ubuntu / Linux**. It covers installing (and cleanly
uninstalling) the NVIDIA stack — driver, CUDA, cuDNN — plus common inference
and training libraries such as ONNX Runtime, TensorRT, PyCUDA and NVIDIA Apex.

> **Tip:** If you only need PyTorch or TensorFlow with GPU support, you usually
> do **not** have to install CUDA / cuDNN system-wide — `conda` can manage them
> per-environment. See [Quick start with Conda](#quick-start-with-conda).

## Contents

- [Quick start with Conda](#quick-start-with-conda)
- [Uninstall and Install CUDA and cuDNN](#uninstall-and-install-cuda-and-cudnn)
- [Install ONNX and ONNX Runtime](#install-onnx-and-onnx-runtime)
- [Install TensorRT](#install-tensorrt)
- [Install PyCUDA](#install-pycuda)
- [Install NVIDIA Apex](#install-nvidia-apex)
- [CUDA C++ programming notes](#cuda-c-programming-notes)

---

## Quick start with Conda

If you want to use PyTorch or the TensorFlow GPU build, the simplest path is to
let `conda` handle CUDA and cuDNN for you:

- You only need to install the **NVIDIA graphics driver** on the host.
- You can create separate environments and keep CUDA / cuDNN versions isolated.
- `conda` takes care of everything (CUDA, cuDNN) inside each environment.
- You can easily switch between environments — no CUDA vs. TF/PyTorch conflicts.

For example, to keep TensorFlow 1.x and 2.x side by side:

**TensorFlow 1.x**

```bash
conda create -n tf1_gpu python==3.7.9
conda activate tf1_gpu
conda install tensorflow-gpu==1.xx
# conda also installs CUDA and cuDNN just for this environment
```

**TensorFlow 2.x**

```bash
conda create -n tf2_gpu python==3.7.9
conda activate tf2_gpu
conda install tensorflow-gpu
# conda also installs CUDA and cuDNN just for this environment
```

**PyTorch** — see the official [PyTorch install guide](https://pytorch.org/get-started/locally/):

```bash
conda create -n torch python==3.7.9
conda activate torch
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
# conda also installs CUDA and cuDNN just for this environment
```

> **Note:** Some libraries still need a specific system CUDA / cuDNN. For
> example, `onnxruntime-gpu` needs CUDA 10.2 and cuDNN 8.0.3, and cuDNN 8.0.3 is
> not available in `conda` — in that case follow the manual install below.

---

## Uninstall and Install CUDA and cuDNN

> If you have never installed CUDA and cuDNN before, skip ahead to
> [Install CUDA and cuDNN](#install-cuda-and-cudnn).

### Uninstall

**Uninstall just `nvidia-cuda-toolkit`**

```bash
sudo apt-get remove nvidia-cuda-toolkit
```

**Uninstall `nvidia-cuda-toolkit` and its dependencies**

```bash
sudo apt-get remove --auto-remove
```

**Purge config/data**

```bash
sudo apt-get purge nvidia-cuda-toolkit
# or
sudo apt-get purge --auto-remove nvidia-cuda-toolkit
```

#### Alternative way

**Uninstall CUDA**

```bash
sudo /usr/local/cuda/bin/uninstallxxx
sudo apt remove --purge cuda
```

**Uninstall cuDNN**

```bash
sudo rm /usr/local/cuda/include/cudnn.h
sudo rm /usr/local/cuda/lib64/libcudnn*
```

**Uninstall NVIDIA driver**

```bash
sudo apt remove --purge nvidia*
```

### Install CUDA and cuDNN

**Install the NVIDIA driver**

```bash
ubuntu-drivers devices
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-drivers-xxx
reboot
```

**Verify the NVIDIA driver**

```bash
nvidia-smi
```

**Install CUDA**

> Download the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

```bash
sudo dpkg -i cuda-repo-ubuntu1804-xx-x-local-xx.x.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

**Verify the CUDA installation**

```bash
nvcc -V
```

**Install cuDNN**

1. Download [cuDNN](https://developer.nvidia.com/rdp/form/cudnn-download-survey).
2. Log in.
3. Accept the terms and agreements.
4. Navigate to the directory containing the cuDNN `.deb` file.
5. Install the runtime library, for example:

   ```bash
   sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_amd64.deb
   ```

6. Install the developer library, for example:

   ```bash
   sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_amd64.deb
   ```

7. Install the code samples and library documentation, for example:

   ```bash
   sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_amd64.deb
   ```

**Verify the cuDNN installation**

Compile and run the `mnistCUDNN` sample:

```bash
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
# Test passed!
```

> ⚠️ **If you hit an error like this:**
>
> ```
> unsupported GNU version! gcc versions later than 8 are not supported!
> ```

Solution:

```bash
sudo apt -y install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
```

Solution for end users:

```bash
cmake -DCMAKE_C_COMPILER=$(which gcc-8) -DCMAKE_CXX_COMPILER=$(which g++-8) -DWITH_CUDA=ON
```

---

## Install ONNX and ONNX Runtime

**Install ONNX**

With `conda`:

```bash
conda install -c conda-forge onnx
```

With `pip`:

```bash
pip install onnx
```

**Install `onnxruntime-gpu`**

Requirements:

- Python 3.6 or 3.7
- CUDA 10.2
- cuDNN 8.0.3

```bash
pip install onnxruntime-gpu
```

**Usage and tutorials**

- [ONNX tutorials](https://github.com/onnx/tutorials)
- [ONNX Runtime docs](https://www.onnxruntime.ai/docs/tutorials/)

---

## Install TensorRT

**Download**

1. Open [TensorRT](https://developer.nvidia.com/tensorrt).
2. Click **Download Now**.
3. Select the version of TensorRT you are interested in.
4. Select the check-box to agree to the license terms.
5. Click the package you want to install — your download begins.

**Install from the Debian local repo package**

```bash
os="ubuntu1x04"
tag="cudax.x-trt7.x.x.x-ga-yyyymmdd"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
```

Optional extras:

```bash
# if you are using python2
sudo apt-get install python-libnvifer-dev

# if you are using python3
sudo apt-get install python3-libnvinfer-dev

# if you plan to use TensorRT with TensorFlow
sudo apt-get install uff-converter-tf

# if you want ONNX graphsurgeon (samples / Python module)
sudo apt-get install onnx-graphsurgeon
```

**Verify the installation**

```bash
dpkg -l | grep TensorRT
```

You should see something similar to the following:

```
ii  graphsurgeon-tf        7.2.1-1+cuda11.1  amd64  GraphSurgeon for TensorRT package
ii  libnvinfer-bin         7.2.1-1+cuda11.1  amd64  TensorRT binaries
ii  libnvinfer-dev         7.2.1-1+cuda11.1  amd64  TensorRT development libraries and headers
ii  libnvinfer-doc         7.2.1-1+cuda11.1  all    TensorRT documentation
ii  libnvinfer-plugin-dev  7.2.1-1+cuda11.1  amd64  TensorRT plugin libraries
ii  libnvinfer-plugin7     7.2.1-1+cuda11.1  amd64  TensorRT plugin libraries
ii  libnvinfer-samples     7.2.1-1+cuda11.1  all    TensorRT samples
ii  libnvinfer7            7.2.1-1+cuda11.1  amd64  TensorRT runtime libraries
ii  libnvonnxparsers-dev   7.2.1-1+cuda11.1  amd64  TensorRT ONNX libraries
ii  libnvonnxparsers7      7.2.1-1+cuda11.1  amd64  TensorRT ONNX libraries
ii  libnvparsers-dev       7.2.1-1+cuda11.1  amd64  TensorRT parsers libraries
ii  libnvparsers7          7.2.1-1+cuda11.1  amd64  TensorRT parsers libraries
ii  python-libnvinfer      7.2.1-1+cuda11.1  amd64  Python bindings for TensorRT
ii  python-libnvinfer-dev  7.2.1-1+cuda11.1  amd64  Python development package for TensorRT
ii  python3-libnvinfer     7.2.1-1+cuda11.1  amd64  Python 3 bindings for TensorRT
ii  python3-libnvinfer-dev 7.2.1-1+cuda11.1  amd64  Python 3 development package for TensorRT
ii  tensorrt               7.2.1.x-1+cuda11.1 amd64 Meta package of TensorRT
ii  uff-converter-tf       7.2.1-1+cuda11.1  amd64  UFF converter for TensorRT package
ii  onnx-graphsurgeon      7.2.1-1+cuda11.1  amd64  ONNX GraphSurgeon for TensorRT package
```

**Usage and tutorials**

- [TensorRT developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

---

## Install PyCUDA

**Install**

> Install PyCUDA along with its dependencies:

```bash
sudo apt-get install build-essential python-dev python-setuptools libboost-python-dev libboost-thread-dev -y
pip install pycuda
```

**Verify the installation**

Create a `.py` file with the following:

```python
import pycuda
import pycuda.driver as drv
drv.init()
print('CUDA device query (PyCUDA version) \n')
print('Detected {} CUDA Capable device(s) \n'.format(drv.Device.count()))
for i in range(drv.Device.count()):

    gpu_device = drv.Device(i)
    print('Device {}: {}'.format( i, gpu_device.name() ) )
    compute_capability = float( '%d.%d' % gpu_device.compute_capability() )
    print('\t Compute Capability: {}'.format(compute_capability))
    print('\t Total Memory: {} megabytes'.format(gpu_device.total_memory()//(1024**2)))

    # The following will give us all remaining device attributes as seen
    # in the original deviceQuery.
    # We set up a dictionary as such so that we can easily index
    # the values using a string descriptor.

    device_attributes_tuples = gpu_device.get_attributes().items()
    device_attributes = {}

    for k, v in device_attributes_tuples:
        device_attributes[str(k)] = v

    num_mp = device_attributes['MULTIPROCESSOR_COUNT']

    # Cores per multiprocessor is not reported by the GPU!
    # We must use a lookup table based on compute capability.
    # See the following:
    # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

    cuda_cores_per_mp = { 5.0 : 128, 5.1 : 128, 5.2 : 128, 6.0 : 64, 6.1 : 128, 6.2 : 128}[compute_capability]

    print('\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores'.format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))

    device_attributes.pop('MULTIPROCESSOR_COUNT')

    for k in device_attributes.keys():
        print('\t {}: {}'.format(k, device_attributes[k]))
```

You should see something similar to the following:

```
CUDA device query (PyCUDA version)

Detected 1 CUDA Capable device(s)

Device 0: GeForce GTX 1060
	 Compute Capability: 6.1
	 Total Memory: 6078 megabytes
	 (10) Multiprocessors, (128) CUDA Cores / Multiprocessor: 1280 CUDA Cores
	 ASYNC_ENGINE_COUNT: 2
	 CAN_MAP_HOST_MEMORY: 1
	 CLOCK_RATE: 1733000
	 COMPUTE_CAPABILITY_MAJOR: 6
	 COMPUTE_CAPABILITY_MINOR: 1
	 COMPUTE_MODE: DEFAULT
	 CONCURRENT_KERNELS: 1
	 ....
	 ....
	 TEXTURE_PITCH_ALIGNMENT: 32
	 TOTAL_CONSTANT_MEMORY: 65536
	 UNIFIED_ADDRESSING: 1
	 WARP_SIZE: 32
```

**Usage and tutorials**

- [PyCUDA tutorial](https://documen.tician.de/pycuda/tutorial.html)

---

## Install NVIDIA Apex

**Requirements**

- Python 3
- CUDA 9 or newer
- PyTorch 0.4 or newer
- The CUDA and C++ extensions require PyTorch 1.0 or newer.

**Install**

### Ubuntu

> For performance and full functionality, install Apex with CUDA and C++
> extensions:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

> Apex also supports a Python-only build (required with PyTorch 0.4):

```bash
pip install -v --disable-pip-version-check --no-cache-dir ./
```

A Python-only build omits:

- Fused kernels required to use `apex.optimizers.FusedAdam`.
- Fused kernels required to use `apex.normalization.FusedLayerNorm`.
- Fused kernels that improve the performance and numerical stability of
  `apex.parallel.SyncBatchNorm`.
- Fused kernels that improve the performance of
  `apex.parallel.DistributedDataParallel` and `apex.amp`.
  `DistributedDataParallel`, `amp` and `SyncBatchNorm` will still be usable, but
  they may be slower.

### Windows

> Windows support is experimental; Linux is recommended.

```bash
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

### Conda install (v0.1)

> To install this package with conda, run one of the following:

```bash
conda install -c conda-forge nvidia-apex
conda install -c conda-forge/label/cf202003 nvidia-apex
```

**Usage and tutorials**

- [NVIDIA Apex](https://github.com/NVIDIA/apex)
- [Apex AMP tutorial](https://nvidia.github.io/apex/amp.html)
- [Mixed-precision webinar](https://info.nvidia.com/webinar-mixed-precision-with-pytorch-reg-page.html)

---

## CUDA C++ programming notes

Once your environment is ready, these notes walk through writing your first
CUDA C++ kernel and making it run in parallel on the GPU:

- [CUDA programming with C++](CUDA_C%2B%2B_start.md) — kernels, unified memory,
  launching a kernel, and profiling with `nvprof`.
- [Picking up the Threads](CUDA_GPU_Threading.md) — thread blocks, grids, and
  grid-stride loops.
