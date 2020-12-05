# Deep-Learning-Setup-Guide

# Contents
- [Uninstall and install CUDA toolkit](#Uninstall-and-Install-CUDA-and-cuDNN )
- [Installation guide for onnx and onnxruntime](#Install-onnx-and-onnxruntime)
- [Installation guide for TensorRT](#Install-TensorRT)
- [Installation guide for pycuda](#Install-pyCUDA)
- [Installaition guide for Nvidia apex](#Install-Nvidia-Apex)


*Note*

> If you want to use pytorch or tensorflow-gpu version, just install using `conda`. 

> You will only need to install nvida graphics drivers.

> You can create various environments and can keep CUDA and 
cuDNN version separately.

> `Conda` will take care of everything such as CUDA or cuDNN.

*For example:*

* If you want to use tensorflow1.xx-gpu and tensorflow2x-gpu separately. 
* For `tensorflow1.xx`
    1. Create envs
    - `$ conda create -n tf1_gpu python==3.7.9`
    2. Activate
    - `$ conda activate tf1_gpu`
    3. Then, install tensorflow1.xx-gpu
    - `$ (tf1_gpu) conda install tensorflow-gpu==1.xx`
    4. Conda will also install CUDA and cuDNN just for this environment.
* For `tensorflow2.xx`
    1. Create envs
    - `$ conda create -n tf2_gpu python==3.7.9`
    2. Activate
    - `$ conda activate tf2_gpu`
    3. Then, install tensorflow1.xx-gpu
    - `$ (tf2_gpu) conda install tensorflow-gpu`
    4. Conda will also install CUDA and cuDNN just for this environment.
* For Pytorch installation see official conda [pytorch Guide](https://pytorch.org/get-started/locally/)
    1. Create envs
    - `$ conda create -n torch python==3.7.9`
    2. Activate
    - `$ conda activate torch`
    3. Then, install tensorflow1.xx-gpu
    - `$ (torch) conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch`
    4. Conda will also install CUDA and cuDNN just for this environment.

> You can easily switch between environments.

> Reduce a lot of unecessary works.

> No CUDA and (TF or Pytorch) version conflict.
> Some library such as `onnxruntime-gpu` needs specific CUDA10.2 and cuDNN8.03 version.(**cuDNN8.03 doesn't availble in `Conda`)


# Uninstall and Install CUDA and cuDNN 

*Note*

>If you never installed CUDA and cuDNN before, skip ahead to [Install Guide](#Install-CUDA-AND-CUDNN)

**Uninstall just nvidia-cuda-toolkit**

```
$ sudo apt-get remove nvidia-cuda-toolkit
```

**Uninstall nvidia-cuda-toolkit and it's dependencies**

```
$ sudo apt-get remove --auto-remove 
```

**Purging config/data**

```
$ sudo apt-get purge nvidia-cuda-toolkit or sudo apt-get purge --auto-remove nvidia-cuda-toolkit
```

## Alternative way

**Uninstall CUDA**

- `$ sudo /usr/local/cuda/bin/uninstallxxx`
- `$ sudo apt remove --purge cuda`

**Uninstall CUDNN**

- `$ sudo rm /usr/local/cuda/include/cudnn.h`
- `$ sudo rm /usr/local/cuda/lib64/libcudnn*`

**Uninstall Nvidia driver**

- `$ sudo apt remove --purge nvidia*`

## Install CUDA AND CUDNN

**Install Nvidia driver**

- `$ ubuntu-drivers devices`
- `$ sudo add-apt-repository ppa:graphics-drivers/ppa`
- `$ sudo apt-get update`
- `$ sudo apt-get install nvidia-drivers-xxx`
- `$ reboot`

**Verify Nvida driver installation**

- `$ nvidia-smi`

**Install CUDA**

>Download [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archivehttps://developer.nvidia.com/cuda-toolkit-archive)

- `$ sudo dpkg -i cuda-repo-ubuntu1804-xx-x-local-xx.x.89-440.33.01_1.0-1_amd64.deb `
- `$ sudo apt-key add /var/cuda-repo-xx-x-local-xx.x.89-440.sudo dpkg -i cuda-repo-ubuntu1804-xx-x-local-xx.x.89-440.33.01_1.0-1_amd64.deb`
- `$ sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub`
- `$ sudo apt-get update`
- `$ sudo apt-get -y install cuda33.01/7fa2af80.pub`
- `$ sudo apt-get update`
- `$ sudo apt-get -y install cuda`

**Verify CUDA Installation**

-  `$ nvcc -V`

**Install cuDNN**

1. Download [cuDNN](https://developer.nvidia.com/rdp/form/cudnn-download-survey)
2. Login 
3. Accept terms and agreements
4. Navigate to your `cudnnpath` directory containing the cuDNN .deb file.
5. Install the runtime library, for example:
    - `$ sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_amd64.deb`
6. Install the developer library, for example:
    - `$ sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_amd64.deb`
7. Install the code samples and the cuDNN library documentation, for example:
    - `$ sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_amd64.deb`

**Verify cuDNN installation**

To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in th

1. Copy the cuDNN samples to a writable path.
    - `$ cp -r /usr/src/cudnn_samples_v8/ $HOME`
2. Go to the writable path.
    - `$ cd  $HOME/cudnn_samples_v8/mnistCUDNN`
3. Compile the mnistCUDNN sample
    - `$ make clean && make`
4. Run the mnistCUDNN sample.
    - `$ ./mnistCUDNN`
    - `Test passed!`

> :warning: **If issues like that happend!** 

```
unsupported GNU version! gcc versions later than 8 are not supported!
```
Solution: 
- `$ sudo apt -y install gcc-8 g++-8`
- `$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8`
- `$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8`

Solution for end users:
- `$ cmake -DCMAKE_C_COMPILER=$(which gcc-8 -DCMAKE_CXX_COMPILER=$(which g++-8) -DWITH_CUDA=O`


# Install onnx and onnxruntime

**Install onnx**

With `Conda`
- `$ conda conda install -c conda-forge onnx`

With `pip`
- `$ pip install onnx`

**Install onnxrutime-gpu**

**Requirements**

- python(3.6 or 3.7)
- CUDA10.2 
- cuDNN8.03

- `$ pip install onnxrutime-gpu`

**Usage and Tutorial**

> More information about [tutorial](https://github.com/onnx/tutorials)

> Documentation about [usage and tutorial](https://www.onnxruntime.ai/docs/tutorials/)

# Install TensorRT

**Downloading Procedure**

1. Download [TensorRT](https://developer.nvidia.com/tensorrt)
2. Click *Download Now*
3. Select the version of TensorRT that you are interested in.
4. Select the check-box to agree to the license terms.
5. Click the package you want to install. Your download begins.

**Installing Procedure**

1. Install TensorRT from the Debian local repo package
    - `$ os="ubuntu1x04"`
    - `$ tag="cudax.x-trt7.x.x.x-ga-yyyymmdd"`
    - `$ sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb`
    - `$ sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub`
    - `$ sudo apt-get update`
    - `$ sudo apt-get install tensorrt`
    > if you are using python2
    - `$ sudo apt-get install python-libnvifer-dev`
    > if you are using python3
    - `$ sudo apt-get install python3-libnvinfer-dev`
    > if you plan to use TensorRT with TensorFlow
    - `$ sudo apt-get install uff-converter-tf`
    > if you would like to run the samples that require ONNX graphsurgeon or use the Python module for your own project, run:
    - `$ sudo apt-get install onnx-graphsurgeon`

**Verify installation**

- `$ dpkg -l | grep TensorRT`

> You should see something similar to the following:
```
ii  graphsurgeon-tf	7.2.1-1+cuda11.1	amd64	GraphSurgeon for TensorRT package
ii  libnvinfer-bin		7.2.1-1+cuda11.1	amd64	TensorRT binaries
ii  libnvinfer-dev		7.2.1-1+cuda11.1	amd64	TensorRT development libraries and headers
ii  libnvinfer-doc		7.2.1-1+cuda11.1	all	TensorRT documentation
ii  libnvinfer-plugin-dev	7.2.1-1+cuda11.1	amd64	TensorRT plugin libraries
ii  libnvinfer-plugin7	7.2.1-1+cuda11.1	amd64	TensorRT plugin libraries
ii  libnvinfer-samples	7.2.1-1+cuda11.1	all	TensorRT samples
ii  libnvinfer7		7.2.1-1+cuda11.1	amd64	TensorRT runtime libraries
ii  libnvonnxparsers-dev		7.2.1-1+cuda11.1	amd64	TensorRT ONNX libraries
ii  libnvonnxparsers7	7.2.1-1+cuda11.1	amd64	TensorRT ONNX libraries
ii  libnvparsers-dev	7.2.1-1+cuda11.1	amd64	TensorRT parsers libraries
ii  libnvparsers7	7.2.1-1+cuda11.1	amd64	TensorRT parsers libraries
ii  python-libnvinfer	7.2.1-1+cuda11.1	amd64	Python bindings for TensorRT
ii  python-libnvinfer-dev	7.2.1-1+cuda11.1	amd64	Python development package for TensorRT
ii  python3-libnvinfer	7.2.1-1+cuda11.1	amd64	Python 3 bindings for TensorRT
ii  python3-libnvinfer-dev	7.2.1-1+cuda11.1	amd64	Python 3 development package for TensorRT
ii  tensorrt		7.2.1.x-1+cuda11.1 	amd64	Meta package of TensorRT
ii  uff-converter-tf	7.2.1-1+cuda11.1	amd64	UFF converter for TensorRT package
ii  onnx-graphsurgeon   7.2.1-1+cuda11.1  amd64 ONNX GraphSurgeon for TensorRT package
```

**Usage and Tutorial**

> More information about [usage and tutorial](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)


# Install pyCUDA

**Installing procedure**
> Use the following commands to install PyCUDA along with its dependencies:
- `$ sudo apt-get install build-essential python-dev python-setuptools libboost-python-dev libboost-thread-dev -y`

- `$ pip install pycuda`

**Verify installation**
> Create *.py file as following:

```
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
        print('\t {}: {}'.format(k, device_attributes[k])
```

> You should see something similar to the following:
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
**Usage and Tutorial**

> More information about [usage and tutorial](https://documen.tician.de/pycuda/tutorial.html)

# Install Nvidia Apex

**Requirements**

- python3
- CUDA 9 or newer
- Pytorch 0.4 or newer
- The CUDA and C++ extensions require pytorch 1.0 or newer.

**Install procedure**

*Ubuntuxx.04*
> For performance and full functionality, we recommend installing Apex with CUDA and C++ extensions via

- `$ git clone https://github.com/NVIDIA/apex`
- `$ cd apex`
- `$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

> Apex also supports a Python-only build (required with Pytorch 0.4) via

- `$ pip install -v --disable-pip-version-check --no-cache-dir ./`

> A Python-only build omits:

- Fused kernels required to use `apex.optimizers.FusedAdam`.
- Fused kernels required to use `apex.normalization.FusedLayerNorm`.
- Fused kernels that improve the performance and numerical stability of `apex.parallel.SyncBatchNorm`.
- used kernels that improve the performance of `apex.parallel.DistributedDataParallel` and `apex.amp`. `DistributedDataParallel`, `amp`, and `SyncBatchNorm` will still be usable, but they may be slower.

*Windows*

> Windows support is experimental, and Linux is recommended.

- `pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .`

*Conda install v0.1*

> To install this package with conda run one of the following:

- `$ conda install -c conda-forge nvidia-apex`
- `$ conda install -c conda-forge/label/cf202003 nvidia-apex`

**Usage and Tutorial**

> More information about [Nvidia apex](https://github.com/NVIDIA/apex)

> More [Tutorial](https://nvidia.github.io/apex/amp.html)

> Watch [Webinar](https://info.nvidia.com/webinar-mixed-precision-with-pytorch-reg-page.html)
