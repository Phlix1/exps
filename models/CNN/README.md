# CNN
We use the training scripts provided by [Nvidia DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/use_cases/pytorch/resnet50/pytorch-resnet50.html) which implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset. Before training, you should download the [ImageNet dataset](https://www.image-net.org/). The folder dataset has a tiny dataset just for testing.

## Requirements
Aside from PyTorch with OmniReduce, ensure you have 
`Pillow`, `torchvision`, `DALI` and `apex`.

**Install Dependencies** :

    pip install Pillow
    pip install torchvision===0.8.0 --no-dependencies
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda100
    git clone https://github.com/NVIDIA/apex
    cd apex
    git reset --hard a651e2c2
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

## CNN Training
### 1. Create and edit omnireduce.cfg
- Read this [example](https://github.com/Phlix1/omnireduce/tree/master/example) and create the your own `omnireduce.cfg` according to the cluster information.
- Copy the `omnireduce.cfg` to all the aggregators and workers. And `omnireduce.cfg` needs to be in the same directory as the PyTorch script (for workers) or program `aggregator` (for aggregators).
### 2. Run aggregators
Aggregator 0 and aggregator 1:

    cd ./omnireduce/example
    ./aggregator

### 3. Run workers
#### ResNet152
Worker 0:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a resnet152 --lr 0.1 --world-size 4 --rank 0 --local_rank 0 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a resnet152 --lr 0.1 --world-size 4 --rank 1 --local_rank 1 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/

Worker 1:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a resnet152 --lr 0.1 --world-size 4 --rank 2 --local_rank 0 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a resnet152 --lr 0.1 --world-size 4 --rank 3 --local_rank 1 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/

#### VGG19
Worker 0:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a vgg19 --lr 0.01 --world-size 4 --rank 0 --local_rank 0 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a vgg19 --lr 0.01 --world-size 4 --rank 1 --local_rank 1 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/

Worker 1:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a vgg19 --lr 0.01 --world-size 4 --rank 2 --local_rank 0 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 python main.py -a vgg19 --lr 0.01 --world-size 4 --rank 3 --local_rank 1 --dist-url tcp://IP_OF_NODE0:FREEPORT --dist-backend gloo  ./dataset/
