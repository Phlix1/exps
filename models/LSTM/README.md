# LSTM
Our LSTM script relies on [this](https://github.com/rdspring1/PyTorch_GBW_LM). We use PyTorch DistributedDataParallel (DDP) package to achieve distributed training. The dataset we use is [Google Billion Word Dataset for Torch](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/leonardn/billionwords.tar.gz), which is in this folder.

## Requirements
Aside from PyTorch with OmniReduce, ensure you have `Cython` and build and install Log_Uniform Sampler.

**Install Dependencies** :

    conda install -y cython
    cd ./lm/log_uniform
    make && python setup.py install

## LSTM Training
### 1. Create and edit omnireduce.cfg
- Read this [example](https://github.com/Phlix1/omnireduce/tree/master/example) and create the your own `omnireduce.cfg` according to the cluster information.
- Copy the `omnireduce.cfg` to all the aggregators and workers. And `omnireduce.cfg` needs to be in the same directory as the PyTorch script (for workers) or program `aggregator` (for aggregators).
### 2. Run aggregators
Aggregator 0 and aggregator 1:

    cd ./omnireduce/example
    ./aggregator

### 3. Run workers
Worker 0:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 OMPI_COMM_WORLD_SIZE=4 OMPI_COMM_WORLD_RANK=0 ./run.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 OMPI_COMM_WORLD_SIZE=4 OMPI_COMM_WORLD_RANK=1 ./run.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo

Worker 1:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 OMPI_COMM_WORLD_SIZE=4 OMPI_COMM_WORLD_RANK=2 ./run.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=ens1f1 OMPI_COMM_WORLD_SIZE=4 OMPI_COMM_WORLD_RANK=3 ./run.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo
