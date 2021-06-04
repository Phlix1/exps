# Benchmark
The benchmark performs allreduce on tensors with different sparsity to test the OmniReduce performance. It is a PyTorch script which uses `torch.distributed` package.

## Requirements
Install PyTorch with OmniReduce according to [this](https://github.com/Phlix1/omnireduce/tree/master/frameworks_integration/pytorch_patch).

## Run the benchmark
### 1. Create and edit omnireduce.cfg
- Read this [example](https://github.com/Phlix1/omnireduce/tree/master/example) and create the your own `omnireduce.cfg` according to the cluster information.
- Copy the `omnireduce.cfg` to all the aggregators and workers. And `omnireduce.cfg` needs to be in the same directory as the PyTorch script (for workers) or program `aggregator` (for aggregators).
### 2. Run aggregators
Aggregator 0 and aggregator 1:

    cd ./omnireduce/example
    ./aggregator

### 3. Run workers
Worker 0:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=eth0 python benchmark.py  -d 1.0 --backend gloo -t 26214400 -r 0 -s 4 --ip IP_OF_NODE0

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=eth0 python benchmark.py  -d 1.0 --backend gloo -t 26214400 -r 1 -s 4 --ip IP_OF_NODE0

Worker 1:

    NCCL_DEBUG=INFO LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=eth0 python benchmark.py  -d 1.0 --backend gloo -t 26214400 -r 2 -s 4 --ip IP_OF_NODE0

    NCCL_DEBUG=INFO LOCAL_RANK=1 CUDA_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=eth0 python benchmark.py  -d 1.0 --backend gloo -t 26214400 -r 3 -s 4 --ip IP_OF_NODE0
