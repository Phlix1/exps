#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL=lstm_luong_wmt_en_de
BATCH_SIZE=64
shared_path="$DIR"
PARAMS=""
HOSTFILE=""
BACKEND=""
WORLDSIZE=""
RANK=""
INIT=""
rm -f $shared_path/python_init_process_group
while (( "$#" )); do
  case "$1" in
    -b|--batch-size)
      BATCH_SIZE=$2
      shift 2
      ;;
    -h|--hostfile)
      HOSTFILE=$2
      shift 2
      ;;
    --backend)
      BACKEND=$2
      shift 2
      ;;
    --worldsize)
      WORLDSIZE=$2
      shift 2
      ;;
    --rank)
      RANK=$2
      shift 2
      ;;
    --init)
      INIT=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
#    -*|--*=) # unsupported flags
#      echo "Error: Unsupported flag $1" >&2
#      exit 1
#      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

if [ "$IFNAME" = "" ]; then
  IFNAME=ens1f0
fi
if [ "$BACKEND" = "" ]; then
  BACKEND=nccl
fi

export CUDA_VISIBLE_DEVICES=1
export OMPI_COMM_WORLD_SIZE=$WORLDSIZE
export OMPI_COMM_WORLD_RANK=$RANK
export OMPI_COMM_WORLD_LOCAL_RANK=0
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME=ens1f1
export NCCL_SOCKET_IFNAME=ens1f1
export NCCL_IB_HCA=mlx5_1

python lm/main.py --data ./Google-Billion-Words/PyTorch_GBW_LM --backend $BACKEND  --shared_path ${shared_path} --batch_size $BATCH_SIZE --lr 1e-3 --init $INIT $PARAMS 

