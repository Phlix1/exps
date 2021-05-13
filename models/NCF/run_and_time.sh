#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=1.0

# Get command line seed
seed=${1:-1}

# Get the multipliers for expanding the dataset
USER_MUL=${USER_MUL:-4}
ITEM_MUL=${ITEM_MUL:-16}


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
OMPI_COMM_WORLD_LOCAL_RANK=0 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=1 NCCL_IB_DISABLE=0 GLOO_SOCKET_IFNAME=ens1f1  NCCL_SOCKET_IFNAME=ens1f1 python ncf.py \
    -l 0.0002 \
    -b 1048576 \
    --layers 256 256 128 64 \
    -f 64 \
	--seed $seed \
    --threshold $THRESHOLD \
    --user_scaling ${USER_MUL} \
    --item_scaling ${ITEM_MUL} \
    --cpu_dataloader \
    --random_negatives \
    --backend gloo
# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="recommendation"
echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"





