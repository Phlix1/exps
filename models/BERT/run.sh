#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#OUT_DIR=/results/SQuAD

echo "Container nvidia build = " $NVIDIA_BUILD_ID
nccl_ib_disable=0
backend="gloo"
worldsize=${1:-"1"}
rank=${2:-"0"}
init=${3:-"tcp://127.0.0.1:4444"}
epochs=${4:-"1.0"}
init_checkpoint=${5:-"./dataset/checkpoint/bert_large_qa.pt"}
learning_rate=${6:-"3e-5"}
precision=${7:-"fp32"}
num_gpu=${8:-"1"}
seed=${9:-"1"}
squad_dir=${10:-"./dataset/squad/v1.1"}
vocab_file=${11:-"./dataset/vocab.txt"}
OUT_DIR=${12:-"."}
mode=${13:-"train"}
CONFIG_FILE=${14:-"./dataset/checkpoint/bert_config.json"}
max_steps=${15:-"200"}
batch_size=${16:-"4"}


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "-1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  #unset CUDA_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=0
  export OMPI_COMM_WORLD_SIZE=$worldsize
  export OMPI_COMM_WORLD_RANK=$rank
  export OMPI_COMM_WORLD_LOCAL_RANK=0
  export NCCL_DEBUG=INFO
  export GLOO_SOCKET_IFNAME=ens1f1 
  export NCCL_SOCKET_IFNAME=ens1f1 
  export NCCL_IB_HCA=mlx5_1
  #mpi_command="OMPI_COMM_WORLD_SIZE=$worldsize OMPI_COMM_WORLD_RANK=$rank OMPI_COMM_WORLD_LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=INFO NCCL_IB_DISABLE=$nccl_ib_disable GLOO_SOCKET_IFNAME=ens1f1 NCCL_SOCKET_IFNAME=ens1f1 NCCL_IB_HCA=mlx5_1 "
fi

      #-x NCCL_DEBUG=INFO -x NCCL_IB_DISABLE=$nccl_ib_disable -x NCCL_SOCKET_IFNAME=ens1f0 -x SYNTHETIC_COMMUNICATION=omnireduce,10,256,1.0 -x SYNTHETIC_PROFILE=1  -np $num_gpu  --hostfile ./hostfile \

CMD="python run_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
fi
CMD+=" --do_lower_case "
# CMD+=" --old "
# CMD+=" --loss_scale=128 "
BUCKET_SIZE_MB=100
if [ ! -z "$BUCKET_SIZE_MB" ] ; then
    CMD+="--bucket_size_mb=$BUCKET_SIZE_MB "
fi
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" --dist-backend=$backend "
CMD+=" --init=$init "
CMD+=" $use_fp16"

echo "$CMD"
time $CMD
