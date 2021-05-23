#read omnireduce.cfg
i=1
wnum=0
anum=0
while read line; do

if [[ $line =~ "num_workers"  ]]
then
    wnum=$((${line: 14}))
    echo "Worker number: $wnum"
fi
if [[ $line =~ "num_aggregators"  ]]
then
    anum=$((${line: 18}))
    echo "Aggregator number: $anum"
fi
if [[ $line =~ "worker_ips"  ]]
then
    #echo "$line"
    line=${line: 13}
    worker_arr=(${line//,/ })
    j=0
    while [ $j -lt $wnum ]
    do
        echo "worker $j IP : ${worker_arr[$j]}"
	j=$((j+1))
    done
fi
if [[ $line =~ "aggregator_ips"  ]]
then
    #echo "$line"
    line=${line: 17}
    aggregator_arr=(${line//,/ })
    j=0
    while [ $j -lt $anum ]
    do
        echo "aggregator $j IP : ${aggregator_arr[$j]}"
	j=$((j+1))
    done
fi
i=$((i+1))
done < omnireduce.cfg

#copy omnireduce.cfg
i=0
while [ $i -lt $wnum ]
do
    scp -P 2222 ./omnireduce.cfg ${worker_arr[$i]}:/usr/local/omnireduce/example/
    i=$((i+1))
done
i=0
while [ $i -lt $anum ]
do
    scp -P 2222 ./omnireduce.cfg ${aggregator_arr[$i]}:/usr/local/omnireduce/example/
    i=$((i+1))
done
    # start aggregators
i=0
while [ $i -lt $anum ]
do
    ssh -p 2222 ${aggregator_arr[$i]} "pkill -9 aggregator"
    ssh -p 2222 ${aggregator_arr[$i]} "cd /usr/local/omnireduce/example; nohup ./aggregator > aggregator.log 2>&1 &"
    i=$((i+1))
done
# start workers
i=0
while [ $i -lt $wnum ]
do
    ssh -p 2222 ${worker_arr[$i]} "cd /home/exps/models/DeepLight; mkdir -p ./100G-results/omnireduce/ ; export CUDA_VISIBLE_DEVICES=1; export GLOO_SOCKET_IFNAME=ens1f1; export OMPI_COMM_WORLD_SIZE=${wnum}; export OMPI_COMM_WORLD_RANK=$i; export OMPI_COMM_WORLD_LOCAL_RANK=0; nohup /usr/local/conda/bin/python main_all.py -l2 6e-7 -n_epochs 2 -warm 2 -prune 1 -sparse 0.90  -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -backend gloo -batch_size 2048  -init tcp://${worker_arr[0]}:4000 > ./100G-results/omnireduce/log.txt 2>&1 &"
    i=$((i+1))
done
# check completed
while [[ 1 ]]
do
    count=`ps -ef |grep python |grep -v "grep" |wc -l`
    if [ 0 == $count ];then
	break
    fi
done
i=0
while [ $i -lt $anum ]
do
    ssh -p 2222 ${aggregator_arr[$i]} "pkill -9 aggregator"
    i=$((i+1))
done
