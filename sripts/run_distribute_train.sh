echo "========================================================================"
echo "Please run the script as: "
echo "bash scripts/run_distribute.sh RANK_TABLE"
echo "For example: bash scripts/run_distribute.sh RANK_TABLE"
echo "It is better to use the absolute path."
echo "========================================================================"
set -e
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
RANK_TABLE=$(get_real_path $1)

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export RANK_TABLE_FILE=$RANK_TABLE
export RANK_SIZE=8

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

rm -rf distribute_train
mkdir distribute_train
cd distribute_train
for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd src
    mkdir dataset
    mkdir model
    mkdir tool
    cd tool
    mkdir eval
    mkdir preprocess
    cd ../
    mkdir multiperson
    cd model
    mkdir resnet
    cd ../
    cd ../../../
    cp ./train.py ./distribute_train/device$i
    cp ./config.py ./distribute_train/device$i
    cp ./__init__.py ./distribute_train/device$i
    cp ./log.py ./distribute_train/device$i
    cp ./pose_cfg.yaml ./distribute_train/device$i
    cp ./src/*.py ./distribute_train/device$i/src
    cp ./src/dataset/*.py ./distribute_train/device$i/src/dataset
    cp ./src/multiperson/*.py ./distribute_train/device$i/src/multiperson
    cp ./src/tool/*.py ./distribute_train/device$i/src/tool
    cp ./src/tool/eval/*.py ./distribute_train/device$i/src/tool/eval
    cp ./src/tool/preprocess/*.py ./distribute_train/device$i/src/tool/preprocess
    cp ./src/model/*.py ./distribute_train/device$i/src/model
    cp ./src/model/resnet/*.py ./distribute_train/device$i/src/model/resnet
    cd ./distribute_train/device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py --device_target 'Ascend' --is_model_arts False --run_distribute True > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
