if [ $# -lt 4 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_train_single_gpu.sh TARGET CONFIG_PATH CUDA_VISIBLE_DEVICES DEVICE_NUM [OPTION] ..."
    echo "TARGET: mpii_single"
    echo "For example: bash scripts/run_train_multiple_gpu.sh mpii_single config/mpii_train_multiple_gpu.yaml \"0,1,2,3,4,5,6,7\" 8 \"dataset.path=./out/train_index_dataset.json\""
exit 1
fi
set -e
index=0
OPTIONS=''
for arg in "$@"
do
    if [ $index -ge 4 ]
    then
        OPTIONS="$OPTIONS --option $arg"
    fi
    let index+=1
done
export CUDA_VISIBLE_DEVICES=$3
echo "$CUDA_VISIBLE_DEVICES"
mpirun -n "$4" python train.py "$1" --config "$2" $OPTIONS | tee "mpii_train_multiple_gpu-`(date +%Y-%m-%d_%H%M%S)`.log"
