if [ $# -lt 2 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_train_single_gpu.sh TARGET CONFIG_PATH [OPTION] ..."
    echo "TARGET: mpii_single"
    echo "For example: bash scripts/run_train_single_gpu.sh mpii_single config/mpii_train_single_gpu.yaml \"dataset.path=./out/train_index_dataset.json\""
exit 1
fi
set -e
index=0
OPTIONS=''
for arg in "$@"
do
    if [ $index -ge 2 ]
    then
        OPTIONS="$OPTIONS --option $arg"
    fi
    let index+=1
done
python train.py "$1" --config "$2" $OPTIONS | tee "mpii_train_single_gpu-`(date +%Y-%m-%d_%H%M%S)`.log"
