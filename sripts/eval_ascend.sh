if [ $# -ne 2 ]
then
    echo "Please run the script as: "
    echo "bash scripts/eval_ascend.sh [mpii_single or coco_multi] [CKPT_PATH]"
    echo "For example: bash scripts/eval_ascend.sh mpii_ single ckpt/rand_0/arttrack-1_356.ckpt"
exit 1
fi
python eval.py "$1" --config config/mpii_eval_ascend.yaml --option "$2" --device_target Ascend
