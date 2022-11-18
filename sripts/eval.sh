if [ $# -lt 3 ]
then
    echo "Please run the script as: "
    echo "bash scripts/eval.sh TARGET CKPT_PATH PREDICTION_PATH"
    echo "TARGET: mpii_single, coco_multi"
    echo "For example: bash scripts/eval.sh mpii_single ./ckpt_0/arttrack.ckpt out/prediction.mat"
exit 1
fi
python eval.py "$1" --config config/mpii_eval.yaml --option "load_ckpt=$2" --output "$3"
python eval.py "$1" --config config/mpii_eval.yaml --accuracy  --prediction "$3"
