import json
import os

import numpy as np


def split(path, out, test_ratio=0.2):
    """
    split dataset
    Args:
        path: path to json format dataset which contains full sample
        out: output dir
        test_ratio: eval ratio
    """
    with open(path, 'r') as f:
        dataset = json.load(f)
        dataset = np.array(dataset)
    np.random.seed(1256)
    dataset_len = len(dataset)
    shuffled_indices = np.random.permutation(dataset_len)
    test_size = int(dataset_len * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    with open(os.path.join(out, "train-dataset.json"), 'w') as f:
        f.write(json.dumps(dataset[train_indices].tolist()))
    with open(os.path.join(out, "eval-dataset.json"), 'w') as f:
        f.write(json.dumps(dataset[test_indices].tolist()))
