import json
import os

import numpy as np
import scipy.io as scio


def mat2json(index_mat, name, dataset_json=None, output_dir=None, index_offset=-1, stdout=False):
    """
    select sample from matlab index mat and and save as json format

    Args:
         index_mat: path to index mat
         name: field in index mat
         dataset_json: path to json format dataset which contains full sample
         output_dir: output dir
         index_offset: offset for index mat. first index in matlab is usually 1, but in python it's 0.
         stdout: if True, print index
    """
    mat = scio.loadmat(index_mat)
    ids = mat.get(name).flatten().tolist()
    ids = [i + index_offset for i in ids]
    if stdout:
        print(ids)
    out_dataset = None
    if dataset_json is not None:
        with open(dataset_json, 'r') as f:
            dataset = f.read()
        dataset = np.array(json.loads(dataset))
        out_dataset = dataset[ids].tolist()

    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
        s = json.dumps(ids)
        with open(os.path.join(output_dir, '%s.json' % name), 'w') as f:
            f.write(s)
        if out_dataset is not None:
            s = json.dumps(out_dataset)
            with open(os.path.join(output_dir, '%s_dataset.json' % name), 'w') as f:
                f.write(s)
