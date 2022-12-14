import json
import os

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from mindspore import Parameter
from mindspore.train.serialization import save_checkpoint

from src.log import log


def tf2ms(tf_checkpoint, out_path, map_path):
    """
    convert tensorflow's checkpoint to mindspore model
    Args:
        tf_checkpoint: path to tensorflow's checkpoint
        out_path: path to output
        map_path: map config
    """
    reader = tf.train.NewCheckpointReader(tf_checkpoint)
    new_params_list = []
    with open(map_path) as f:
        s = f.read()
        param = json.loads(s)
    for k, v in param.items():
        param_dict = {}
        try:
            parameter = reader.get_tensor(k)
        except NotFoundError:
            log.warning("not found %s skip", k)
            continue
        if len(parameter.shape) == 4:
            parameter = np.transpose(parameter, axes=[3, 2, 0, 1])
        elif len(parameter.shape) != 1:
            log.error('unknown shape %s for %s', parameter.shape, k)
            exit(1)
        log.info('convert %s -> %s', k, v)

        param_dict['name'] = param[k]
        param_dict['data'] = Parameter(parameter, requires_grad=False)
        new_params_list.append(param_dict)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_checkpoint(new_params_list, out_path)
