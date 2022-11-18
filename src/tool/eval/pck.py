import numpy as np
from numpy import array as arr


def enclosing_rect(points):
    """
    enclose rectangle
    """
    xs = points[:, 0]
    ys = points[:, 1]
    return np.array([np.amin(xs), np.amin(ys), np.amax(xs), np.amax(ys)])


def rect_size(rect):
    """
    get rectangle size
    """
    return np.array([rect[2] - rect[0], rect[3] - rect[1]])


def print_results(pck, cfg):
    """
    print result
    """
    _str = ""
    for heading in cfg.all_joints_names + ["total"]:
        _str += " & " + heading
    print(_str)

    _str = ""
    all_joint_ids = cfg.all_joints + [np.arange(cfg.num_joints)]
    for j_ids in all_joint_ids:
        j_ids_np = arr(j_ids)
        pck_av = np.mean(pck[j_ids_np])
        _str += " & {0:.1f}".format(pck_av)
    print(_str)
