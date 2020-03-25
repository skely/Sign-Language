from lib import BVH
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_rotation(_rotation, _offset):
    alpha, beta, gamma = _rotation  # Z, Y, X
    R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
                  [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
                  [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])

    return R.dot(_offset)


def get_rotation_axes_ids(_names, _channels, _joint_name, order='ZYX'):
    Z = BVH.get_joint_id(_names, _channels, _joint_name, 'Zrotation')[0]
    Y = BVH.get_joint_id(_names, _channels, _joint_name, 'Yrotation')[0]
    X = BVH.get_joint_id(_names, _channels, _joint_name, 'Xrotation')[0]

    retval = [locals()[order[0]], locals()[order[1]], locals()[order[2]]]
    return retval


def get_values(_query, _verbose=True):
    offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == _query][0]], dtype=float)
    rotation_ids = get_rotation_axes_ids(names, channels, _query)
    rotation_deg = frame_trajectory[rotation_ids]
    rotation = np.deg2rad(rotation_deg)

    position_ids = tuple([BVH.get_joint_id(names, channels, _query, 'Xposition') + BVH.get_joint_id(names, channels, _query, 'Yposition') + BVH.get_joint_id(names, channels, _query, 'Zposition')])
    position = frame_trajectory[position_ids]
    if _verbose:
        print('\n' + _query)
        print('Offset {}'.format(offset))
        print('Rotation: {}'.format(rotation))
    return offset, rotation, position


if __name__ == '__main__':
    BVH_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_29_c_FR.bvh'
    names, channels, offsets = BVH.get_joint_list(BVH_file)
    trajectory = BVH.load_trajectory(BVH_file)
    header = BVH.load_raw_header((BVH_file))

    BVH.get_tree_structure(header)

