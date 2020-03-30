from lib import BVH
from BVwHacker import bvh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def make_rotation(_rotation, _offset, _order='XYZ'):
    """

    :param _rotation:
    :param _offset:
    :param _order: order of rotation channels
    :return:
    """
    # if _order == 'XYZ':
    #     alpha, beta, gamma = _rotation
    # elif _order == 'XZY':
    #     alpha, gamma, beta = _rotation
    # elif _order == 'YXZ':
    #     beta, alpha, gamma = _rotation
    # elif _order == 'YZX':
    #     beta, gamma, alpha = _rotation
    # elif _order == 'ZXY':
    #     gamma, alpha, beta  = _rotation
    # elif _order == 'ZYX':
    #     gamma, beta, alpha = _rotation


    # alpha, beta, gamma = _rotation
    gamma, beta, alpha = _rotation

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    if _order == 'XYZ':
        R = Rz.dot(Ry).dot(Rx)
    elif _order == 'XZY':
        R = Ry.dot(Rz).dot(Rx)
    elif _order == 'YXZ':
        R = Rz.dot(Rx).dot(Ry)
    elif _order == 'YZX':
        R = Rx.dot(Rz).dot(Ry)
    elif _order == 'ZXY':
        R = Ry.dot(Rx).dot(Rz)
    elif _order == 'ZYX':
        R = Rx.dot(Ry).dot(Rz)
    else:
        print('ERROR: order of rotation channels should by combination of "X", "Y", "Z", not {}'.format(_order))

    return R.dot(_offset)
    # return _offset.transpose().dot(R)


def get_rotation_axes_ids(_names, _channels, _joint_name):
    id = _names.index(_joint_name)
    if len(_channels[id]) == 3:
        tmp = _channels[id]
    else:
        tmp = _channels[id][3:]

    tmp_order = tmp[0][0] + tmp[1][0] + tmp[2][0]
    Z = BVH.get_joint_id(_names, _channels, _joint_name, 'Zrotation')[0]
    Y = BVH.get_joint_id(_names, _channels, _joint_name, 'Yrotation')[0]
    X = BVH.get_joint_id(_names, _channels, _joint_name, 'Xrotation')[0]
    # rotation_idxs = [locals()[tmp_order[0]], locals()[tmp_order[1]], locals()[tmp_order[2]]]  # rotation vector in original order
    rotation_idxs = [X, Y, Z]
    return rotation_idxs, tmp_order


def get_position_axes_ids(_names, _channels, _joint_name):
    try:
        Z = BVH.get_joint_id(_names, _channels, _joint_name, 'Zposition')[0]
        Y = BVH.get_joint_id(_names, _channels, _joint_name, 'Yposition')[0]
        X = BVH.get_joint_id(_names, _channels, _joint_name, 'Xposition')[0]
    except:
        return []
    return [X, Y, Z]


def get_values(_query, _tree_structure, _frame, _verbose=True):
    names, channels, offsets, _, _ = _tree_structure

    offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == _query][0]], dtype=float)

    rotation_ids, rot_order = get_rotation_axes_ids(names, channels, _query)
    rotation_deg = _frame[rotation_ids]
    rotation = np.deg2rad(rotation_deg)

    position_ids = get_position_axes_ids(names, channels, _query)
    position = _frame[position_ids]
    if len(position) == 0:
        position = [0, 0, 0]

    if _verbose:
        print('\n' + _query)
        print('Offset {}'.format(offset))
        print('Rotation: {}, ({})'.format(rotation_deg, rot_order))
        print('Position: {}'.format(position))

    return offset, rotation, position, rot_order


def calculate_positions(_joint, _tree_structure, _selected_frame, T_pose=False):
    parent_list = BVH.get_all_ancestors(_joint, _tree_structure)[:-1][::-1]
    parent_list.append(_joint)
    # parent_list = parent_list[::-1]
    # print(parent_list)
    last_position = [0, 0, 0]
    last_rotation = [0, 0, 0]
    last_rot_order = 'XYZ'
    for parent in parent_list:

        tmp_offset, tmp_rotation, tmp_position, tmp_order = get_values(parent, _tree_structure, _selected_frame, _verbose=False)
        if T_pose:
            tmp_rotation = [0, 0, 0]

        transformed = last_position + make_rotation(last_rotation, tmp_offset + tmp_position, _order=last_rot_order)  #

        last_offset = tmp_offset.copy()
        last_position = transformed.copy()  # pro jistotu..
        last_rotation = tmp_rotation.copy()
        last_rot_order = tmp_order
        # print(parent)
        # print('position: {}'.format(transformed))
    return last_position


def my_colored_plot(_joint):
    if any(n in _joint for n in ['Right']):
        m_color = 'r'
    elif any(n in _joint for n in ['Left']):
        m_color = 'b'
    else:
        m_color = 'g'

    if any(n in _joint for n in ['1', '2', '3']):  # prsty
        m_shape = '+'
    elif any(n in _joint for n in ['Shoulder']):
        m_shape = 's'
    elif any(n in _joint for n in ['ForeArm']):
        m_shape = 'p'
    elif any(n in _joint for n in ['Arm']):
        m_shape = 'v'
    else:
        m_shape = '*'

    return m_color, m_shape

if __name__ == '__main__':
    BVH_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_29_c_FR.bvh'
    header = BVH.load_raw_header(BVH_file)
    trajectory = BVH.load_trajectory(BVH_file)

    tree_structure = BVH.get_tree_structure_joint_list(header)
    joints, channels, offsets, _, _ = tree_structure

    # joints = ['Hips', 'Spine', 'RightShoulder', 'LeftShoulder', 'RightArm', 'LeftArm']
    # removes finger data
    # joints = [name for name in joints if '1' not in name]
    # joints = [name for name in joints if '2' not in name]
    # joints = [name for name in joints if '3' not in name]

    frame_number = 1000
    selected_frame = trajectory[frame_number, :]

    skeleton = bvh.Skeleton(BVH_file, 1)
    skeleton.updateFrame(frame_number)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title='frame: {}'.format(frame_number))
    unisize = 100

    for joint in joints:
        my_result = calculate_positions(joint, tree_structure, selected_frame, T_pose=False)
        j = skeleton.getJoint(joint)
        result = [j.worldpos[0], j.worldpos[1], j.worldpos[2]]

        print('Joint: {}'.format(joint))
        print('My result: {}'.format(my_result))
        print('Correct result: {}'.format(result))

        m_color, m_shape = my_colored_plot(joint)
        ax.scatter(result[0], result[1], result[2], label=joint, color=m_color, marker=m_shape)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-unisize, unisize)
    ax.set_ylim3d(-unisize, unisize)
    ax.set_zlim3d(-unisize, unisize)
    # ax.legend()
    plt.show()
    # break