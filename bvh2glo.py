from lib import BVH
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def make_rotation(_rotation, _offset, _order='ZYX'):
    X, Y, Z = _rotation
    alpha = locals()[_order[0]]
    beta = locals()[_order[1]]
    gamma = locals()[_order[2]]

    R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
                  [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
                  [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])

    return R.dot(_offset)


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
    rotation_idxs = [locals()[tmp_order[0]], locals()[tmp_order[1]], locals()[tmp_order[2]]]

    return rotation_idxs, tmp_order


def get_position_axes_ids(_names, _channels, _joint_name):
    try:
        Z = BVH.get_joint_id(_names, _channels, _joint_name, 'Zposition')[0]
        Y = BVH.get_joint_id(_names, _channels, _joint_name, 'Yposition')[0]
        X = BVH.get_joint_id(_names, _channels, _joint_name, 'Xposition')[0]
    except:
        return []
    return [X, Y, Z]


def get_values(_query, _tree_structure, _frame, _verbose=False):
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
        print('Rotation: {}'.format(rotation))
        print('Position: {}'.format(position))

    return offset, rotation, position, rot_order


def calculate_positions(_joint, _tree_structure, _selected_frame, T_pose=False, _rotation_order='ZYX'):
    parent_list = BVH.get_all_ancestors(_joint, _tree_structure)[:-1][::-1]
    parent_list.append(_joint)
    # parent_list = parent_list[::-1]
    # print(parent_list)
    last_position = [0, 0, 0]
    last_rotation = [0, 0, 0]
    for parent in parent_list:

        tmp_offset, tmp_rotation, tmp_position, tmp_order = get_values(parent, _tree_structure, _selected_frame, _verbose=False)
        if T_pose:
            tmp_rotation = [0, 0, 0]


        # transformed = last_position + make_rotation(last_rotation, tmp_offset+tmp_position, _order=_rotation_order)

        transformed = last_position + make_rotation(last_rotation, tmp_offset, _order=_rotation_order) + tmp_position
        # transformed = last_position + make_rotation(last_rotation, tmp_offset, _order=tmp_order[::-1]) + tmp_position
        # transformed = last_position + make_rotation(last_rotation, tmp_position, _order=_rotation_order) + tmp_offset


        # transformed = make_rotation(tmp_rotation, tmp_offset, _order=_rotation_order) + last_position + tmp_position

        # transformed = make_rotation(tmp_rotation, tmp_offset+last_position, _order=_rotation_order) + tmp_position
        # transformed = make_rotation(tmp_rotation, tmp_offset + last_position + tmp_position, _order=_rotation_order)
        # transformed = make_rotation(tmp_rotation, tmp_offset+tmp_position, _order=_rotation_order) + last_position

        last_position = transformed.copy()  # pro jistotu..
        last_rotation = tmp_rotation.copy()

    return last_position


def my_colored_plot(_joint):
    if any(n in joint for n in ['Right']):
        m_color = 'r'
    elif any(n in joint for n in ['Left']):
        m_color = 'b'
    else:
        m_color = 'g'

    if any(n in joint for n in ['1', '2', '3']):  # prsty
        m_shape = '+'
    elif any(n in joint for n in ['Shoulder']):
        m_shape = 's'
    elif any(n in joint for n in ['ForeArm']):
        m_shape = 'p'
    elif any(n in joint for n in ['Arm']):
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

    rotation_orders = ['ZYX', 'ZXY', 'XYZ', 'XZY', 'YXZ', 'YZX']
    for rotation_order in rotation_orders:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', title=rotation_order)
        unisize = 50

        for joint in joints:
            result = calculate_positions(joint, tree_structure, selected_frame, T_pose=False, _rotation_order=rotation_order)
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