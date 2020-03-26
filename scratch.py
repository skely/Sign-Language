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
    ignore_labels = BVH.get_joint_id(_names, _channels, _joint_name, 'rotation')
    Z = BVH.get_joint_id(_names, _channels, _joint_name, 'Zrotation')[0]
    Y = BVH.get_joint_id(_names, _channels, _joint_name, 'Yrotation')[0]
    X = BVH.get_joint_id(_names, _channels, _joint_name, 'Xrotation')[0]
    # retval = [locals()[order[0]], locals()[order[1]], locals()[order[2]]]
    retval = [X, Y, Z]
    # retval = ignore_labels
    return retval


def get_values(_query, _tree_structure, _verbose=True):
    names, channels, offsets, _, _ = _tree_structure
    # print(channels)
    id = (names.index(_query))
    if len(channels[id]) == 3:
        channel_order = channels[id][:]
    else:
        channel_order = channels[id][3:]

    # print(channel_order)
    channel_order_ret = channel_order[0][0] + channel_order[1][0] + channel_order[2][0]

    offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == _query][0]], dtype=float)
    rotation_ids = get_rotation_axes_ids(names, channels, _query)
    rotation_deg = frame_trajectory[rotation_ids]
    rotation = np.deg2rad(rotation_deg)

    position_ids = tuple([BVH.get_joint_id(names, channels, _query, 'Xposition') + BVH.get_joint_id(names, channels, _query, 'Yposition') + BVH.get_joint_id(names, channels, _query, 'Zposition')])
    position = frame_trajectory[position_ids]
    if len(position) == 0:
        position = [0, 0, 0]
    if _verbose:
        print('\n' + _query)
        print('Offset {}'.format(offset))
        print('Rotation: {}'.format(rotation))
        print('Position: {}'.format(position))
        print('Channel order: {}'.format(channel_order_ret))
    return offset, rotation, position, channel_order_ret[::-1]


def calculate_positions(rot_zero=False):
    for joint in names:
        parents = BVH.get_all_ancestors(joint, tree_structure)
        parents.reverse()
        result = [0, 0, 0]
        last_rotation = [0, 0, 0]
        print(joint, parents)
        for parent in parents:
            # print(parent)
            if parent == 'root':
                pass
            else:
                tmp_offset, tmp_rotation, tmp_position, chorder = get_values(parent, tree_structure)
                # print(parent, tmp_offset)
                # print(chorder)
                if rot_zero:
                    tmp_rotation = [0, 0, 0]

                print(result)
                # print(position)
                result = result + tmp_position + make_rotation(last_rotation, tmp_offset, rot_channel_order)
                # result = result + make_rotation(last_rotation, tmp_offset, chorder)
                last_rotation = tmp_rotation
                # result = make_rotation(tmp_rotation, tmp_offset + result, rot_channel_order)
                # print(result)

        offset, rotation, position, chorder = get_values(joint, tree_structure)
        if rot_zero:
            rotation = [0, 0, 0]
        result = result + position + make_rotation(last_rotation, offset, rot_channel_order)
        # result = result + make_rotation(last_rotation, offset, chorder)
        # result = make_rotation(rotation, offset + result, rot_channel_order)
        # print(joint, offset)
        # print(joint, rotation)
        # print(joint, result)
        if rot_zero:
            ax.scatter(result[0], result[1], result[2], label=joint, marker='*')
        else:
            ax.scatter(result[0], result[1], result[2], label=joint, marker='+')
        # input()

if __name__ == '__main__':
    BVH_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_29_c_FR.bvh'
    header = BVH.load_raw_header(BVH_file)
    trajectory = BVH.load_trajectory(BVH_file)

    tree_structure = BVH.get_tree_structure_joint_list(header)
    names, channels, offsets, _, _ = tree_structure

    # names = ['Hips', 'Spine', 'RightShoulder', 'LeftShoulder', 'RightArm', 'LeftArm']
    # removes finger data
    # names = [name for name in names if '1' not in name]
    # names = [name for name in names if '2' not in name]
    # names = [name for name in names if '3' not in name]

    # frame_number = 1000
    frame_number = 0
    frame_trajectory = trajectory[frame_number, :]
    # all_rots = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
    all_rots = ['ZYX']
    for rot_channel_order in all_rots:

        # ****************************************************

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', title=rot_channel_order)
        unisize = 50
        last_position = [0, 0, 0]


        calculate_positions()
        # plt.gca().set_prop_cycle(None)
        # calculate_positions(rot_zero=True)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim3d(-unisize, unisize)
        ax.set_ylim3d(-unisize, unisize)
        ax.set_zlim3d(-unisize, unisize)
        ax.legend()
        plt.show()
    # #
    # #         x, y, z
    # hips = [0, 0, 0]
    # spine = [0, 2, 0]
    # R_hand = [1, 0, 0]
    # L_hand = [-1, 0, 0]
    # # #            z, y, x
    # rotations1 = [0, 0, 0]
    # rotations2 = [0, 0, 90]
    # rotations3 = [0, 0, 0]
    # rotations4 = [0, 0, 0]
    # #
    # rot_rad1 = np.deg2rad(rotations1)
    # rot_rad2 = np.deg2rad(rotations2)
    # rot_rad3 = np.deg2rad(rotations3)
    # rot_rad4 = np.deg2rad(rotations4)
    # #
    #
    # result = make_rotation(rot_rad1, hips)
    # print('{:.2f}, {:.2f}, {:.2f}'.format(result[0], result[1], result[2]))
    #
    # result2 = result  + make_rotation(rot_rad1, spine)
    # print('{:.2f}, {:.2f}, {:.2f}'.format(result2[0], result2[1], result2[2]))
    #
    # result3 = result2  + make_rotation(rot_rad2, R_hand)
    # print('{:.2f}, {:.2f}, {:.2f}'.format(result3[0], result3[1], result3[2]))
    #
    # result4 = result2  + make_rotation(rot_rad2, L_hand)
    # print('{:.2f}, {:.2f}, {:.2f}'.format(result4[0], result4[1], result4[2]))
