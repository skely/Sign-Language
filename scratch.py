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

    frame_number = 0
    frame_trajectory = trajectory[frame_number, :]

    # ****************************************************
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unisize = 10
    last_position = [0, 0, 0]
    for name in names[:4]:
        query = name
        offset, rotation, position = get_values(query)
        abs_position = last_position + make_rotation(rotation, offset)
        # abs_position = offset
        print('Absolute position {}'.format(abs_position))
        ax.scatter(abs_position[0], abs_position[1], abs_position[2], label=query)
        last_position = abs_position.copy()


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-unisize, unisize)
    ax.set_ylim3d(-unisize, unisize)
    ax.set_zlim3d(-unisize, unisize)
    ax.legend()
    plt.show()

    #         x, y, z
    offset = [0, 1, 0]
    offset2 = [-1, 0, 0]
    offset3 = [1, 0, 0]
    # #            z, y, x
    rotations1 = [0, 0, 0]
    rotations2 = [0, 0, 0]
    rotations3 = [0, 0, 0]
    #
    rot_rad1 = np.deg2rad(rotations1)
    rot_rad2 = np.deg2rad(rotations2)
    rot_rad3 = np.deg2rad(rotations3)
    #
    result = make_rotation(offset, rot_rad1)
    print('{:.2f}, {:.2f}, {:.2f}'.format(result[0], result[1], result[2]))
    result2 = result  + make_rotation(rot_rad2, offset2)
    print('{:.2f}, {:.2f}, {:.2f}'.format(result2[0], result2[1], result2[2]))
    result3 = result2  + make_rotation(rot_rad3, offset3)
    print('{:.2f}, {:.2f}, {:.2f}'.format(result3[0], result3[1], result3[2]))
    # result4 = result2  + make_rotation(offset3, rot_rad3)
    # print('{:.2f}, {:.2f}, {:.2f}'.format(result4[0], result4[1], result4[2]))

