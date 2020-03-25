from lib import BVH
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_rotation(_offset, _rotation):
    alpha, beta, gamma = _rotation  # Z, Y, X
    R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
                  [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
                  [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])

    return R.dot(_offset)


def get_rotation_axes_ids(_names, _channels, _joint_name):
    Z = BVH.get_joint_id(names, channels, query, 'Zrotation')[0]
    Y = BVH.get_joint_id(names, channels, query, 'Yrotation')[0]
    X = BVH.get_joint_id(names, channels, query, 'Xrotation')[0]

    return [Z, Y, X]


# if __name__ == '__main__':
#     BVH_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_29_c_FR.bvh'
#     names, channels, offsets = BVH.get_joint_list(BVH_file)
#     trajectory = BVH.load_trajectory(BVH_file)
#
#     frame_number = 0
#     frame_trajectory = trajectory[frame_number, :]
#
#     # ****************************************************
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     query = 'Hips'
#     print(query)
#     offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == query][0]], dtype=float)
#     ax.scatter(offset[0], offset[1], offset[2], 'ro', label=query)
#
#     rotation_ids = (get_rotation_axes_ids(names, channels, query))
#     rotation_deg = (frame_trajectory[rotation_ids])
#     rotation = np.deg2rad(rotation_deg)
#     # print(offset)
#     # print(rotation_ids)
#     # print(rotation_deg)
#     # print(rotation)
#
#
#
#     query = 'Spine'
#     print(query)
#     offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == query][0]], dtype=float)
#     rotated_position = make_rotation(offset, rotation)
#     ax.scatter(rotated_position[0], rotated_position[1], rotated_position[2], label=query)
#
#     rotation_ids = (get_rotation_axes_ids(names, channels, query))
#     rotation_deg = (frame_trajectory[rotation_ids])
#     rotation = np.deg2rad(rotation_deg)
#     # print(offset)
#     # print(rotation_ids)
#     # print(rotation_deg)
#     # print(rotation)
#
#     query = 'Head'
#     print(query)
#     offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == query][0]], dtype=float)
#     rotated_positionH = rotated_position + make_rotation(offset, rotation)
#     ax.scatter(rotated_position[0], rotated_position[1], rotated_position[2], label=query)
#
#     # rotation_ids = (get_rotation_axes_ids(names, channels, query))
#     # rotation_deg = (frame_trajectory[rotation_ids])
#     # rotation = np.deg2rad(rotation_deg)
#     # print(offset)
#     # print(rotation_ids)
#     # print(rotation_deg)
#     # print(rotation)
#
#     query = 'RightShoulder'
#     print(query)
#     offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == query][0]], dtype=float)
#     rotated_position = rotated_positionH + make_rotation(offset, rotation)
#     print(rotated_position)
#     ax.scatter(rotated_position[0], rotated_position[1], rotated_position[2], label=query)
#
#     query = 'LeftShoulder'
#     print(query)
#     offset = np.asarray(offsets[[i for i, name in enumerate(names) if name == query][0]], dtype=float)
#     rotated_position = rotated_positionH + make_rotation(offset, rotation)
#     print(rotated_position)
#     ax.scatter(rotated_position[0], rotated_position[1], rotated_position[2], label=query)
#
#     # rotation_ids = (get_rotation_axes_ids(names, channels, query))
#     # rotation_deg = (frame_trajectory[rotation_ids])
#     # rotation = np.deg2rad(rotation_deg)
#
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#
#     ax.set_xlim3d(-100, 100)
#     ax.set_ylim3d(-100, 100)
#     ax.set_zlim3d(-100, 100)
#     ax.legend()
#     plt.show()
#
#     #         x, y, z
#     offset = [1, 0, 0]
#     offset2 = [0, 1, 0]
#     #            z, y, x
#     rotations1 = [90, 0, 0]
#     rotations2 = [90, 0, 0]
#
#     rot_rad1 = np.deg2rad(rotations1)
#     rot_rad2 = np.deg2rad(rotations2)
#
#     result = make_rotation(offset, rot_rad1)
#     print('{:.2f}, {:.2f}, {:.2f}'.format(result[0], result[1], result[2]))
#     result2 = make_rotation(result  + offset2, rot_rad2)
#     print('{:.2f}, {:.2f}, {:.2f}'.format(result2[0], result2[1], result2[2]))
#
