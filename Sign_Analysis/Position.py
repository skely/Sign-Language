import BC.BVH as BVH_tools
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D


def read_file(_file):
    """
    :param _file:
    :return: raw_data
    """
    _raw_data = []
    with open(_file, 'r') as _f:
        for _line in _f:
            _raw_data.append(_line)
    return _raw_data


def strip_data(_raw_data):
    """
    :param _raw_data: raw data
    :return: stripped data
    """
    _stripped_data = []
    for _line in _raw_data:
        _line = _line.strip()
        _stripped_data.append(_line)
    return _stripped_data


def load_data(_raw_data):
    _header = []
    _data_list = []
    _it = 0
    header_limit = np.infty
    _raw_data = strip_data(_raw_data)
    for _line in _raw_data:
        if 'Frames:' in _line:
            header_limit = _it + 2
        if _it < header_limit:
            _header.append(_line)
        if _it >= header_limit:
            _data_list.append(_line)
        _it += 1

    _tst = _data_list[0].strip().split(' ')
    if _tst[0] == '':
        padding = 1
        data_matrix = np.zeros((len(_data_list), len(_data_list[0].split(' '))-1))  # (frames, coordinates(x1, y1, z1, rx1, ry1, rz1, rx2, rx3, ...)
    else:
        padding = 0
        data_matrix = np.zeros((len(_data_list), len(_data_list[0].split(' '))))  # (frames, coordinates(x1, y1, z1, rx1, ry1, rz1, rx2, rx3, ...)
    for i in range(len(_data_list)):
        _temp = _data_list[i].split(' ')
        if padding:
            for j in range(len(_temp)-1):
                data_matrix[i, j] = float(_temp[j+1])
        else:
            for j in range(len(_temp)):
                data_matrix[i, j] = float(_temp[j])


    return data_matrix, _header

def explain_header(_header):
    """
    :param _header: header
    :return: dictionary of marker info
    """
    _data_header = _header[:len(_header) - 3]
    _header_dict = []
    _parents = {}
    _idx = 0
    while _idx < len(_data_header):
        if _data_header[_idx].startswith('ROOT') or _data_header[_idx].startswith('JOINT'):
            _marker_info = {}
            # if not _data_header[_idx].startswith('ROOT'):
            # print(_data_header[_idx].split(' '))
                # _marker_info['parent'] = _data_header[_idx-4].split(' ')[1]
            _marker_info['m_name'] = _data_header[_idx].split(' ')[1]
            if _data_header[_idx+1] == '{' and _data_header[_idx+2].startswith('OFFSET'):
                _marker_info['offset'] = [_data_header[_idx+2].split(' ')[1], _data_header[_idx+2].split(' ')[2], _data_header[_idx+2].split(' ')[3]]
            else:
                print('Invalid BVH structure: OFFSET')
                _marker_info['offset'] = -1
            if _data_header[_idx+1] == '{' and _data_header[_idx+3].startswith('CHANNELS'):
                _marker_info['ch_count'] = int(_data_header[_idx+3].split(' ')[1])
                _ch_names = []
                for _i, _channel_attribute in enumerate(_data_header[_idx+3].split(' ')):
                    if _i < 2:
                        continue
                    _ch_names.append(_data_header[_idx+3].split(' ')[_i])
                _marker_info['ch_names'] = _ch_names
            else:
                print('Invalid BVH structure: CHANNELS')
                _marker_info['ch_count'] = -1
                _marker_info['ch_names'] = -1
            _joints = []
            _children = []
            _brac_count = 0
            for _i, _line in enumerate(_data_header[_idx+1:]):
                if _line.startswith('{'):
                    _brac_count += 1
                elif _line.startswith('}'):
                    _brac_count -= 1
                if _brac_count <= 0:
                    break
                if _line.startswith('JOINT'):
                    if _brac_count == 1:
                        _joints.append(_line.split(' ')[1])
                    _children.append(_line.split(' ')[1])
                    _parents[_line.split(' ')[1]] = _marker_info['m_name']
            _marker_info['joints'] = _joints
            _marker_info['children'] = _children
            if not _marker_info['m_name'] in _parents.keys():
                _marker_info['parent'] = None
            else:
                _marker_info['parent'] = _parents[_marker_info['m_name']]
            _header_dict.append(_marker_info)
        _idx += 1
    return _header_dict


def get_index(_name, _header):
    """
    :param _name: string
    :param _header: header
    :return: indexes of marker
    """
    _indexes = []
    _marker_dict = explain_header(_header)
    _idx = 0
    for _marker in _marker_dict:
        if _marker['m_name'] == _name:
            while len(_indexes) < _marker['ch_count']:
                _indexes.append(_idx)
                _idx += 1
            break
        else:
            _idx += _marker['ch_count']
    return _indexes


def get_children_index(_name, _header):
    """
    :param _name: string
    :param _header: header
    :return: indexes of all children
    """
    _indexes = []
    _marker_dict = explain_header(_header)
    for _marker in _marker_dict:
        # print(_marker)
        if _marker['m_name'] == _name:
            for _child in _marker['children']:
                # print(_child)
                _indexes.extend(get_index(_child, _header))
            break
    return _indexes


def get_parent_name(_name, _header):
    """
    :param _name: string
    :param _header: header
    :return: name of parent
    """
    # _prev_marker = None
    _marker_dict = explain_header(_header)
    for _marker in _marker_dict:
        # print(_marker)
        if _marker['m_name'] == _name:
            return _marker['parent']
            # return _prev_marker
        # _prev_marker = _marker
    return -1


if __name__ == '__main__':
    if os.path.exists('D:/Znakovka/dict_solved/'):
        in_path = 'D:/Znakovka/dict_solved/'
        out_path = 'D:/Škola/FAV/PRJ/BC/Output/'
        dict_path = 'D:/Znakovka/Dictionary'
        done_dir_name = 'D:/Znakovka/BVH_reload'
    elif os.path.exists('C:/Znakovka/dict_solved/'):
        in_path = 'C:/Znakovka/dict_solved/'
        out_path = 'C:/Škola/FAV/PRJ/PRJ5/Output/'
        dict_path = 'C:/Znakovka/Dictionary'
        done_dir_name = 'C:/Znakovka/BVH_reload'
    else:
        print("BAD path")
        sys.exit(-1)
    in_file = os.path.join(in_path, 'ciselne_hodnoty_04_solved_body_R.bvh')
    bvh_list = os.listdir(done_dir_name)
    working_trajectory = BVH_tools.load_trajectory(in_file)
    # header = BVH_tools.load_raw_header(in_file)
    _, header = load_data(read_file(in_file))
    joint_list = BVH_tools.get_joint_list(in_file)
    result = np.zeros([np.size(working_trajectory, 0), len(joint_list[0])])
    print(np.shape(working_trajectory))
    print(get_parent_name('RightMiddle2b', header))

    idx = 0
    print('Start----------------------------------------------------')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print(get_parent_name('LeftShoulder', header))
    for frame in range(np.size(working_trajectory, 0)):
        position_per_joint = {}
        for i_joint, joint in enumerate(joint_list[0]):
            if len(joint_list[2][i_joint]) != 3:
                print('FATAL ERROR')
                break
            joint_offset = [float(i) for i in joint_list[2][i_joint]]
            joint_channels = joint_list[1][i_joint]
            joint_name = joint_list[0][i_joint]

            print(joint_name)
            print('Channels: {}'.format(joint_channels))
            # print(joint_offset, joint_channels, joint_name)
            # print(BVH_tools.get_joint_id(joint_list[0], joint_list[1], joint_list[0][i_joint]))
            if joint_list[0][i_joint] == 'Hips':
                parent_position = np.zeros([len(joint_list[2][i_joint])])
            else:
                parent_position = position_per_joint[get_parent_name(joint_list[0][i_joint], header)]

            position_from_data = np.zeros([3])
            if 'Xposition' in joint_list[1][i_joint]:
                joint_ids = BVH_tools.get_joint_id(joint_list[0], joint_list[1], joint_list[0][i_joint])
                pos_x = int(joint_ids[joint_channels.index('Xposition')])
                pos_y = int(joint_ids[joint_channels.index('Yposition')])
                pos_z = int(joint_ids[joint_channels.index('Zposition')])
                position_from_data[0] = working_trajectory[frame, pos_x]
                position_from_data[1] = working_trajectory[frame, pos_y]
                position_from_data[2] = working_trajectory[frame, pos_z]
            else:
                print('DOES NOT HAVE POSITION')

            rotation_from_data = np.zeros([3])
            if 'Xrotation' in joint_list[1][i_joint]:
                joint_ids = BVH_tools.get_joint_id(joint_list[0], joint_list[1], joint_list[0][i_joint])
                rot_x = int(joint_ids[joint_channels.index('Xrotation')])
                rot_y = int(joint_ids[joint_channels.index('Yrotation')])
                rot_z = int(joint_ids[joint_channels.index('Zrotation')])
                rotation_from_data[0] = working_trajectory[frame, rot_x]
                rotation_from_data[1] = working_trajectory[frame, rot_y]
                rotation_from_data[2] = working_trajectory[frame, rot_z]

                joint_ids = BVH_tools.get_joint_id(joint_list[0], joint_list[1], joint_list[0][i_joint])
                joint_len = np.sqrt(np.power(joint_offset[0], 2)+np.power(joint_offset[1], 2)+np.power(joint_offset[2], 2))
                rotation_position = np.sin(rotation_from_data * np.pi/180) * joint_len
            else:
                print('DOES NOT HAVE ROTATION')

            print('Offset: {} Position(from data): {} Rotation position (from offset): {} Rotation(from data): {}'.format(
                joint_offset, list(position_from_data), list(rotation_position), list(rotation_from_data)))
            final_position = parent_position + joint_offset + position_from_data + rotation_position
            print('Parent:', parent_position)
            print('Diff:  {}'.format(np.linalg.norm(final_position - parent_position)))
            print('Len:   {}'.format(joint_len))
            print('Ratio: {}'.format(np.linalg.norm(final_position - parent_position)/joint_len))
            position_per_joint[joint_name] = final_position
            print('Final position: {}'.format(final_position))
            print()
            ax.scatter(final_position[0], final_position[1], final_position[2], color='blue')
            ax.text(final_position[0], final_position[1], final_position[2], joint_name, color='red')
        break

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()