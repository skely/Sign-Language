import numpy as np


def load_raw(_in_file):
    """
    Loads bvh file as text.
    :param _in_file: path
    :return: list of lines
    """
    with open(_in_file, 'r') as f:
        content = f.readlines()

    headder = -1
    for i, line in enumerate(content):
        if 'MOTION' in line:
                headder = i+1

    return content[:headder], content[headder+2:]


def load_trajectory(_in_file):
    """
    Loads trajectory from BVH file.
    :param _in_file: path
    :return: numpy array [time, features]
    """
    _, _raw_trajectory = load_raw(_in_file)

    _trajectory = []
    for _lin in _raw_trajectory:
        _frame = _lin.strip().split(' ')
        _trajectory.append(_frame)

    _ret_trajectory = np.asarray(_trajectory, dtype=float)
    return _ret_trajectory


def load_raw_header(_in_file):
    """
    Loads header as text.
    :param _in_file: path
    :return: list of lines (inc. formatting chars)
    """
    _raw_header, _ = load_raw(_in_file)
    return _raw_header


def get_joint_list(_in_file):
    """
    Lists all joints and root names. (In order same as trajectory features, not hierarchy)
    :param _in_file: path
    :return: joint name list, channels list, offset
    """
    _header = load_raw_header(_in_file)
    _marker_list = []
    _channel_list = []
    _offset_list = []
    for i, line in enumerate(_header):
        if 'JOINT' in line or 'ROOT' in line:
            _marker_list.append(line.strip().split(' ')[1])
            _channel_list.append(_header[i+3].strip().split(' ')[2:])
            _offset_list.append(_header[i+2].strip().split(' ')[1:])
    return _marker_list, _channel_list, _offset_list


def get_joint_id(_marker_list, _channel_list, _joint_name, _channel_name='all'):
    """
    Gets joint positions for given channel names (or pattern line 'rotation', 'X', ...)
    inputs from get_joint_list
    :param _marker_list: list of joints
    :param _channel_list: list of channels
    :param _joint_name: searched joint name
    :param _channel_name: searched channel name
    :return:
    """
    _ret_list = []
    _channel_counter = 0
    for _i, (_m, _c) in enumerate(zip(_marker_list, _channel_list)):
        _tmp_channel_count = len(_c)
        if _m == _joint_name or _joint_name == '':
            if _channel_name == 'all':
                _ret_list += np.arange(_channel_counter, _channel_counter+_tmp_channel_count).tolist()
                _channel_counter += _tmp_channel_count
            else:
                for _tmp_channel in _c:
                    if _channel_name in _tmp_channel:
                        _ret_list.append(_channel_counter)
                    _channel_counter += 1
        else:
            _channel_counter += _tmp_channel_count
    return _ret_list


def get_joint_name(_marker_list, _channel_list, _joint_id):
    """
    Gets joint name from index. Inputs from get_joint_list
    :param _marker_list: list of joints
    :param _channel_list: list of channels
    :param _joint_id: searched id
    :return: joint name, channel name
    """
    index = 0
    for _i, (_m, _ch) in enumerate(zip(_marker_list, _channel_list)):
        index_new = index + len(_channel_list[_i])
        if index <= _joint_id < index_new:
            channel = _ch[_joint_id - index]
            marker = _m
            break
        index = index_new
    return marker, channel


def get_tree_structure(_raw_header):
    """
    Parse BVH header to tree structure
    :param _raw_header:
    :return: list of joints [joint_name, parent, [children]]
    """


def get_ancester(_joint_name, _tree_structure):
    """
    returns name of parent joint
    :param _joint_name:
    :param _tree_structure:
    :return: joint_name of parent
    """


def get_children(_join_name, _tree_structure):
    """
    returns list of children joints
    :param _join_name:
    :param _tree_structure:
    :return:
    """


def get_all_children(_join_name, _tree_structure):
    """
    returns list of all joints in subtree of the input.
    :param _join_name:
    :param _tree_structure:
    :return:
    """