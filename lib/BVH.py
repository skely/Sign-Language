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

    return content[:headder+2], content[headder+2:]


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
    print('obsolete, use get_tree_structure_joint_list instead.')
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


def get_joint_list_from_header(_header):
    """
    Lists all joints and root names. (In order same as trajectory features, not hierarchy)
    :param _in_file: path
    :return: joint name list, channels list, offset
    """
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


def get_tree_structure_joint_list(_raw_header):
    """
    Parse BVH header to tree structure to enhanced joint list (get_joint_list - deprected)
    :param _raw_header:
    :return: joint_list: tuple (names, channels, offset, parent, [children])
    """
    joint_list = get_joint_list_from_header(_raw_header)
    parent_list = []
    joint_stack = ['root']
    for i, line in enumerate(_raw_header):
        if '{' in line:
            joint_name = _raw_header[i-1].strip().split(' ')[1]
            joint_stack.append(joint_name)
            parent_list.append([joint_name, joint_stack[-2]])
        if '}' in line:
            joint_stack.pop()

    parent_ordered_list = []
    child_ordered_list = []
    for joint in joint_list[0]:
        parent = [pair[1] for pair in parent_list if pair[0] == joint][0]
        parent_ordered_list.append(parent)
        children = [pair[0] for pair in parent_list if pair[1] == joint]
        child_ordered_list.append(children)

    joint_list += (parent_ordered_list,)
    joint_list += (child_ordered_list,)

    return joint_list


def get_ancestor(_joint_name, _tree_structure_joint_list):
    """
    returns name of parent joint
    :param _joint_name:
    :param _tree_structure_joint_list: list from get_tree_structure
    :return: joint_name of parent
    """
    tmp = [_tree_structure_joint_list[3][i] for i, a in enumerate(_tree_structure_joint_list[0]) if a == _joint_name][0]
    return tmp


def get_children(_joint_name, _tree_structure_joint_list):
    """
    returns list of children joints
    :param _join_name:
    :param _tree_structure:
    :return:
    """
    tmp = [_tree_structure_joint_list[4][i] for i, a in enumerate(_tree_structure_joint_list[0]) if a == _joint_name][0]
    return tmp


def get_all_children(_joint_name, _tree_structure_joint_list):
    """
    returns list of all joints in subtree of the input.
    :param _join_name:
    :param _tree_structure:
    :return:
    """
    tmp_name = _joint_name
    stack = [tmp_name]
    child_list = []
    while stack != []:
        tmp_name = stack.pop(0)
        if tmp_name != 'Site':
            stack = stack + get_children(tmp_name, _tree_structure_joint_list)
        if tmp_name != _joint_name:
            child_list.append(tmp_name)
    return child_list


def get_all_ancestors(_joint_name, _tree_structure_joint_list):
    """
    returns list of all ancestors first is closest
    :param _joint_name:
    :param _tree_structure_joint_list:
    :return:
    """
    tmp_name = _joint_name
    stack = []
    while tmp_name != 'root':
        stack.append(get_ancestor(tmp_name, _tree_structure_joint_list))
        tmp_name = stack[-1]
    return stack


def generate_BVH(_trajectory, _BVH_output_file, _template_BVH_file, channels='rotation', zero_shift=True):
    """
    Saves angular trajectory as BVH file (using template BVH for header creation)
    :param _trajectory:
    :param _BVH_output_file:
    :param _template_BVH_file:
    :param channels:
    :return:
    """
    template_header, _ = load_raw(_template_BVH_file)
    template_trajectories = load_trajectory(_template_BVH_file)

    trajectory_length, feature_length = np.shape(_trajectory)
    write_stream = template_header[:-2]
    write_stream.append('Frames: {}\n'.format(trajectory_length))
    write_stream.append(template_header[-1])  # fps

    expected_number_of_channels = np.size(template_trajectories, 1)

    m, c, _ = get_joint_list(_template_BVH_file)
    channel_ids = get_joint_id(m, c, '', _channel_name=channels)

    for i in range(trajectory_length):
        new_line = ''
        iter_pick = 0
        for j in range(expected_number_of_channels):
            if j in channel_ids:
                tmp = _trajectory[i, iter_pick]
                iter_pick += 1
            else:
                if j < 3 and zero_shift:
                    tmp = 0  # shift root position to 0,0,0
                else:
                    tmp = np.mean(template_trajectories[:, j])

            if j < expected_number_of_channels - 1:
                new_line += '{:.6f} '.format(tmp)
            else:
                new_line += '{:.6f}\n'.format(tmp)
        write_stream.append(new_line)

    with open(_BVH_output_file, 'w+') as f:
        f.writelines(write_stream)