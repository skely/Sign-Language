import numpy as np


def load_raw(_in_file):
    with open(_in_file, 'r') as f:
        content = f.readlines()

    headder = -1
    for i, line in enumerate(content):
        if 'MOTION' in line:
                headder = i+1

    return content[:headder], content[headder+2:]


def load_trajectory(_in_file):
    _, _raw_trajectory = load_raw(_in_file)

    _trajectory = []
    for _lin in _raw_trajectory:
        _frame = _lin.strip().split(' ')
        _trajectory.append(_frame)

    _ret_trajectory = np.asarray(_trajectory, dtype=float)
    return _ret_trajectory


def load_raw_header(_in_file):
    _raw_header, _ = load_raw(_in_file)
    return _raw_header


def get_joint_list(_in_file):
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
    index = 0
    for _i, (_m, _ch) in enumerate(zip(_marker_list, _channel_list)):
        index_new = index + len(_channel_list[_i])
        if index <= _joint_id < index_new:
            channel = _ch[_joint_id - index]
            marker = _m
            break
        index = index_new
    return marker, channel