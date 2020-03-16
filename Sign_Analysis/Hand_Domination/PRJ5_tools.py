import os
import numpy as np
import PRJ4_tools
import json
import fastdtw
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from operator import itemgetter


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
    _idx = 0
    while _idx < len(_data_header):
        if _data_header[_idx].startswith('ROOT') or _data_header[_idx].startswith('JOINT'):
            _marker_info = {}
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
            _marker_info['joints'] = _joints
            _marker_info['children'] = _children
            _header_dict.append(_marker_info)
        _idx += 1
    return _header_dict


def same_headers(_path_to_data):
    """
    returns True if all files in path have same header
    :param _path_to_data: directory
    :return: boolean
    """
    _heads = []
    _result = True
    for _i, _file in enumerate(os.listdir(_path_to_data)):
        if _file.endswith('.bvh'):
            _raw = read_file(os.path.join(_path_to_data, _file))
            # print(raw)
            _, _header = load_data(_raw)
            _heads.append([_file, _header[:len(_header) - 2]])
    for _i, file_i in enumerate(_heads):
        for _j, file_j in enumerate(_heads):
            if _i > _j:
                if file_j[1] == file_i[1]:
                    pass
                else:
                    print(file_i[0], 'and', file_j[0], 'not same')
                    _result = False
    return _result


def center_data(_data):
    """
    Centers data
    :param _data: trajectory data
    :return: edited trajectory data
    """
    _data_edited = _data.copy()
    _data_edited[:, :3] = 0
    # _data_edited[:, 0] = 50
    return _data_edited


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


def get_name(_index, _header):
    """
    :param _index: index of marker
    :param _header: header
    :return: string
    """
    _marker_dict = explain_header(_header)
    _real_idx = 0
    for _marker in _marker_dict:
        _index -= _marker['ch_count']
        if _index < 0:
            if _marker['ch_count'] <= 0:
                return -1
            else:
                return _marker['m_name']
    return None


def get_static_position(_data):
    """
    :param _data: trajectory data
    :return: vector len(_data)
    """
    return _data[0, :]


def remove_fingers(_data, _header):
    """
    Replace hands with static position
    :param _data: trajectory data
    :param _header: header
    :return: edited trajectory data
    """
    _data_edited = _data.copy()
    _right_hand = get_children_index('RightWrist', _header)
    _left_hand = get_children_index('LeftWrist', _header)
    # _data_edited[:, _right_hand[0]:_right_hand[-1]] = get_static_position(_data_edited[:, _right_hand[0]:_right_hand[-1]])
    # _data_edited[:, _left_hand[0]:_left_hand[-1]] = get_static_position(_data_edited[:, _left_hand[0]:_left_hand[-1]])
    _data_edited[:, _right_hand[0]:_right_hand[-1]] = 0
    _data_edited[:, _left_hand[0]:_left_hand[-1]] = 0
    return _data_edited


def dict_read(_dict_file):
    """
    json load file tool
    :param _dict_file:
    :return:
    """
    with open(_dict_file, 'r') as _file:
        _tmp_dict = json.load(_file)
    return _tmp_dict


def find_restpose(_raw_data, _boundaries, _print=False):
    """

    :param _raw_data:
    :param _boundaries:
    :param _print:
    :return:
    """
    if _print:
        print('Boundaries: {}'.format(_boundaries))
    _trajectory_data, _header = load_data(_raw_data)
    _boundaries_fingers = get_index('RightThumbBall', _header)
    _boundaries_fingers.extend(np.array(get_index('RightIndex1', _header)))
    _boundaries_fingers.extend(np.array(get_index('RightMiddle1', _header)))
    _boundaries_fingers.extend(np.array(get_index('RightRing1', _header)))
    _boundaries_fingers.extend(np.array(get_index('RightLittle1', _header)))
    _boundaries_fingers.extend(np.array(get_index('LeftThumbBall', _header)))
    _boundaries_fingers.extend(np.array(get_index('LeftIndex1', _header)))
    _boundaries_fingers.extend(np.array(get_index('LeftMiddle1', _header)))
    _boundaries_fingers.extend(np.array(get_index('LeftRing1', _header)))
    _boundaries_fingers.extend(np.array(get_index('LeftLittle1', _header)))
    _average_trajectories = np.zeros(np.size(_trajectory_data[_boundaries[0]:_boundaries[1]], 0))
    for _marker in range(np.size(_trajectory_data, 1)):
        _trajectory_data[:, _marker] = savgol_filter(_trajectory_data[:, _marker], 31, 7)
    for _marker_index in _boundaries_fingers:
        _average_trajectories += np.abs(_trajectory_data[_boundaries[0]:_boundaries[1], _marker_index])
    _average_trajectories = _average_trajectories/len(_boundaries_fingers)
    _expansion = np.zeros((len(_average_trajectories), 2))
    _average_trajectories = np.concatenate((np.expand_dims(_average_trajectories, axis=1), _expansion), axis=1)
    _velocity, _ = PRJ4_tools.sign_velocity_acceleration(_average_trajectories)
    _threshold = np.sum(_velocity[:20])
    _threshold += np.sum(_velocity[len(_velocity)-20:len(_velocity)])
    _threshold = _threshold/40
    _threshold += np.amax(np.abs([(_velocity[:20], _velocity[len(_velocity)-20:len(_velocity)])]))
    _left_index = 0
    _right_index = len(_velocity)
    if _print:
        plt.plot(_velocity)
        plt.title('Velocity sum')
        print('Threshold: {}'.format(_threshold))
    for _i in range(len(_velocity)):
        if _velocity[_i] > _threshold:
            _left_index = _i
            break
    for _i in range(len(_velocity)):
        if _velocity[len(_velocity)-_i-1] > _threshold:
            _right_index = len(_velocity)-_i-1
            break
    if _print:
        print('Right index: {}'.format(_left_index))
        print('Left index: {}'.format(_right_index))
        plt.show()
    return [_left_index+_boundaries[0], _right_index+_boundaries[0]]


def fourier_average(_data, _max_frequency=120.):
    """

    :param _data:
    :param _max_frequency:
    :return:
    """
    for _traj in _data:
        _fty = np.fft.fft(_data)
        _ftx = np.arange(0., _max_frequency, _max_frequency / (len(_fty) - 1))
    return


def loadd(_file: object, raw: object = False) -> object:
    """
    loads TRC file (known issues: metadata read - \n \t removal issues
    :param _file: path
    :param raw: returns raw header
    :return: np.array data, list header
    """
    with open(_file, 'r') as f:
        content = f.readlines()

    _raw_data = content[5:]
    data_list = []
    for i, _line in enumerate(_raw_data):
        clear_line = _line.split('\t')
        clear_line[-1] = clear_line[-1][:-2]
        np_line = np.zeros((len(clear_line[2:]), ))
        for j, number in enumerate(clear_line[2:]):
            if number != '':
                # print('all markers are not present file: {}, (frame: {})'.format(_file, i))
                # number = float('inf')
                number = float(number)
                np_line[j] = number
            else:
                continue
        # print(np_line)
        data_list.append(np_line)

    return data_list


if __name__ == '__main__':

    # infile = '/home/jedle/data/Sign-Language/projevy_pocasi_02_solved_body.bvh'
    in_path = 'D:/Znakovka/dict_solved/'
    # in_path = 'C:/Znakovka/dict_solved/'

    # outfile = '/home/jedle/data/Sign-Language/tst2.bvh'
    out_path = 'D:/Škola/FAV/PRJ/PRJ5/Output/'
    # out_path = 'C:/Škola/FAV/PRJ/PRJ5/Output/'

    dict_path = 'D:/Škola/FAV/PRJ/PRJ5/'
    # dict_path = 'C:/Škola/FAV/PRJ/PRJ5/'

    done_dir_name = 'D:/Znakovka/BVH_reload'
    done_file_list = os.listdir(done_dir_name)

    # file_dict = os.path.join(dict_path, 'pocasi_slovnik9.txt')
    file_dict = os.path.join(dict_path, 'new_dictionary.txt')
    dictionary = dict_read(file_dict)

    out_table1 = os.path.join(out_path, 'table1.csv')
    in_file = os.path.join(in_path, 'ciselne_hodnoty_04_solved_body.bvh')  # predlozky_spojky_02_solved_body
    # numbers_file = os.path.join(in_path, 'ciselne_hodnoty_04_solved_body.bvh')

    raw_data = read_file(in_file)
    trajectory_data, header = load_data(raw_data)

    header_dict = explain_header(header)

    # # Vizualizace trajektorie a vzdáleností (od n_5) markeru daných znaků
    startnumber = 41  # 40 for n_0
    number_of_signs_in_take = 5  # 21 for ciselne_hodnoty_04_solved_body
    # wanted_marker = get_index('RightIndex1', header)
    # figure, (ax1, ax2, ax3) = plt.subplots(3)
    # figure.suptitle(get_name(wanted_marker[0], header), fontsize=16)
    # dist = [0]
    # for sign in range(number_of_signs_in_take):
    #     sign_idx = sign+startnumber
    #     boundaries = dictionary[sign_idx]['bvh_boundaries']
    #     boundaries = list(map(int, boundaries))
    #     print('Sign: {} Boundaries: {}'.format(dictionary[sign_idx]['sign_id'], boundaries))
    #     ax1.plot(trajectory_data[boundaries[0]:boundaries[1], wanted_marker[0]], label=dictionary[sign_idx]['sign_id'])
    #     ax2.plot(trajectory_data[boundaries[0]:boundaries[1], wanted_marker[1]])
    #     ax3.plot(trajectory_data[boundaries[0]:boundaries[1], wanted_marker[2]])
    #     ax1.legend(loc=1)
    #     dist_temp, _ = fastdtw.dtw(trajectory_data[2036:2303, wanted_marker[0]:wanted_marker[2]], trajectory_data[boundaries[0]:boundaries[1], wanted_marker[0]:wanted_marker[2]])
    #     dist.append(dist_temp)
    #     # print(dist)
    # plt.figure()
    # plt.plot(dist)
    # plt.title('{}\nDTW: n_5 VS {}-{}'.format(get_name(wanted_marker[0], header), dictionary[startnumber]['sign_id'], dictionary[startnumber+number_of_signs_in_take-1]['sign_id']))
    # plt.show()
    wanted_signs = ['n_1', 'n_2', 'n_3', 'n_4', 'n_5']
    # marker_parent = 'RightForeArm'
    marker_parent = 'RightWrist'
    marker_parent_v2 = 'RightHand'
    # print(wanted_markers)
    # print(get_name(wanted_markers[0], header))
    # print(get_name(wanted_markers[-1], header))

    for sign in range(number_of_signs_in_take):
        sign_idx = sign + startnumber
        print(dictionary[sign_idx]['bvh_boundaries'])

    dictionary[41]['bvh_boundaries'] = [1096, 1118]
    dictionary[42]['bvh_boundaries'] = [1361, 1380]
    dictionary[43]['bvh_boundaries'] = [1621, 1641]
    dictionary[44]['bvh_boundaries'] = [1881, 1911]
    dictionary[45]['bvh_boundaries'] = [2154, 2177]

    dict_boundaries = []
    done_list = set()
    for line in done_file_list:
        if '.bvh' in line:
            # print(line)
            one_file_list = []
            time_stmps = []
            for item in dictionary:
                if 'bvh_continuous' in item.keys():
                    for cont_item in item['bvh_continuous']:
                        if cont_item[0] == line:
                            tmp = cont_item.copy()
                            tmp.append(item['sign_id'])
                            one_file_list.append(tmp)
            one_file_list.sort()
            for item in one_file_list:
                if item[3] in wanted_signs:
                    dict_boundaries_tmp = {}
                    dict_boundaries_tmp['sign_id'] = item[3]
                    dict_boundaries_tmp['bvh_source'] = item[0]
                    if dict_boundaries_tmp['sign_id'] == 'n_4' and dict_boundaries_tmp['bvh_source'] == '17_02_15_b_FR.bvh':
                        dict_boundaries_tmp['bvh_boundaries'] = [5768, 5777]
                    elif dict_boundaries_tmp['sign_id'] == 'n_4' and dict_boundaries_tmp['bvh_source'] == '17_03_15_b_FR.bvh':
                        dict_boundaries_tmp['bvh_boundaries'] = [3616, 3630]
                    else:
                        dict_boundaries_tmp['bvh_boundaries'] = [item[1], item[2]]
                    dict_boundaries.append(dict_boundaries_tmp)
                # print(item), i['bvh_source']
    dict_boundaries = sorted(dict_boundaries, key=itemgetter('sign_id', 'bvh_source'))
    print(dict_boundaries)

    for sign in range(number_of_signs_in_take):
        sign_idx = sign + startnumber
        boundaries = dictionary[sign_idx]['bvh_boundaries']
        boundaries = list(map(int, boundaries))
        print('Dictionary: {} Boundaries: {}'.format(dictionary[sign_idx]['sign_id'], boundaries))
        dict_boundaries_tmp = {}
        dict_boundaries_tmp['sign_id'] = 'd{}'.format(dictionary[sign_idx]['sign_id'])
        dict_boundaries_tmp['bvh_source'] = dictionary[sign_idx]['bvh_source']
        dict_boundaries_tmp['bvh_boundaries'] = boundaries
        dict_boundaries.append(dict_boundaries_tmp)

    print(dict_boundaries)

    with open(out_table1, 'w') as outfile:
        for itemi in (dict_boundaries):
            for itemj in (dict_boundaries):

                print('Computing {} {}...'.format(itemi['sign_id'], itemj['sign_id']))
                source1 = os.path.join(done_dir_name, itemi['bvh_source'])
                source2 = os.path.join(done_dir_name, itemj['bvh_source'])
                trajectory1, header1 = load_data(read_file(source1))
                trajectory2, header2 = load_data(read_file(source2))
                wanted_markers1 = get_children_index(marker_parent, header1)
                # print(len(wanted_markers1))
                if len(wanted_markers1) == 0:
                    wanted_markers1 = get_children_index(marker_parent_v2, header1)
                wanted_markers2 = get_children_index(marker_parent, header2)
                # print(len(wanted_markers2))
                if len(wanted_markers2) == 0:
                    wanted_markers2 = get_children_index(marker_parent_v2, header2)
                trajectory1 = remove_fingers(trajectory1, header)
                trajectory2 = remove_fingers(trajectory2, header)

                print(len(wanted_markers1))
                print(len(wanted_markers2))

                if len(wanted_markers1) != len(wanted_markers1):
                    break
                # dist, path = fastdtw.dtw(trajectory1[itemi['bvh_boundaries'][0]:itemi['bvh_boundaries'][1], wanted_markers1[0]:wanted_markers1[-1]], trajectory2[itemj['bvh_boundaries'][0]:itemj['bvh_boundaries'][1], wanted_markers2[0]:wanted_markers2[-1]])
                dist, path = fastdtw.dtw(trajectory1[itemi['bvh_boundaries'][0]:itemi['bvh_boundaries'][1], :], trajectory2[itemj['bvh_boundaries'][0]:itemj['bvh_boundaries'][1], :])
                dist = dist/len(path)
                # dist = dist / len(dictionary)
                # dist = dist / len(wanted_markers1)
                dist = dist / (len(dictionary)-150)
                outfile.write('{},'.format(dist))
            outfile.write('\n')
            print('Line written')
