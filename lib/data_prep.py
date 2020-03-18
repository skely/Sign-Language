import os
from lib import BVH, SL_dict
import sys
import numpy as np


def mine_sign_trajectories(_bvh_src_path, _dict_take_file, _surroundings, _sign_id='', _sign_name='tra.', _channels=[], _verbose=True):
    """
    returns list of numpy arrays containing trajectory. works both on signs (_sign_name not used) and transitions (sign_id='', _sign_name='tra.)
    :param _bvh_src_path:
    :param _dict_take_file:
    :param _surroundings:
    :param _sign_id:
    :param _sign_name:
    :return:
    """
    if type(_surroundings) == int:
        _surrs = [_surroundings, _surroundings]
    else:
        _surrs = _surroundings

    ret = SL_dict.search_take_sign(_dict_take_file, _sign_id)
    sign_meta_list = []
    if _sign_id != '':
        sign_meta_list = ret
    else:
        for r in ret:
            if r['sign_name'] == _sign_name:
                sign_meta_list.append(r)

    pattern_list = set()
    for item in sign_meta_list:
        pattern_list.add(item['src_pattern'])

    bvh_list = os.listdir(_bvh_src_path)
    files_list = []
    for bvh_file in bvh_list:
        for pattern in pattern_list:
            if pattern in bvh_file:
                files_list.append(bvh_file)
    if _verbose:
        if _sign_id == '' and _sign_name == 'tra.':
            print('searching for transitions.'.format(_sign_id))
        else:
            print('searching for sign: [{}].'.format(_sign_id))
        print('{} occurances found.\nLoading data:'.format(len(sign_meta_list)))
    ret_list = []
    tot = len(files_list)
    for i, tmp_file in enumerate(files_list):
        tmp_trajectory = BVH.load_trajectory(os.path.join(_bvh_src_path, tmp_file))
        if _channels == []:
            _channels = np.arange(np.size(tmp_trajectory, 1))
        for item in sign_meta_list:
            if item['src_pattern'] in tmp_file:
                beg_frame, end_frame = item['annotation_Filip_bvh_frame']
                tmp_akt_traj = tmp_trajectory[beg_frame - _surrs[0]:end_frame + _surrs[1], _channels]
                ret_list.append(tmp_akt_traj)
        if _verbose:
            sys.stdout.write('\rprocessing... {:.2f}% done.'.format(100*(i+1)/tot))
    if _verbose:
        sys.stdout.write('\rdone.\n')
    return ret_list, sign_meta_list


def resample_trajectory(_trajectory, _desired_length):
    """
    linear resample trajectory[time, features] to _desired_length
    :param _trajectory:
    :param _desired_length:
    :return:
    """
    new_trajectory = np.zeros((_desired_length, np.size(_trajectory, 1)))
    for j in range(np.size(_trajectory, 1)):
        xp = np.arange(len(_trajectory[:, j]))
        xvals = np.linspace(0, len(_trajectory[:, j]), _desired_length)
        yinter = np.interp(xvals, xp, _trajectory[:, j])
        new_trajectory[:, j] = yinter
    return new_trajectory


def normalize(_data, _minmax=np.array([-360, 360])):
    """
    normalize features in data [item, time, features)
    :param _data:
    :param _minmax:
    :return:
    """
    _new_data = _data.copy()
    for i in range(np.size(_data, 0)):
        for ch in range(np.size(_data, 2)):
            _new_data[i, :, ch] = (_data[i, :, ch] - _minmax[0]) / (_minmax[1] - _minmax[0])
    return _new_data


def shuffle(_data, _seed):
    """
    Randomize data order.
    :param _data: np.array dim=3 [sample, time, features]
    :param _seed: random seed
    :return: randomized data
    """
    np.random.seed(_seed)
    randomize_vector = np.arange(np.size(_data, 0))
    np.random.shuffle(randomize_vector)
    randomized_data = _data[randomize_vector, :, :]
    return randomized_data


def sign_synthesis(_sign_1, _sign_2, _gap_length, _type='cubic'):
    """
    Synthesizes linear interpolation of movement
    :param _sign_1: trajectory frames1 X markers
    :param _sign_2: trajectory frames2 X markers
    :param _gap_length: number of frames (int)
    :param _type: type of interpolation: 'linear', 'cubic'
    :return: resulting_trajectory framesR X markers (dim = frames1+frames2+_gap_length X markers)
    """
    if _type == 'linear':
        _sign_1_t = _sign_1[-1:, :]
        _sign_2_t = _sign_2[0:1, :]
        inter = np.zeros((_gap_length, np.size(_sign_1_t, 1)))
        for traj in range(np.size(_sign_1_t, 1)):
            for frame in range(_gap_length):
                inter[frame, traj] = -1*(_sign_1_t[0, traj]-_sign_2_t[0, traj])/_gap_length*frame+_sign_1_t[0, traj]
        res = inter
    elif _type == 'cubic':
        if np.size(_sign_1, 0) < 2:
            res = -1
        else:
            _sign_1_t = _sign_1[-2:, :]
            _sign_2_t = _sign_2[0:2, :]
            inter = np.zeros((_gap_length, np.size(_sign_1_t, 1)))

            for traj in range(np.size(_sign_1_t, 1)):
                y1 = _sign_1_t[1, traj]
                y2 = _sign_2_t[0, traj]
                k1 = _sign_1_t[1, traj]-_sign_1_t[0, traj]
                k2 = _sign_2_t[1, traj]-_sign_2_t[0, traj]
                a = k1*_gap_length - (y2-y1)
                b = -k2*_gap_length + (y2-y1)

                for frame in range(_gap_length):
                    t = frame/_gap_length
                    inter[frame, traj] = (1-t)*y1 + t*y2 + t*(1-t)*((1-t)*a + t*b)
            res = inter
    else:
        res = -1
    return res[1:, :]