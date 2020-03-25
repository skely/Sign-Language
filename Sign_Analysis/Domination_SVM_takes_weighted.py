import PRJ5_tools
import PRJ4_tools
import numpy as np
import os
import sys
from sklearn import svm
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from operator import itemgetter


def movement_ratio(_data):
    """

    :param _data:
    :return:
    """
    _velocity, _ = PRJ4_tools.sign_velocity_acceleration(_data)
    # plt.figure()
    # plt.plot(_velocity)
    # print(len(_velocity))
    # print(np.shape(_velocity))
    return np.sum(np.abs(_velocity))/np.size(_velocity,1)


def dominant_hand(_first, _second):
    """

    :param _first:
    :param _second:
    :return:
    """
    return movement_ratio(_first)/movement_ratio(_second)


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

    bvh_list = os.listdir(done_dir_name)
    dictionary_takes = PRJ5_tools.dict_read(os.path.join(dict_path, 'dictionary_takes_v3.txt'))
    dictionary_dict = PRJ5_tools.dict_read(os.path.join(dict_path, 'dictionary_dict_v3_edited.txt'))

    weighted = False

    dominations = []
    domination_teacher = []
    idx = 0
    count = 0

    for sign in dictionary_takes:
        if sign['sign_id'] in ['tra.', 'T-pose', 'klapka', 'rest pose', '']:
            idx += 1
            continue
        if sign['annotation_flag'] == '-1':
            idx += 1
            continue
        percentage = 100*idx/len(dictionary_takes)
        print('Computing from {} {}... {:0.2f}%'.format(sign['src_pattern'], sign['sign_id'], percentage))
        source = [f for f in bvh_list if sign['src_pattern'] in f][0]
        trajectory, header = PRJ5_tools.load_data(PRJ5_tools.read_file(os.path.join(done_dir_name, source)))

        idx_RH = PRJ5_tools.get_children_index('RightShoulder', header)
        idx_RH_arm = sorted(list(set(idx_RH) - set(PRJ5_tools.get_children_index('RightHand', header))))
        idx_LH = PRJ5_tools.get_children_index('LeftShoulder', header)
        idx_LH_arm = sorted(list(set(idx_LH) - set(PRJ5_tools.get_children_index('LeftHand', header))))

        traj_RH = trajectory[:, idx_RH[0]:idx_RH[-1]]
        traj_RH_arm = trajectory[:, idx_RH_arm[0]:idx_RH_arm[-1]]
        traj_LH = trajectory[:, idx_LH[0]:idx_LH[-1]]
        traj_LH_arm = trajectory[:, idx_LH_arm[0]:idx_LH_arm[-1]]

        if weighted:
            weights = np.zeros(np.size(traj_RH, 1))
        marker_list = []
        for i, idx_traj in enumerate(idx_RH):
            if weighted:
                weights[i-1] = PRJ5_tools.get_offset_sum(idx_traj, header)
            marker_list.append(PRJ5_tools.get_name(idx_traj, header))

        if sign['annotation_Filip_bvh_frame'][0] < 0:
            sign['annotation_Filip_bvh_frame'][0] = 0
        if weighted:
            ratio_LH = movement_ratio(traj_LH[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :]*weights)/len(Counter(marker_list).keys())
            ratio_RH = movement_ratio(traj_RH[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :]*weights)/len(Counter(marker_list).keys())
        else:
            ratio_LH = movement_ratio(traj_LH_arm[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :]/len(Counter(marker_list).keys()))
            ratio_RH = movement_ratio(traj_RH_arm[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :]/len(Counter(marker_list).keys()))
        # domination_arm = ratio_LH_arm/ratio_RH_arm
        domination = ratio_LH/ratio_RH

        hand_count = 0
        symmetry = 0
        for item in dictionary_dict:
            if item['sign_id'] == sign['sign_id']:
                hand_count = item['hand_count']
                symmetry = item['symmetry']
                break
        if hand_count == 0:
            if sign['sign_id'] == 'snezi-nesnezi-snezi':
                hand_count = 2
            else:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                hand_count = -1
                continue
        print(domination)
        print(hand_count)
        dominations.append([domination])
        domination_teacher.append(hand_count)
        idx += 1
        count += 1

    print(count)
    print(len(dominations))
    print(len(domination_teacher))
    print((dominations))
    print((domination_teacher))
    # svm_dom = svm.LinearSVC()
    # svm_dom.fit(dominations, domination_teacher)
    # print(svm_dom)
    # print(svm_dom.coef_)
    with open(os.path.join(out_path, 'uni_takes.pkl'), 'wb') as f:
        pickle.dump([dominations, domination_teacher], f)
