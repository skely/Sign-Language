import PRJ5_tools
import PRJ4_tools
import numpy as np
import os
import sys
from sklearn import svm
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
    return (np.sum(np.abs(_velocity))/np.size(_velocity,1))/np.size(_velocity,0)


def dominant_hand(_first, _second):
    """

    :param _first:
    :param _second:
    :return:
    """
    return movement_ratio(_first)/movement_ratio(_second)


if __name__ == '__main__':
    if os.path.exists('D:/Znakovka/BVH_reload/'):
        in_path = 'D:/Znakovka/dict_solved/'
        out_path = 'D:/Škola/FAV/PRJ/BC/Output/'
        dict_path = 'D:/Znakovka/Dictionary'
        done_dir_name = 'D:/Znakovka/BVH_reload'
    elif os.path.exists('C:/Znakovka/BVH_reload/'):
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

    all_signs_list = []
    dominations = []
    domination_teacher = []
    done_list = set()
    idx = 0
    result_length = 0
    prev_hand_count = 0
    prev_sign = {'annotation_Filip_bvh_frame': '', 'annotation_Filip': '', 'sign_id': '', 'annotation_flag': '', 'sign_name': '', 'src_pattern': '', 'domination_arm': '', 'hand_count': -1, 'symmetry': -1}
    prevprev_sign = {'annotation_Filip_bvh_frame': '', 'annotation_Filip': '', 'sign_id': '', 'annotation_flag': '', 'sign_name': '', 'src_pattern': '', 'domination_arm': '', 'hand_count': -1, 'symmetry': -1}
    result = []
    for sign in dictionary_takes:
        if sign['sign_name'] in ['tra.']:
            idx += 1
            continue
        if 'hand_count' not in sign.keys():
            sign['hand_count'] = -1
        # if sign['sign_id'] in ['tra.', 'T-pose', 'klapka', 'rest pose', '']:
        #     idx += 1
        #     continue
        # print(sign['sign_id'])
        # print(sign)
        # print(len(dictionary_takes))
        percentage = 100*idx/len(dictionary_takes)
        print('Computing from {} {}... {:0.2f}%'.format(sign['src_pattern'], sign['sign_name'], percentage))
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

        if sign['annotation_Filip_bvh_frame'][0] < 0:
            sign['annotation_Filip_bvh_frame'][0] = 0
        ratio_LH = movement_ratio(traj_LH[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :])
        ratio_LH_arm = movement_ratio(traj_LH_arm[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :])
        ratio_RH = movement_ratio(traj_RH[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :])
        ratio_RH_arm = movement_ratio(traj_RH_arm[sign['annotation_Filip_bvh_frame'][0]:sign['annotation_Filip_bvh_frame'][1], :])
        domination = ratio_LH/ratio_RH
        domination_arm = ratio_LH_arm/ratio_RH_arm
        sign['domination_arm'] = domination_arm


        hand_count = 0
        symmetry = 0
        for item in dictionary_dict:
            if item['sign_id'] == sign['sign_id']:
                sign['hand_count'] = item['hand_count']
                sign['symmetry'] = item['symmetry']
                break
        # if sign['hand_count'] == 0:
        #     sign['hand_count'] = -1

        if prev_sign['src_pattern'] != sign['src_pattern'] and prevprev_sign['src_pattern'] != sign['src_pattern']:
            print('First sign')
            print(sign)
            prevprev_sign = prev_sign
            prev_sign = sign
            prev_hand_count = sign['hand_count']
            continue
        if prevprev_sign['src_pattern'] != sign['src_pattern']:
            # print('Second sign, Hand count = {}, dom = {}'.format(hand_count, domination_arm))
            print('Second sign')
            print(sign)
            prevprev_sign = prev_sign
            prev_sign = sign
            prev_hand_count = sign['hand_count']
            continue
        if sign['sign_id'] == '':
            print('Middle sign, no annotation')
            print(sign)
            prevprev_sign = prev_sign
            prev_sign = sign
            prev_hand_count = -1
            continue
        result_length += 1
        result.append([prevprev_sign, prev_sign, sign, prev_hand_count])
        # dominations.append([domination_arm, prev_dom, prevprev_dom])
        # domination_teacher.append(prev_hand_count)
        idx += 1
        print(prevprev_sign)
        print(prev_sign)
        print(prev_hand_count)
        print(sign)
        prevprev_sign = prev_sign
        prev_sign = sign
        prev_hand_count = sign['hand_count']

    # svm_dom = svm.LinearSVC()
    # svm_dom.fit(dominations, domination_teacher)
    # print(svm_dom)
    # print(svm_dom.coef_)
    # with open(os.path.join(out_path, 'save2.pkl'), 'wb') as f:
    #     pickle.dump(svm_dom, f)
    print(len(result))
    print(result_length)
    with open(os.path.join(out_path, 'trigrams.pkl'), 'wb') as f:
        pickle.dump(result, f)
