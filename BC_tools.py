import PRJ5_tools
import PRJ4_tools
import numpy as np
import os
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

    # infile = '/home/jedle/data/Sign-Language/projevy_pocasi_02_solved_body.bvh'
    in_path = 'D:/Znakovka/dict_solved/'
    # in_path = 'C:/Znakovka/dict_solved/'

    # outfile = '/home/jedle/data/Sign-Language/tst2.bvh'
    out_path = 'D:/Škola/FAV/PRJ/BC/Output/'
    # out_path = 'C:/Škola/FAV/PRJ/PRJ5/Output/'

    dict_path = 'D:/Škola/FAV/PRJ/BC/'
    # dict_path = 'C:/Škola/FAV/PRJ/PRJ5/'

    done_dir_name = 'D:/Znakovka/BVH_reload'
    # done_dir_name = 'C:/Znakovka/BVH_reload'
    done_file_list = os.listdir(done_dir_name)

    # file_dict = os.path.join(dict_path, 'pocasi_slovnik9.txt')
    file_dict = os.path.join(dict_path, 'new_dictionary.txt')
    file_dict_dict = os.path.join(dict_path, 'dictionary_dict_v3_edited.txt')
    dictionary = PRJ5_tools.dict_read(file_dict)
    dictionary_dict = PRJ5_tools.dict_read(file_dict_dict)

    out_table1 = os.path.join(out_path, 'table1.csv')
    out_table2 = os.path.join(out_path, 'table2.csv')
    in_file = os.path.join(in_path, 'ciselne_hodnoty_04_solved_body_R.bvh')  # predlozky_spojky_02_solved_body
    # numbers_file = os.path.join(in_path, 'ciselne_hodnoty_04_solved_body.bvh')

    raw_data = PRJ5_tools.read_file(in_file)
    trajectory_data, header = PRJ5_tools.load_data(raw_data)

    header_dict = PRJ5_tools.explain_header(header)

    idx_RH = PRJ5_tools.get_children_index('RightArm', header)
    idx_LH = PRJ5_tools.get_children_index('LeftArm', header)
    traj_RH = trajectory_data[:, idx_RH[0]:idx_RH[-1]]
    traj_LH = trajectory_data[:, idx_LH[0]:idx_LH[-1]]
    print(np.shape(traj_RH))
    print(np.shape(traj_LH))
    # print(PRJ5_tools.get_name(idx_RH[0], header))
    # print(PRJ5_tools.get_name(idx_RH[-1], header))
    # print(PRJ5_tools.get_name(idx_LH[0], header))
    # print(PRJ5_tools.get_name(idx_LH[-1], header))
    # movement_ratio(traj_RH)
    # movement_ratio(traj_LH)
    # plt.figure()
    # print(PRJ5_tools.get_name(idx_LH[0]+71, header))
    # plt.plot(trajectory_data[:,idx_RH[0]+131])
    # plt.plot(traj_LH[:,71])
    # plt.show()
    # print((dominant_hand(traj_LH[1000:2000,:], traj_RH[1000:2000,:])))


    # # Vizualizace trajektorie a vzdáleností (od n_5) markeru daných znaků
    startnumber = 41  # 40 for n_0
    number_of_signs_in_take = 5  # 21 for ciselne_hodnoty_04_solved_body


    # for sign in range(number_of_signs_in_take):
    #     sign_idx = sign + startnumber
    #     print(dictionary[sign_idx]['bvh_boundaries'])

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
    # dict_boundaries = sorted(dict_boundaries, key=itemgetter('sign_id', 'bvh_source'))
    print(dict_boundaries)

    with open(out_table2, 'w') as outfile:
        idx = 0
        # print(len(dict_boundaries))
        outfile.write('bvh_source,sign_id,ratio_LH,ratio_RH,domination,ratio_LH_arm,ratio_RH_arm,domination_arm,hand_count,symmetry\n')
        for sign in dict_boundaries:
            if sign['sign_id'] in ['tra.', 'T-pose', 'klapka', 'rest pose']:
                idx += 1
                continue
            # if sign['bvh_source'] != '17_04_18_b_R.bvh':
            #     continue
            percentage = 100*idx/len(dict_boundaries)
            print('Computing from {} {}... {:0.2f}%'.format(sign['bvh_source'], sign['sign_id'], percentage))
            source = os.path.join(done_dir_name, sign['bvh_source'])
            trajectory, header = PRJ5_tools.load_data(PRJ5_tools.read_file(source))

            idx_RH = PRJ5_tools.get_children_index('RightShoulder', header)
            idx_RH_arm = sorted(list(set(idx_RH)-set(PRJ5_tools.get_children_index('RightHand', header))))
            idx_LH = PRJ5_tools.get_children_index('LeftShoulder', header)
            idx_LH_arm = sorted(list(set(idx_LH)-set(PRJ5_tools.get_children_index('LeftHand', header))))
            traj_RH = trajectory[:, idx_RH[0]:idx_RH[-1]]
            # print(sorted(idx_RH_arm))
            # print(type(idx_RH_arm))
            # print(type(idx_RH))
            traj_RH_arm = trajectory[:, idx_RH_arm[0]:idx_RH_arm[-1]]
            traj_LH = trajectory[:, idx_LH[0]:idx_LH[-1]]
            traj_LH_arm = trajectory[:, idx_LH_arm[0]:idx_LH_arm[-1]]
            # print(np.shape(traj_RH_arm))

            # print(np.shape(traj_LH[sign['bvh_boundaries'][0]:sign['bvh_boundaries'][1], :]))
            # print((sign['bvh_boundaries']))
            if sign['bvh_boundaries'][0] < 0:
                sign['bvh_boundaries'][0] = 0
            ratio_LH = movement_ratio(traj_LH[sign['bvh_boundaries'][0]:sign['bvh_boundaries'][1], :])
            ratio_LH_arm = movement_ratio(traj_LH_arm[sign['bvh_boundaries'][0]:sign['bvh_boundaries'][1], :])
            ratio_RH = movement_ratio(traj_RH[sign['bvh_boundaries'][0]:sign['bvh_boundaries'][1], :])
            ratio_RH_arm = movement_ratio(traj_RH_arm[sign['bvh_boundaries'][0]:sign['bvh_boundaries'][1], :])
            domination = ratio_LH/ratio_RH
            domination_arm = ratio_LH_arm/ratio_RH_arm
            hand_count = 0
            symmetry = 0
            for item in dictionary_dict:
                if item['sign_id'] == sign['sign_id']:
                    hand_count = item['hand_count']
                    symmetry = item['symmetry']
                    break
            outfile.write('{},{},{},{},{},{},{},{},{},{}\n'.format(sign['bvh_source'], sign['sign_id'], ratio_LH, ratio_RH, domination, ratio_LH_arm, ratio_RH_arm, domination_arm, hand_count, symmetry))
            idx += 1
            # if idx > 5:
            #     break
