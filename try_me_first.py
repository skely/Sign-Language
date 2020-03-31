import bvh2glo_simple
import numpy as np
from lib import SL_dict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    BVH_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_29_c_FR.bvh'
    dictionary_file = '/home/jedle/data/Sign-Language/_source_clean/ultimate_dictionary2.txt'

    # načtení BVH souboru a přepočítání angulárních data na trajektorii v globáních souřadnicích
    BVH_infile = '16_05_20_a_R.bvh'
    joints, trajectory = bvh2glo_simple.calculate(BVH_file)
    frames, joint_id, channels = np.shape(trajectory)

    # Vypsání slovníku
    dictionary = SL_dict.search_take_file(dictionary_file, BVH_file)
    for line in dictionary:
        for tmp_key in line.keys():
            print('{} : {}'.format(tmp_key, line[tmp_key]))
        print('*****')
        if line['sign_name'] == 'tra.':
            print(line['annotation_Filip_bvh_frame'])

    # plot vybraného markeru
    selected_joint = [i for i, j in enumerate(joints) if j == 'RightHand']
    channel_names = ['X', 'Y', 'Z']
    fig, a = plt.subplots(len(channel_names), 1)
    fig.suptitle('trajectory of {}'.format(joints[selected_joint[0]]), fontsize=10)
    fig.tight_layout(pad=3.0)
    for i in range(len(channel_names)):
        a[i].plot(trajectory[1125:1162, selected_joint[0], i])
        a[i].set_xlabel('frame number')
        a[i].set_ylabel('coord {}'.format(channel_names[i]))
    plt.show()