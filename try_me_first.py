import bvh2glo_simple
import numpy as np
from lib import SL_dict
import simple_run

if __name__ == "__main__":
    BVH_file = '/home/jedle/data/Sign-Language/_source_clean/bvh/16_05_29_c_FR.bvh'
    dictionary_file = '/home/jedle/data/Sign-Language/_source_clean/ultimate_dictionary2.txt'

    # BVH_infile = '16_05_20_a_R.bvh'
    # joints, trajectory = bvh2glo_simple.calculate(BVH_file)
    #
    # print(joints)
    # frames, joints, channels = np.shape(trajectory)

    dictionary = SL_dict.search_take_file(dictionary_file, BVH_file)
    for line in dictionary:
        # print(line)
        # print(line['annotation_Filip'], line['sign_id'], line['sign_name'])
        if line['sign_name'] == 'tra.':
            print(line['annotation_Filip_bvh_frame'])
