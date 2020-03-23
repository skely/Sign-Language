from lib import SL_dict
import os
import json
import numpy as np


if __name__ == '__main__':
    work_path = '/home/jedle/data/Sign-Language/_source_clean/'
    slovnik = os.path.join(work_path, 'dictionary_dict_v3.txt')
    slovnik_takes = os.path.join(work_path, 'dictionary_takes_v3.txt')

    new_dict_file = os.path.join(work_path, 'ultimate_dictionary.txt')

    exceptions = ['predlozky_a_spojky_01', 'ostatni_03']
    exceptions_takes = ['17_03_25_a', '16_11_11_b', '16_10_25_a', ]

    new_dictionary = SL_dict.read_raw(slovnik)
    take_dictionary = SL_dict.read_raw(slovnik_takes)

    for item in new_dictionary:
        item['src_pattern'] = item['src_vid'][:-4]
        if any(exception in item['src_pattern'] for exception in exceptions):
            item['src_video'] = item['src_pattern'][:-5] + item['src_pattern'][-4:] + '.avi'
            item['src_mocap'] = item['src_pattern'][:-2] + '_solved_body_R.bvh'
        else:
            item['src_video'] = item['src_pattern'][:-3] + item['src_pattern'][-2:] + '.avi'
            item['src_mocap'] = item['src_pattern'] + '_solved_body_R.bvh'
        item.pop('src_vid')
        item.pop('src_pattern')
        # for kee in item.keys():
        #     print('{} : {}'.format(kee, item[kee]))
        # print()

    for item in take_dictionary:
        item['src_mocap'] = item['src_pattern'] + '.bvh'
        item['src_video'] = item['src_pattern'] + '.avi'
        # if any(exception in item['src_pattern'] for exception in exceptions):
        #     print('here')
        item.pop('src_pattern')
        for kee in item.keys():
            print('{} : {}'.format(kee, item[kee]))
        print()


    ultimate_dictionary = {'dictionary_items' : new_dictionary, 'dictionary_takes' : take_dictionary}
    json_dict = json.dumps(ultimate_dictionary)

    with open(new_dict_file, 'w') as f:
        f.write(json_dict)
        # json.dump(f, ultimate_dictionary)
