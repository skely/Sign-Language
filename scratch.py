import os
import json
from lib import SL_dict


work_path = '/home/jedle/data/Sign-Language/_source_clean/'
new_dict_file = os.path.join(work_path, 'ultimate_dictionary.txt')
bvh_path = '/home/jedle/data/Sign-Language/_source_clean/bvh/'

dict_takes = SL_dict.read_raw(new_dict_file, 'dictionary_takes')
dict_dict = SL_dict.read_raw(new_dict_file, 'dictionary_items')

real_bvh_file_list = os.listdir(bvh_path)

for item in dict_takes:
    real_name = [bvh_name for bvh_name in real_bvh_file_list if  item['src_mocap'][:-4] in bvh_name][0]
    item['src_mocap'] = real_name


new_dictionary = {'dictionary_items' : dict_dict, 'dictionary_takes' : dict_takes}

SL_dict.save_dict(os.path.join(work_path, 'ultimate_dictionary2.txt'), new_dictionary)
