from lib import SL_dict
import os
import json
import numpy as np


if __name__ == '__main__':
    work_path = '/home/jedle/data/Sign-Language/_source_clean/'
    new_dict_file = os.path.join(work_path, 'ultimate_dictionary.txt')

    dictionary_types = ['dictionary_takes', 'dictionary_items']

    # dictionary = SL_dict.read_dictionary(new_dict_file, dictionary_type)
    dict_selection = int(input('Select dictionary type: [0] dictionary, [1] takes :'))

    dictionary_type = dictionary_types[dict_selection]

    searched_sign = input('Zadejte znak nebo jeho část: ')
    matches = []
    if dictionary_type == 'dictionary_takes':
        matches = SL_dict.search_take_sign(new_dict_file, searched_sign, _pattern=True)
    else:
        matches = SL_dict.search_dict_sign(new_dict_file, searched_sign, _pattern=True)

    for i, m in enumerate(matches):
        print('[{}] {}'.format(i, m))

    selected_edit = input('Select item to edit or [q] to quit: ')
    if selected_edit == 'q':
        pass
    else:
        tmp_item = matches[int(selected_edit)]
        print(tmp_item)

    for i, kee in enumerate(tmp_item):
        print('[{}] {:<28} : {}'.format(i, kee, tmp_item[kee]))
    selected_key = int(input('Select key to change: '))
    new_value = input('Enter new value: ')

    

