from lib import SL_dict
import os
import json
import numpy as np


if __name__ == '__main__':
    work_path = '/home/jedle/data/Sign-Language/_source_clean/'
    new_dict_file = os.path.join(work_path, 'ultimate_dictionary.txt')

    dictionary = SL_dict.read_dictionary(new_dict_file, 'dictionary_takes')

    for item in dictionary:
        print(item)