import os
from lib import EAF_tools

if __name__ == "__main__":
    work_path = '/home/jedle/data/Sign-Language/_source_clean/annotations/new_eafs/'
    flist = [f for f in os.listdir(work_path) if '.eaf' in f]
    slovnik_path = '/home/jedle/data/Sign-Language/dictionary/'
    slovnik = os.path.join(slovnik_path, 'dictionary_dict_v4.txt')

    for tmp_file in flist:
        EAF_tools.parse_EAF(tmp_file, slovnik, work_path)

