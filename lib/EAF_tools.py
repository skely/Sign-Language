import os
import numpy as np
from lib import SL_dict


def read_eaf(_infile):
    """
    reads eaf annotation file
    :param _infile: eaf file
    :return: list of annotations
    """
    with open(_infile, 'r') as f:
        tmp = f.readlines()
    time_stamps = []
    for line in tmp:
        if '<TIME_SLOT TIME_SLOT_ID=' in line:
            stamp = line.split('"')[3]
            time_stamps.append(stamp)
    annotation = []
    for line in tmp:
        if 'ANNOTATION_ID=' in line:
            tmp = line.split('"')
            ts_number = (int(tmp[3].replace('ts', '')), int(tmp[5].replace('ts', '')))
        if '<ANNOTATION_VALUE>' in line:
            line = line.replace('>', '<')
            meaning = line.split('<')
            annotation.append((time_stamps[ts_number[0]-1], time_stamps[ts_number[1]-1], meaning[2]))
    return annotation


def make_translation_matrix(tmp='ťěščřžýáíéúůóŤĚŠČŘÝÁÍÉÚŮÓ', trl='tescrzyaieuuoTESCRYAIEUUO'):
    translation_tab = np.zeros((len(tmp), 2), dtype=int)
    for _i in range(len(tmp)):
        translation_tab[_i, 0] = ord(tmp[_i])
        translation_tab[_i, 1] = ord(trl[_i])
    return translation_tab


def remove_wedges(in_string):
    tab = make_translation_matrix()
    out_string = ''
    for _i in range(len(in_string)):
        searched_ord = ord(in_string[_i])
        if searched_ord in tab[:, 0]:
            out_string += chr(tab[np.where(tab[:, 0] == searched_ord), 1])
        else:
            out_string += in_string[_i]
    return out_string


def parse_EAF(eaf_file, dictionary_file, out_file_path):
    """
    Parse EAF annotation file and save data for annotation-check-file
    :param eaf_file: path to EAF file
    :param dictionary_file: dictionary file
    :param out_file_path: path for output file
    """
    dictionary = SL_dict.read_valid(dictionary_file)

    # ***** checklist name generator ******
    sep = eaf_file.split(' ')[0].split('.')
    if len(eaf_file.split('.')) > 3:
        if ')' in sep[2][-1]:
            out_file = os.path.join(out_file_path, 'anot_checklist_' + str(sep[2][2:4]) + str(
                sep[1].zfill(2) + str(sep[0].zfill(2)) + str(sep[2][4:]) + '_FB.txt'))
        else:
            out_file = os.path.join(out_file_path, 'anot_checklist_' + str(sep[2][2:4]) + str(
                sep[1].zfill(2) + str(sep[0].zfill(2)) + '_FB.txt'))
    else:
        out_file = os.path.join(out_file_path, 'anot_checklist_' + eaf_file.split(' Filip')[0] + '_FB.txt')

    # ***** eaf parser *****
    eaf_infile = os.path.join(out_file_path, eaf_file)
    annot = read_eaf(eaf_infile)
    outfile_feed = []

    for i, line in zip(range(len(annot)), annot):
        classed_flag = False
        dict_flag = False
        meaning = remove_wedges(line[2])
        meaning_split = (meaning.split(' '))

        # ***** filter events *****
        if meaning == 'tra.':
            # print('annot code found')
            classed_flag = True
        elif meaning == 'T-poza':
            # print('annot code found')
            classed_flag = True
        elif meaning == 'T-pose':
            # print('annot code found')
            classed_flag = True
        elif meaning == 'klapka':
            # print('annot code found')
            classed_flag = True
        elif meaning == 'rest pose':
            # print('annot code found')
            classed_flag = True
        else:
            dict_matches = []
            dict_matches_id = []
            for item in dictionary:
                if 'sign_meaning' not in item.keys():
                    print(item)
                item_meaning_wedgeless = remove_wedges(item['sign_meaning'])
                if item_meaning_wedgeless == meaning_split[0]:
                    # print('found dict item: {}, with meaning: \"{}\".'.format(i, item['meaning']))
                    dict_flag = True
                    dict_matches.append(item['sign_meaning'])
                    dict_matches_id.append(item['sign_id'])
                elif meaning_split[0] in item_meaning_wedgeless:
                    # print('found as part in dict item: {}, with meaning: \"{}\".'.format(i, item['meaning']))
                    dict_flag = True
                    dict_matches.append(item['sign_meaning'])
                    dict_matches_id.append(item['sign_id'])
                elif item_meaning_wedgeless in meaning_split[0]:
                    # print('part found in dict item: {}, with meaning: \"{}\".'.format(i, item['meaning']))
                    dict_flag = True
                    dict_matches.append(item['sign_meaning'])
                    dict_matches_id.append(item['sign_id'])

        # ***** clasification and output to anot file *****
        if classed_flag:
            answer = '{}\t{}\t{} : {}\n'.format(line[0], line[1], line[2], 'classed')
            print(answer)
            outfile_feed.append(answer)
        elif dict_flag:
            answer = '{}\t{}\t{} : {}'.format(line[0], line[1], line[2], 'found dict item: {}'.format(dict_matches))
            print(answer)
            outfile_feed.append(answer)
            answer = '\t\t : {}\n'.format(dict_matches_id)
            print(answer)
            outfile_feed.append(answer)
        else:
            answer = '{}\t{}\t{} : {}\n'.format(line[0], line[1], line[2], 'NOT FOUND!!!')
            print(answer)
            outfile_feed.append(answer)

    # ***** write annot_checklist file *****
    with open(out_file, 'w') as f:
        f.writelines(outfile_feed)
