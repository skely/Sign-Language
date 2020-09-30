import os
import json


def ts_conversion(_ts_avi, _align_avi, _align_bvh, _vid_fps=25, _bvh_fps=120):
    """

    :param _ts_avi: [ms]
    :param _align_avi: [video frames]
    :param _align_bvh: [bvh_frames]
    :return:
    """
    res = ((_ts_avi/1000 - _align_avi/_vid_fps) * _bvh_fps) + _align_bvh
    return res


if __name__ == '__main__':
    work_dir = '/home/jedle/data/Sign-Language/dictionary_rev2'
    checklists_dir = os.path.join(work_dir, 'annotation_checklists')
    checklists_list = os.listdir(checklists_dir)

    dict_file = os.path.join(work_dir, 'dictionary3.txt')
    alignment_file = os.path.join(work_dir, 'prehled3.txt')

    new_dictionary_file = os.path.join(work_dir, 'new_dictionary.txt')

    with open(dict_file, 'r') as f:
        dictionary = json.load(f)

    with open(alignment_file, 'r') as f:
        aligns_raw = f.readlines()

    for line in aligns_raw:
        vals = line.strip().split('\t')
        # print(vals)
        if vals[1] == 'yes':
            akt_file_pattern = vals[0][0:2] + vals[0][3:5] + vals[0][6:8]
            # print(akt_file_pattern)
            with open(os.path.join(checklists_dir, akt_file_pattern + '.tab'), 'r') as f:
                tmp_anot = f.readlines()
            vid_align = int(vals[3])
            bvh_align = int(vals[4])
    # ---------------------- checklist mining -----------------------------------
            new_vid_cont_item = []
            for item in tmp_anot:
                akt_line = item.split('\t')
                # print(akt_line)
                sign_id = '-1'
                if len(akt_line) > 1:
                    vid_file_name = vals[0][:10] + '.avi'
                    bvh_file_name = vals[0] + '.bvh'
                    vid_ts_beg = int(akt_line[0])
                    vid_ts_end = int(akt_line[1])
                    bvh_ts_beg = int(round(ts_conversion(vid_ts_beg, vid_align, bvh_align)))
                    bvh_ts_end = int(round(ts_conversion(vid_ts_end, vid_align, bvh_align)))

                    if 'classed' in akt_line[2] or 'rest póza : NOT FOUND!!!' in akt_line[2]:
                        if akt_line[2] == 'T-póza : classed' or akt_line[2] =='T-pose : classed':
                            sign_id = 'T-pose'
                        elif akt_line[2] == 'tra. : classed':
                            sign_id = 'tra.'
                        elif akt_line[2] == 'rest pose : classed' or akt_line[2] == 'rest póza : NOT FOUND!!!':
                            sign_id = 'rest pose'
                        elif akt_line[2] == 'klapka : classed':
                            sign_id = 'klapka'
                        else:
                            print('Not handled line:')
                            print(akt_line)
                    elif akt_line[6] == '':
                        # print(akt_line)
                        pass
                    elif int(akt_line[6]) > 0:
                        sign_id = akt_line[5]
                    elif int(akt_line[6]) < 0:
                        # print(akt_line)
                        pass
                    else:
                        print('Not handled line:')
                        print(akt_line)

                    # print(sign_id)
                    # print(vid_file_name, vid_ts_beg, vid_ts_end)
                    # print(bvh_file_name, bvh_ts_beg, bvh_ts_end)
                    match_number = ([i for i, d in enumerate(dictionary) if d['sign_id'] == sign_id])
                    if match_number == []:
                        # print(sign_id + ' not found in dictionary')
                        pass
                    else:
                        # print(sign_id)
                        # print(dictionary[match_number[0]])
                        if 'vid_continuous' in dictionary[match_number[0]].keys():
                            dictionary[match_number[0]]['vid_continuous'].append([vid_file_name, vid_ts_beg, vid_ts_end])
                            dictionary[match_number[0]]['bvh_continuous'].append([bvh_file_name, bvh_ts_beg, bvh_ts_end])
                        else:
                            dictionary[match_number[0]]['vid_continuous'] = [[vid_file_name, vid_ts_beg, vid_ts_end]]
                            dictionary[match_number[0]]['bvh_continuous'] = [[bvh_file_name, bvh_ts_beg, bvh_ts_end]]
        # break

    # for item in dictionary:
    #     # print(item['sign_id'], item.keys())
    #     if 'bvh_continuous' in item.keys():
    #         for kee in item.keys():
    #             print(kee,' : ',  item[kee])
    #
    #         break

    with open(new_dictionary_file, 'w') as jf:
        json.dump(dictionary, jf)