from lib import SL_dict
import os
import datetime


if __name__ == '__main__':
    """
    Program allows manual change of any sign. Saves changes to new dictionary. Name of new dictionary is based on time of change.
    """
    work_path = '/home/jedle/data/Sign-Language/_source_clean/'
    dict_file_name = 'ultimate_dictionary2.txt'
    orig_dict_file = os.path.join(work_path, dict_file_name)
    akt_time = datetime.datetime.now()
    new_file_name = '{}_manual_changed_{}-{}-{}-{}-{}.txt'.format(os.path.splitext(dict_file_name)[0], akt_time.year, akt_time.month, akt_time.day, akt_time.hour, akt_time.minute)
    changed_dict_file = os.path.join(work_path, new_file_name)

    dictionary_types = ['dictionary_takes', 'dictionary_items']

    # dictionary = SL_dict.read_dictionary(new_dict_file, dictionary_type)
    dict_selection = int(input('Select dictionary type: [0] dictionary, [1] takes :'))

    dictionary_type = dictionary_types[dict_selection]

    searched_sign = input('Select sign (or part of sign): ')
    matches = []
    if dictionary_type == 'dictionary_takes':
        matches = SL_dict.search_take_sign(orig_dict_file, searched_sign, _pattern=True)
    else:
        matches = SL_dict.search_dict_sign(orig_dict_file, searched_sign, _pattern=True)

    if len(matches) != 0:
        for i, m in enumerate(matches):
            print('[{}] {}'.format(i, m))

        selected_edit = input('Select item to edit or [q] to quit: ')
        if selected_edit != 'q':
            tmp_item = matches[int(selected_edit)]
            print(tmp_item)

            kee_list = []
            for i, kee in enumerate(tmp_item):
                kee_list.append(kee)
                print('[{}] {:<28} : {}'.format(i, kee, tmp_item[kee]))
            selected_key = int(input('Select key to change: '))
            old_value = tmp_item[kee_list[selected_key]]
            if type(old_value) is not list:
                new_value = input('Enter new value: ')
                print('Do you want to change: {}'.format(kee_list[selected_key]))
                print('from : {}'.format(old_value))
                print('to: {}'.format(new_value))
                confirm = input('Are you sure to commit change? (yes)')
                if confirm == 'yes':
                    print('Changed')
                    new_item = tmp_item.copy()
                    new_item[kee_list[selected_key]] = new_value
                    dictionary = SL_dict.read_raw(orig_dict_file)
                    for i, item in enumerate(dictionary[dictionary_type]):
                        if item == tmp_item:
                            print(item)
                            dictionary[dictionary_type][i] = new_item
                            print(dictionary[dictionary_type][i])
                    SL_dict.save_dict(changed_dict_file, dictionary)
                    print('Saved to file: {}'.format(new_file_name))
                else:
                    print('Change aborted.')
            else:
                print('This value cannot be changed! (changes in list variables not implemented (yet :D)).')
        else:
            print('Quitting')
    else:
        print('There is no match for: {}'.format(searched_sign))
        print('Quitting')