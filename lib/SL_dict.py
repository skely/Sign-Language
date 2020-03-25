import json


def read_raw(_in_file):
    """
    Reads json dictionary file.
    :param _in_file: path to dictionary file
    :return: list of dictionaries items
    """
    with open(_in_file, 'r') as f:
        _dict = json.load(f)
    return _dict


def read_valid(_in_file):
    """
    Reads json dictionary file and returns valid items only (annotation flag-wise)
    :param _in_file: path to dictionary
    :return: list of dictionary items
    """
    with open(_in_file, 'r') as f:
        _dict = json.load(f)
    _dict = [i for i in _dict if i['annotation_flag'] == 1]
    return _dict


def save_dict(_file, _dictionary):
    """
    Saves json dictionary to file
    :param _file: path to dictionary file
    :param _dictionary: list of dictionary items
    """
    with open(_file, 'w') as jf:
        json.dump(_dictionary, jf)


def search_dict_sign(_dict_file, _sign_id):
    """
    search dictionary items for given sign_id
    :param _dict_file: path to dictionary file
    :param _sign_id: sign_id
    :return: sign info (dict)
    """
    _dict = read_raw(_dict_file)
    _item = {}
    for _item in _dict:
        if _item['sign_id'] == _sign_id:
            break
    return _item


def search_take_file(_dict_file, _take_file):
    """
    search annotations for take file.
    :param _dict_file: path to "takes annotation" file
    :param _take_file: name of searched take
    :return: list of items in take
    """
    _dict = read_raw(_dict_file)
    ret_list = []
    for _item in _dict:
        if _item['src_pattern'] in _take_file:
            ret_list.append(_item)
    return ret_list


def search_take_sign(_dict_file, _sign_id):
    """
    search for sign_id in all takes annotation
    :param _dict_file: path to "takes annotation" file
    :param _sign_id: sign_id
    :return: list of items
    """
    _dict = read_raw(_dict_file)
    ret_list = []
    for _item in _dict:
        if _item['sign_id'] == _sign_id:
            ret_list.append(_item)
    return ret_list