import json


def read_raw(_in_file):
    with open(_in_file, 'r') as f:
        _take_dict = json.load(f)
    return _take_dict


def search_dict_sign(_dict_file, _sign_id):
    _dict = read_raw(_dict_file)
    _item = {}
    for _item in _dict:
        if _item['sign_id'] == _sign_id:
            break
    return _item


def search_take_file(_dict_file, _take_file):
    _dict = read_raw(_dict_file)
    ret_list = []
    for _item in _dict:
        if _item['src_pattern'] in _take_file:
            ret_list.append(_item)
    return ret_list


def search_take_sign(_dict_file, _sign_id):
    _dict = read_raw(_dict_file)
    ret_list = []
    for _item in _dict:
        if _item['sign_id'] == _sign_id:
            ret_list.append(_item)
    return ret_list