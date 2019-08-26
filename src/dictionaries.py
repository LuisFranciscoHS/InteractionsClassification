import os


def read_dictionary(file_path, file_name, order_pairs=False, col_indices=(0, 1), ignore_header=False):
    """
    Read two columns mapping file into a dictionary key --> set.\n
    Removes repeated values for each key.\n
    If order_pairs=True pairs are added in Lexicographical order: key < value
    """
    result = {}
    with open(file_path + file_name) as file:
        if ignore_header:
            file.readline()
        for line in file:
            fields = line.split('\t')
            if max(col_indices) >= len(fields):
                raise ValueError(f"File does not have the columns requested: {col_indices}")
            key, value = fields[col_indices[0]], fields[col_indices[1]]
            if order_pairs and key > value:
                key, value = value, key
            result.setdefault(key.strip(), set()).add(value.strip())
    return result


def read_set_from_columns(file_path, file_name, col_indices=(0, 1), ignore_header=False):
    """Creates a set with the values of the selected columns"""
    result = set()
    with open(file_path + file_name) as file:
        if ignore_header:
            file.readline()
        for line in file:
            fields = line.split('\t')
            if max(col_indices) >= len(fields):
                raise ValueError(f"File does not have the columns requested: {col_indices}")
            for index in col_indices:
                result.add(fields[index].strip())
    return result


def convert_tab_to_dict(response):
    """Convert tab response to dictionary.\n
    Response is a string made of rows with two tab separated columns."""
    from collections import defaultdict

    if not type(response) == type("hola"):
        print("The argument is not a string.")
        return {}
    result = defaultdict(set)
    for entry in response.splitlines()[1:]:
        from_id, to_id = tuple(entry.split())
        result[from_id].add(to_id)
    return result


def write_dictionary_one_to_one(dictionary, file_path, file_name):
    if len(file_path) > 0:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path + file_name, 'w') as file:
        for k, v in dictionary.items():
            file.write(f"{k}\t{v}\n")
        file.close()


def write_dictionary_one_to_set(dictionary, file_path, file_name):
    """Writes as two column file each pair of key --> value in the set of values of the keys."""
    if len(file_name) == 0:
        raise ValueError('File name can not be the empty string')
    if len(file_path) > 0:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path + file_name, 'w') as file:
        for key, values in dictionary.items():
            for value in values:
                file.write(f"{key}\t{value}\n")


def get_intersection(dictionary1, dictionary2):
    """Get a new dictionary key --> {value set} with the keys of the first dictionary that appear in the second"""
    result = {}
    for k1, s1 in dictionary1.items():
        for v1 in s1:
            if k1 in dictionary2.keys() and v1 in dictionary2[k1]:
                result.setdefault(k1, set()).add(v1)
    return result


def in_dictionary(pairs, dictionary):
    """Returns a list of booleans indicating if each pair of the list appears in the dictionary"""
    result = []
    for k, v in pairs:
        result.append(1 if k in dictionary and v in dictionary[k] else 0)
    return result


def flatten_dictionary(dictionary):
    """Get an unpacked list of the key value pairs of the dictionary"""
    result = []
    for k, s in dictionary.items():
        for v in s:
            result.append((k, v))
    return result
