def read_dictionary(file_path, file_name):
    """
    Read two columns mapping file into a dictionary key --> tuple.\n
    Removes repeated values in each tuple.\n
    Returns the dictionary object. 
    """
    file = open(file_path + file_name)
    result = {}
    for line in file:
        key, value = line.split()
        result.setdefault(key, set()).add(value)
    return result
