def read_config(path_config):
    config = {}
    with open(path_config + 'config.txt', "r") as file_config:
        for line in file_config:
            (key, value) = line.split()
            config[key] = value
    paths = [key for key in config.keys() if 'PATH' in key]
    append_relative_path(config, path_config, paths)
    print("\nConfiguration READY")
    return config


def append_relative_path(config, prefix, paths):
    for path in paths:
        config[path] = prefix + config[path]