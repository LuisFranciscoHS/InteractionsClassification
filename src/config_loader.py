def read_config(path_config):
    config = {}
    with open(path_config + 'config.txt', "r") as file_config:
        cont = 0
        for line in file_config:
            cont += 1
            if len(line) == 0:
                print(f"Warning: line {cont} is empty")
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: line '{cont}' does not have two fields.")
                continue
            (key, value) = line.split()
            config[key] = value
    paths = [key for key in config.keys() if 'PATH_' in key]
    append_relative_path(config, path_config, paths)
    print("\nConfiguration READY")
    return config


def append_relative_path(config, prefix, paths):
    for path in paths:
        config[path] = prefix + config[path]