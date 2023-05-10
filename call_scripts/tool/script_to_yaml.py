import yaml



def convert_to_yaml(filename):
    config = {}
    current_section = ""
    import pdb;pdb.set_trace()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                current_section = line[1:-1]
                if current_section not in config:
                    config[current_section] = {}
            else:
                key, value = line.split("=", 1)
                config[current_section][key.strip()] = value.strip()
    return yaml.dump(config)

ans=convert_to_yaml('checkpoints/m-B-1-1-N-UR20M/m-B-1-1-N-UR20M.sh')