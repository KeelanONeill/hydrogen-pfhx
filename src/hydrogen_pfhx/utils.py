import yaml

def load_config(configuration_file_path):
    # Step 1. read config file
    with open(configuration_file_path, "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    return configuration

# convert from tpd to kg/s
def tpd_to_kps(mass_flow_rate):
    return mass_flow_rate * 1e3 / (60**2*24)
