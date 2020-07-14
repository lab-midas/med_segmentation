from ..Patient.Patient import *
import yaml
import argparse
from ..util import convert_yaml_config

def args_argument():
    parser = argparse.ArgumentParser(prog='Nora')
    parser.add_argument("--gpu", type=int, default=0, help=" which gpu")
    parser.add_argument('-c', '--config_path', type=str, default='../config/config_default.yaml', help='Config path of the project')
    a = parser.parse_args()
    return a

args = args_argument()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

with open(args.config_path, "r") as yaml_file:
    config = yaml.load(yaml_file.read())
    config = convert_yaml_config(config)


patient=Patient().initialize(config=config, name_ID='KORA1234567', datatype='KORA')
patient.get_plot()