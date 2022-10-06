import os, sys, json, argparse
prev_dir = os.path.dirname(os.path.dirname("./"))
config_path = os.path.join(prev_dir, 'machine_config.json')

global main_dir
global data_dir
global pth_dir

with open(config_path, 'r') as f:
    machine_config= json.load(f)
    f.close()

main_dir = machine_config['main_dir']
data_dir = machine_config['data_dir']
pth_dir = machine_config['pth_dir']

sys.path.append(main_dir)

def main_train_parser():
    parser = argparse.ArgumentParser("script for training differentiaion diffusion")
    parser.add_argument("--CUDA", type=str, default='cpu', help="nominate the GPU to use if there are multiple devices")
    parser.add_argument("--model_config", type=str, required=True, help='the abs path of config json file which has all hyper the parameters')
    return parser