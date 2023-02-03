import os, sys, json, argparse
dn = os.path.dirname
prev_dir = dn(dn(os.path.abspath(__file__)))
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