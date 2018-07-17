import json
import argparse
import subprocess
import shlex
import os
import multiprocessing

parser = argparse.ArgumentParser(description='Create obj+mtl files for the houses in the dataset.')
parser.add_argument('--eqa-path', help='/path/to/eqa.json', required=True)
parser.add_argument('--suncg-path', help='/path/to/suncg/toolbox', required=True)
parser.add_argument('--suncg-data-path', help='/path/to/suncg/data_root')
parser.add_argument('--nprocs', help='number of threads to use', default=multiprocessing.cpu_count())
args = parser.parse_args()


if args.suncg_data_path is None:
    args.suncg_data_path = os.path.join(args.suncg_path, 'data_root')
eqa_data = json.loads(open(args.eqa_path).read())
houses = list(eqa_data['questions'].keys())
start_dir = os.getcwd()
def extract_threaded(house):
    os.chdir(start_dir)
    os.chdir(os.path.join(args.suncg_data_path, 'house', house))
    subprocess.call(shlex.split(
        '%s house.json house.obj' % (
            os.path.join(os.pardir, os.pardir, os.pardir, 'gaps', 'bin', 'x86_64', 'scn2scn'),
        )))
    print('extracted', house)

pool = multiprocessing.Pool(args.nprocs)
pool.map(extract_threaded, houses)

