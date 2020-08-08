import argparse
import toml
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--type", help="AM type", type=str)
parser.add_argument("--file_index", help="File index", type=int, default=0)
args = parser.parse_args()

config = toml.load('config.toml')

for i in range(4):
    for phase in range(config['main']['npoints']//100 + 1):
        command = ("addqueue -s -q berg -n 1x10 -m 7 /usr/bin/pyhon3 "
                   "sim_cut.py --name {} --type {} --cut {} --file_index {} "
                   "--phase {}"
        commmand = command.format(args.name, args.type, i, args.file_index,
                                  phase)
