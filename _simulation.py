import os
import argparse

from PySHAM import utils

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str)
args = parser.parse_args()

inp = utils.load_pickle(args.fname)
grid = inp['grid']

grid.run_stored()

out = {'grid': grid, 'fname': inp['fname'], 'i': inp['i'] + 1}
# save the new interim file
utils.dump_pickle(inp['fname'].format(inp['i'] + 1), out)
# delete the old interim file
os.remove(args.fname)

print('done')
