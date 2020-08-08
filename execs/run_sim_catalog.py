import os

name = 'NYUmatch'
scope = (-30.0, -22.0)
# bounds
alpha = (0.0, 1.0)
scatter = (0.005, 0.3)
nd_gal = '../../Data/NYUmatch/LF_NYU.npy'
Nnew = 100

command = ("addqueue -s -q berg -n 1x10 -m 4.5 /usr/bin/python3 sim_cut.py "
           "--name {} --scope {} {} --nd_gal {} --nnew {} "
           "--alpha {} {} --scatter {} {}").format(
                   name, scope[0], scope[1], nd_gal, Nnew, alpha[0], alpha[1],
                   scatter[0], scatter[1])

print(command)
os.system(command)
