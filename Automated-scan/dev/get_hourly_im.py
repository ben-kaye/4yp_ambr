# import numpy as np
import shutil
from os.path import exists


# example
shutil.copyfile('source.txt', 'destination.txt')

loc = './../Exp-03-10/Part A/'
dest = loc + 'HR/'

for i in range(24):
    file_name = 'scan_' + str(i*60)

    src = loc + file_name
    dest_fl = dest + file_name

    if exists(src):
        shutil.copyfile(src, dest_fl)
