from time import sleep, time
import cv2
# import exifread
from Detect import Well_Detector
import numpy as np
from os.path import exists
import re

import os


class Controller:

    current_index = 0
    data_path = './Experiment-data/scan_'
    out_dir = './Experiment-processed/'
    ft = '.bmp'
    poll_time = 10
    im_write = True
    wells = None
    data = []

    radial_amount = 0.5  # [%]
    mask = []

    # cwd = ''
    N_wells = 12
    R = 38

    exp_running = True

    def __init__(self, start_index=0, data_folder='Experiment-data', out_folder='Experiment-processed'):

        print('controller running from ' + os.getcwd())

        self.data_path = './' + data_folder + '/scan_'
        self.out_dir = './' + out_folder + '/'

        self.current_index = start_index

        self.mask = self.compute_mask()

    def run_control(self, overwrite = True):

        if overwrite:
            
            if self.current_index > 0:
                print('WARNING start index (' + str(self.current_index) + ') NOT 0, ABORTING')
                return False

            # self.current_index = 0

            cols = ['w' + str(k) for k in range(1, 13)]
            # cols.extend(['w' + str(k) + 'c' for k in range(1, 13)])
            pattern = r'[\[\]\(\)\']'
            print_data = 'time,' + re.sub(pattern, '', str(cols))

            # print('writing with cwd ' + self.cwd)

            with open(self.out_dir + 'exp_r.csv', 'w') as p:
                p.writelines(print_data + '\n')
            with open(self.out_dir + 'exp_g.csv', 'w') as p:
                p.writelines(print_data + '\n')
            with open(self.out_dir + 'exp_b.csv', 'w') as p:
                p.writelines(print_data + '\n')
        else: 
            row_count = 0
            with open(self.out_dir +'exp_r.csv', 'r') as f:
                row_count = sum(1 for line in f)

            expected_idx = row_count - 1 # num rows - 1 as header col + want to be next idx
            if expected_idx != self.current_index:
                self.current_index = expected_idx
                print('WARNING start index (' + str(self.current_index) + ') replaced with ' + str(expected_idx))

        while self.exp_running:
            t_start = time()
            self.read_scan()
            t_delta = time() - t_start

            if ~exists(self.get_path(self.current_index)) & (t_delta < self.poll_time):
                sleep(self.poll_time - t_delta)

    def get_path(self, idx):
        return self.data_path + str(idx) + self.ft

    def read_scan(self, abort=False):

        path_file = self.get_path(self.current_index)

        file_exists = exists(path_file)
        dateTaken = None
        im = None

        if file_exists:
            im = self.read_im(path_file)
        else:
            print('data processing up to date')

        if im is not None:
            dateTaken = os.path.getmtime(path_file)

            success = self.process_scan(im, dateTaken)

            if success:
                self.write_data()
                self.current_index += 1

    def recover_wells(self, im_path):
        WF = Well_Detector()
        self.wells = WF.return_wells(im_path)

    def recover_wells(self):
        self.wells = Well_Detector.read_well_loc()

    def process_scan(self, im, dateTaken):

        condensed_im = Controller.crop_ims(self.wells, im, mask=self.mask)

        if len(condensed_im) < self.N_wells:
            return False

        data_at_t = [[round(k,4) for k in self.avg_well(i)] for i in condensed_im]
        # self.data.append((densities, dateTaken))

        # time_at_t

        time_min = round(dateTaken/60, 2)

        self.data.append((time_min, data_at_t))

        if self.im_write:
            write_im = np.hstack(condensed_im)

            path = self.out_dir + 'ims/wells_' + \
                str(self.current_index) + '.png'

            cv2.imwrite(path, write_im)

        return True

    def read_im(self, path):
        return cv2.imread(path)

    def crop_ims(wells, image, mask=[]):

        well_ims = []
        for x, y, r in wells:

            sub_im = image[y-r:y+r, x-r:x+r]

            imx, imy, z = np.shape(sub_im)

            # for u in range(imx):
            # for v in range(imy):
            # if mask[imx*u + v] <= 0 :
            # sub_im[v,u] = np.array([0,0,0], dtype=np.uint8)

            if sub_im.any():
                well_ims.append(sub_im)

        return well_ims

    def compute_mask(self):

        width = 2*self.R
        mask = np.zeros((width, width))

        mask = []

        for u in range(width):
            for v in range(width):
                mask.append(1 if (u - self.R)**2 + (v - self.R) **
                            2 < (self.R*self.radial_amount)**2 else 0)
        f = sum(mask)

        return np.array(mask)/f

    def write_data(self):

        next_datum = self.data[-1]

        time = next_datum[0]
        bgr = next_datum[1]
        # dens, color = list(zip(*next_datum))

        b, g, r = list(zip(*bgr))

        # line = []
        # line.extend(dens)
        # line.extend(color)

        pattern = r'[\[\]\(\)\']'
        r_line = str(time) + ',' + re.sub(pattern, '', str(r))
        g_line = str(time) + ',' + re.sub(pattern, '', str(g))
        b_line = str(time) + ',' + re.sub(pattern, '', str(b))

        with open(self.out_dir + 'exp_r.csv', 'a') as p:
            p.writelines(r_line + '\n')

        with open(self.out_dir + 'exp_g.csv', 'a') as p:
            p.writelines(g_line + '\n')

        with open(self.out_dir + 'exp_b.csv', 'a') as p:
            p.writelines(b_line + '\n')

    def avg_well(self, well_im):
        (x, y, z) = np.shape(well_im)

        avg = np.zeros((3,))

        # TODO THIS MAY BE WRONG AS IM[Y][X]

        for u in range(x):
            for v in range(y):
                avg += well_im[v][u] * self.mask[x*u + v]

        pixel = avg.astype(np.single).reshape((1, 1, 3))
        density = cv2.cvtColor(pixel, cv2.COLOR_RGB2GRAY)[0, 0]

        cols = [hex(round(w)) for w in list(avg)]
        colorhex = '#'+str.upper(cols[0][2:] + cols[1][2:] + cols[2][2:])

        return list(avg)
