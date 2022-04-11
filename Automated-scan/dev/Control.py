from concurrent.futures import process
from inspect import stack
import itertools
from msilib.schema import Control, ControlCondition
import queue
from time import sleep, time
import cv2
from cv2 import split
from cv2 import mean
# import exifread
from Detect import Well_Detector
import numpy as np
from os.path import exists
import re
import os

import math

import serial


class Controller:

    current_index = 0
    data_path = './Experiment-data/scan_'
    out_dir = './Experiment-processed/'
    alt_out_dir = './Experiment-processed/'

    ft = '.bmp'
    poll_time = 10
    im_write = True
    wells = None
    data = []

    data_files = ('time', 'exp_b', 'exp_g', 'exp_r',
                  'exp_b_sd', 'exp_g_sd', 'exp_r_sd')

    num_data_round = 4
    time_data_round = 2

    offset = (0, 0)
    offsets = {  # dict of index and offset:
        1417: (0, -17),
        # 2649: (0, -14),
        2685: (0, -11),  # up 11 from baseline
        3910: (0, -16)  # up 16 from baseline
    }

    control_depth = 4
    past_observation = []
    error = [0,0]

    radial_amount = 0.5  # [%]
    mask = []

    # cwd = ''
    N_wells = 12
    R = 38

    intensity_baseline = 0

    T_sample = 60  # [s]

    s_pattern = r'[\[\]\(\)\'\s]'

    exp_running = True

    serial_connection = None

    def __init__(self, start_index=0, im_write=True, data_folder='Experiment-data', out_folder='Experiment-processed', alt_out_folder=None):

        print('controller running from ' + os.getcwd())

        self.data_path = './' + data_folder + '/scan_'
        self.out_dir = './' + out_folder + '/'

        if alt_out_folder is not None:
            self.alt_out_dir = './' + alt_out_folder + '/'
        else:
            self.alt_out_dir = self.out_dir

        self.im_write = im_write

        self.current_index = start_index

        for k in self.offsets:
            if self.current_index > k:
                self.offset = self.offsets[k]

        self.mask = self.compute_mask()

        if self.im_write:
            im_path = self.out_dir + 'ims/'
            if not exists(im_path):
                os.mkdir(im_path)

    def run_online_controller(self):
        pass

        # should include error mitigation in case of arduino disconnect.

        ser = serial.Serial('/dev/ttyACM1', 9600)  # baudrate

        ser.write('000,1,')

        self.process_loop(start_index=0, overwrite=True,
                          func=self.control_observer_iter)

    def control_observer_iter(self):

        # OBSERVER
        p_error = self.run_observe_iter()

        # CONTROLLER

        self.error = (p_error, self.error[1] + p_error/3600) # create error vector by integrating past value

        dem = Controller.PI_controller(self.error)
        self.past_demand = dem
        command = Controller.convert_demand(dem)

        self.serial_connection.write(command)

    def run_observe_iter(self):
        # create copy of last observations in case
        old_data = tuple(self.past_observation)

        next_data_point = self.data[-1]  # assuming exactly T seconds later.

        # average data
        processed_data = []

        # filter data to generate intensity
        observed_data = processed_data

        self.past_observation.pop(0)
        self.past_observation.append(observed_data)

        return observed_data

    def PI_controller(e_x):
    # returns pump demand fraction from given error and its integral

        e, e_I = e_x

        ff = 37.3  # feed forward

        u = 300*e + 50*e_I + ff

        p_demand = Controller.umH2O_to_pump(u)

        if p_demand >= 1:
            p_demand = 0.999
        if p_demand <= -1:
            p_demand = -0.999

        return p_demand

    def umH2O_to_pump(umH2O):
    # take units umH2O & convert to pump fraction in (-1,1)

        # R = 59.8 # [um/(mL/hr)]
        # Q = umH2O/R # [mL/hr]
        # pump_factor = 91.7 # [(mL/hr)/(pump_fraction)]
        # demand = Q*conv_factor
        return umH2O/91.7/59.8 # pump fraction

    def convert_demand(val):
        dir_s = '1' if math.sign(val) > 0 else '0'

        v = abs(val)

        if v < 0 or v >= 1:
            print('warning: demand out of range\n')
            ret = '000'
        else:
            ret = str(round(v, 3))[2:]  # get vals after dp
        if len(ret) < 3:
            x = 3 - len(ret)
            ret = ret + '0'*x
        return ret + ',' + dir_s + ','

    def set_up_data(self, start_index, overwrite):
        names = self.data_files

        self.current_index = start_index

        if start_index == 0:
            cols = ['w' + str(k) for k in range(1, 13)]

            col_names = 'time,' + re.sub(self.s_pattern, '', str(cols)) + '\n'

            for i in range(len(names)):
                with open(self.out_dir + names[i]+'.csv', 'w') as p:
                    if i > 0:
                        p.writelines(col_names)
                    else:
                        p.writelines('index,time\n')
        else:
            if overwrite:

                if start_index > 0:
                    print('WARNING overwriting from index (' +
                          str(start_index) + ') NOT 0')

                    path = self.out_dir

                    for n in names:
                        data = []
                        with open(path+n+'.csv', 'r') as f:
                            data = f.readlines()[0:start_index]

                        with open(path+n+'.csv', 'w') as f:
                            f.writelines(data)

            else:
                row_count = 0
                with open(self.out_dir + 'exp_r.csv', 'r') as f:
                    row_count = sum(1 for line in f)

                expected_idx = row_count - 1  # num rows - 1 as header col + want to be next idx
                if expected_idx is not start_index:
                    self.current_index = expected_idx
                    print('WARNING start index (' + str(start_index) +
                          ') replaced with ' + str(expected_idx))

    def process_loop(self, start_index=0, overwrite=True, post_process_func=None):
        # check for new scans and process them.

        self.set_up_data(start_index=start_index, overwrite=overwrite)

        while self.exp_running:
            t_start = time()
            new_data = self.read_scan()
            if new_data and post_process_func is not None:
                if self.current_index == 1:
                    data = self.data[-1][1] 
                    k = zip(*data)
                    i = mean((mean(k)[0:3]))
                    self.intensity_baseline = i

                post_process_func()
            t_delta = time() - t_start

            if ~exists(self.get_sc_path(self.current_index)) & (t_delta < self.poll_time):
                sleep(self.poll_time - t_delta)

    def get_sc_path(self, idx):
        return self.data_path + str(idx) + self.ft

    def get_pr_path(self, idx):
        return self.out_dir + 'ims/wells_' + \
            str(idx) + '.png'

    def read_scan(self, abort=False):

        path_file = self.get_sc_path(self.current_index)

        # update offset if at shift
        if self.current_index in self.offsets:
            self.offset = self.offsets[self.current_index]

        im, date = Controller.read_im_from_disk(path_file)

        success = False

        if im is not None:

            success = self.process_scan(im, date)

            if success:
                self.write_data()
                self.current_index += 1
        return success

    def recover_wells(self, im_path):
        WF = Well_Detector()
        self.wells = WF.return_wells(im_path)

    def recover_wells(self):
        self.wells = Well_Detector.read_well_loc()

    def find_offset(self, index):
    
        rgbs = self.data[index][1]

        diffs = [0, 0, 0]

        # TODO FINISH THIS

        loc_diffs = ((0,0))

        # if max(abs(diffs)) > 4: come up with good automatic correction detection mechanism

        WD = Well_Detector()
        new_wells = WD.return_wells(self.get_sc_path(index))

        loc_diffs = tuple((new_wells[i][0] - self.wells[i][0], new_wells[i][1] - self.wells[i][1]) for i in range(len(self.wells)))
                
        x_dif, y_dif = zip(*loc_diffs)
        return (round(mean(x_dif)), round(mean(y_dif)))        

    def recover_cropped(self):
    # utility to process data from series of cropped wells instead of fulls scans. for use offline.

        processing = True
        while(processing):
            processing = self.read_stacked_well_im()

        with open(self.out_dir + 'exp_r.csv', 'r') as f:  # get the times.
            old_data = f.readlines()
        del old_data[0]
        times_old = tuple(float(x.split(',')[0]) for x in old_data)
        # indices = tuple(i for i in range(len(old_data)))
        # sort_time = zip(times_old, indices)

        # _, u = sort_time

        self.data = list(zip(times_old, self.data))
        self.data.sort(key=lambda tup: tup[0])

        # times_new =

        self.write_data_oneshot()

    def write_data_oneshot(self):
    # rewrites extracted data.
        
        col = self.data_files[1:]

        time, dat = zip(*self.data)
        # got [ .. , ..]
        # with list of 12 x 6 tuples
        # u = tuple(zip(*dat))
        u = [[] for x in range(6)]
        # for d in dat:
        for k in range(len(dat)):
            d = dat[k]

            x = tuple(zip(*d))
            for i in range(len(u)):

                pr = tuple(map(lambda z: round(z, self.num_data_round), x[i]))

                line = str(time[k]-time[0])+',' + \
                    re.sub(self.s_pattern, '', str(pr)) + '\n'

                u[i].append(line)

        if not exists(self.alt_out_dir):
            os.mkdir(self.alt_out_dir)

        for k in range(len(col)):
            path = self.alt_out_dir+col[k]+'.csv'

            if exists(path):
                with open(path, 'w') as f:
                    f.writelines(u[k])
            else:
                with open(path, 'x') as f:
                    f.writelines(u[k])

    def read_stacked_well_im(self):
    # read stacked-well image, process, and record data
        success = False

        path_to_im = self.get_pr_path(self.current_index)
        # time not reliable metric, must use the recorded value in the data:
        im_, date_ = Controller.read_im_from_disk(path_to_im)

        if im_ is not None:
            success = True
            # do process:
            ims = Controller.extract_well_snapshot(self.N_wells, im_)

            data = [self.avg_well(x) for x in ims]

            # time_in_min = round(date_/60, 3)

            self.data.append(data)

            self.current_index += 1
        return success

    def extract_well_snapshot(N_wells, stacked_im):
        ims = []
        for i in range(N_wells):
            ims.append(stacked_im[:, i*76:(i+1)*76])
        return ims

    def read_im_from_disk(path):
    # returns image and date modified
        dateTaken = None
        im = None

        if exists(path):
            dateTaken = os.path.getmtime(path)
            im = cv2.imread(path)

        return im, dateTaken

    def process_scan(self, im, dateTaken):
    # extract scan data

        condensed_im = self.crop_ims(im)

        if len(condensed_im) < self.N_wells:
            return False

        data_at_t = [[round(k, self.num_data_round) for k in self.avg_well(i)]
                     for i in condensed_im]
        # self.data.append((densities, dateTaken))

        # time_at_t

        time_min = round(dateTaken/60, self.time_data_round)

        self.data.append((time_min, data_at_t))

        if self.im_write:
            write_im = np.hstack(condensed_im)

            path = self.get_pr_path(self.current_index)

            cv2.imwrite(path, write_im)

        return True

    def read_im(self, path):
        return cv2.imread(path)

    def crop_ims(self, image):
    # return list of images centered on well.

        well_ims = []
        for x, y, r in self.wells:

            xs = x + self.offset[0]
            ys = y + self.offset[1]

            sub_im = image[ys-r:ys+r, xs-r:xs+r]

            # imx, imy, z = np.shape(sub_im)

            # for u in range(imx):
            # for v in range(imy):
            # if self.mask[imx*u + v] <= 0 :
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
    # write next line of data to disk. appends to existing files
        next_datum = self.data[-1]

        time = next_datum[0]
        bgr = next_datum[1]
        # dens, color = list(zip(*next_datum))

        # b, g, r = list(zip(*bgr))

        lines = (str(time) + ',' + re.sub(self.s_pattern, '', str(l)) +
                 '\n' for l in list(zip(*bgr)))

        for i in range(len(self.data_files)):
            with open(self.out_dir + self.data_files[i] + '.csv', 'a') as p:
                if i > 0:
                    p.writelines(next(lines))
                else:
                    p.writelines(str(self.current_index) +
                                 ',' + str(time) + '\n')

    def avg_well(self, well_im):
    # return average BGR and standard deviation BGR of provided image including values masked by self.mask    
        (x, y, z) = np.shape(well_im)

        avg = np.zeros((3,))

        flat_im = []

        for u in range(x):
            for v in range(y):
                if self.mask[x*u + v] > 0:
                    flat_im.append(well_im[v][u])

        avg = tuple(np.average(flat_im, axis=0))
        sd = tuple(np.std(flat_im, axis=0))

        # return_data = avg

        # pixel = avg.astype(np.single).reshape((1, 1, 3))
        # density = cv2.cvtColor(pixel, cv2.COLOR_RGB2GRAY)[0, 0]

        # cols = [hex(round(w)) for w in list(avg)]
        # colorhex = '#'+str.upper(cols[0][2:] + cols[1][2:] + cols[2][2:])

        return tuple(avg + sd)
