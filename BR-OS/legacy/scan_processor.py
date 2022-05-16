from typing import Tuple
from time import sleep, localtime

import cv2
from image_process import im_process

import csv
from os.path import exists
from os import system
import json
# 64 bit process

# ready flag contains byte indicating whether scan image is ready to process (0 no, 1 yes) required for cross-process communication
# setup flag contains byte to request scan process to start scanning

# class to receive scans and process the data


class sc_processor():

    wells = []
    avg_col_time = []
    count = 0
    exp_max = 10

    def sc_processor(self):
        with open('./Automated-scan/settings.json') as json_file:
            settings = json.load(json_file)
        if settings:
            self.exp_max = settings["exp_max"]


    def check_scanner(self):
        im = None

        z = False
        with open('./Automated-scan/bin/scan_ready.bin', 'rb') as x:
            z = x.read()

        ready_flag = int(z[-1])

        if ready_flag:
            im = cv2.imread('./Scans/scan001.png')

            with open('./Automated-scan/bin/scan_ready.bin', 'wb') as x:
                x.write(bytes([0b0]))

            return True, im
        else:
            return False, None

    def poll_scanner(self):
        im_received = False
        while(~im_received):
            im_received, im = self.check_scanner()
            sleep(1)

        self.process_image(self, im)
        self.count += 1

        # break scan at 
        if self.count < self.exp_max:
            self.poll_scanner()
        else:
            self.record_data()
            sc_processor.command_stop()

    def process_image(self, im):
        well_ims = im_process.extract_well_ims(self.wells, im)

        avg_col = []

        for well_im in well_ims:
            avg_col.append(sc_processor.avg_image(well_im))

        self.avg_col_time.append((avg_col, localtime()))

        # stack and write image?

    def avg_image(im):
        return cv2.mean(im)

    def record_data(self):
        with open('./Experiment-data/exp1.csv', 'w') as csv_file:
            fieldnames = ['time'].append([i for i in range(self.wells.len())])
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for col_ss, time in self.avg_col_time:
                writer.writerow([time, col_ss])

    def setup_scan(self):
        # locate wells from first scan, if does not exist, instruct scanner to make it

        if ~exists('./Scans/scan_setup.bmp'):
            with open('./Automated-scan/bin/setup_flag.bin', 'wb') as x:
                x.write(bytes([0b1]))
            sleep(1)  # time out 1 seconds
            self.setup_scan()
        else:
            im_process.locate_wells('./Scans/scan_setup.bmp')
            self.wells = im_process.read_well_loc()

    def command_stop():
        with open('./Automated-scan/bin/stop_scan.bin', 'wb') as x:
                x.write(bytes([0b1]))

def main():

    # launch scan process
    system('python-32 scanner.py')

    scan_processor = sc_processor()
    scan_processor.setup_scan()

    print('starting experiment, %d mins'.format(scan_processor.exp_max))

    scan_processor.poll_scanner()

    print('experiment complete')