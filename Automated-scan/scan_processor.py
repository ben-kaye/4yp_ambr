from typing import Tuple
from time import sleep, localtime

from cv2 import mean
import image_process

import csv
from os.path import exists
# 64 bit process


class sc_processor():

    wells = []
    avg_col_time = []
    count = 0
    exp_max = 10

    def check_scanner(self):
        im = None

        x = open('./Automated-scan/scan_ready.bin', 'rb')
        z = x.read()
        x.close()

        ready_flag = int(z[-1])

        if ready_flag:
            # TODO retrive im
            # im =

            x = open('./Automated-scan/scan_ready.bin', 'wb')
            x.write(bytes([0b0]))
            x.close()

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

        if self.count < self.exp_max:
            self.poll_scanner()
        else:
            self.record_data()

    def process_image(self, im):
        well_ims = image_process.im_process.extract_well_ims(self.wells, im)

        avg_col = []

        for well_im in well_ims:
            avg_col.append(sc_processor.avg_image(well_im))

        self.avg_col_time.append((avg_col, localtime()))

        # stack and write image?

    def avg_image(im):
        return mean(im)

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
            with open('./Automated-scan/setup_flag.bin', 'wb') as x:
                x.write(bytes([0b1]))
            sleep(1)  # time out 1 seconds
            self.setup_scan()
        else:
            image_process.im_process.locate_wells('./Scans/scan_setup.bmp')
            self.wells = image_process.im_process.read_well_loc()
