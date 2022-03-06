from time import sleep, time
import cv2
import exifread
from Detect import Well_Detector
import numpy as np


class Controller:

    current_index = 0
    inpath = './Experiment-data/scan_'
    poll_time = 10
    wells = None
    data = []

    def next_scan(self, abort=False):

        path_file = self.inpath + str(self.current_index) + '.bmp'

        exists = False
        dateTaken = None

        # with open(path_file, 'rb') as fh:
        #     tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal")
        #     if tags:
        #         dateTaken = tags["EXIF DateTimeOriginal"]

        #     exists = True  # is this safe??? TODOs

        im = self.read_im(path_file)

        t_start = time()

        if im is not None:
            self.process_scan(im, dateTaken)
            self.current_index += 1

        t_delta = time() - t_start

        if (t_delta < self.poll_time):
            sleep(self.poll_time - t_delta)

        if not abort:
            self.next_scan()

    def recover_wells(self, im_path):
        WF = Well_Detector()
        self.wells = WF.return_wells(im_path)

    def process_scan(self, im, dateTaken):
        condensed_im = Controller.crop_ims(self.wells, im)
        densities = [Controller.avg_well(i) for i in condensed_im]
        self.data.append((densities, dateTaken))

        write_im = np.hstack(condensed_im)
        # write to file
        cv2.imwrite('./Experiment-processed/wells_' +
                    str(self.current_index) + '.png', write_im)

    def read_im(self, path):
        return cv2.imread(path)

    def crop_ims(wells, image):

        well_ims = []

        for x, y, r in wells:

            sub_im = image[y-r:y+r, x-r:x+r]

            if sub_im.any():
                well_ims.append(sub_im)

        return well_ims

    def avg_well(well_im):

        return np.average(well_im)
