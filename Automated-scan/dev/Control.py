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

    radial_amount = 0.5  # [%]
    mask = []

    R = 38

    def __init__(self):
        self.mask = self.compute_mask()

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

    def recover_wells(self):
        self.wells = Well_Detector.read_well_loc()

    def process_scan(self, im, dateTaken):
        condensed_im = Controller.crop_ims(self.wells, im)
        densities = [self.avg_well(i) for i in condensed_im]
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

    def compute_mask(self):

        width = 2*self.R
        mask = np.zeros((width, width))

        mask = []

        for u in range(width):
            for v in range(width):
                mask.append(1 if (u - self.R)**2 + (v - self.R)**2 < (self.R*self.radial_amount)**2 else 0)
        f = sum(mask)

        return np.array(mask)/f

    def avg_well(self, well_im):
        (x,y,z) = np.shape(well_im)


        avg = np.zeros((3,))
        for u in range(x):
            for v in range(y):
                avg += well_im[u][v] * self.mask[x*u + v]

        density = round((0.2989*avg[0]+ 0.5870*avg[1] + 0.1140*avg[2])/255, 6)

        cols = [ hex(round(w)) for w in list(avg)]
        hex_code = cols[0] + cols[1][2:] + cols[2][2:]

        return (density, hex_code)
