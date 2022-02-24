from time import sleep, time
import cv2
import exifread


class Controller:

    current_index = 0
    inpath = './../Scans/scan_'
    poll_time = 10
    circles = None
    data = []

    def next_scan(self):

        path_file = self.inpath + self.current_index + '.bmp'

        exists = False
        dateTaken = None

        with open(path_file, 'rb') as fh:
            tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal")
            dateTaken = tags["EXIF DateTimeOriginal"]

            exists = True # is this safe??? TODOs

        im = self.read_im()

        t_start = time()

        if im is not None:
            self.process_scan(im, dateTaken)
            self.current_index += 1

        t_delta = time() - t_start

        if (t_delta < self.poll_time):
            sleep(self.poll_time - t_delta)
        self.next_scan()

    def process_scan(self, im, dateTaken):
        condensed_im = Controller.crop_ims(self.circles, im)
        densities = [ Controller.avg_well(i) for i in condensed_im ]
        self.data.append((densities, dateTaken))

        write_im = cv2.hcat()
        # write to file
        cv2.imwrite('./../State/wells_' + self.current_index + '.png', write_im)

    def read_im(path):
        return None

    def crop_ims(circles, image):

        well_ims = []

        for x, y, r in circles:
            well_ims.append(image[x-r:x+r, y-r:y+r])

        return well_ims

    def avg_well(well_im):
        return CV2.average(well_im)
