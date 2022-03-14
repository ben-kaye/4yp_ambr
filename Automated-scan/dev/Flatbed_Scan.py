from time import sleep, time
import twain
import Util
import os


class SC:
    scan_index = 0  # current scan index
    # out_path = './Automated-scan/Scans/'
    out_path = './Experiment-data/'
    name_conv = 'scan_'
    filetype = '.bmp'
    wait_time = 60  # seconds
    sc_name = b'EPSON Perfection V200'
    # frame = (3.1, 4, 5.8, 6.8) #

    frame = (3.1, 4.8, 5.8, 7.5)
    # frame = (0, 0, 8, 11) # A4 FRAME
    dpi = 300

    max_scans = 1440

    def __init__(self, t_scan=60):
        if t_scan < 20:
            print('Warning: scans take between 10 & 15s')
        self.wait_time = t_scan

    def take_scan(self):
        success = False
        # next_index = self.scan_index + 1

        start_time = time()

        try:
            result = twain.acquire(self.get_filename(
            ), ds_name=self.sc_name, dpi=self.dpi, frame=self.frame, pixel_type='color')

            if result is not None:
                success = True
        except Exception as e:
            print(e)

        if success:
            self.scan_index += 1
        else:

            delta_time = time() - start_time
            remainder = max(self.wait_time - delta_time, 0)
            sleep(min(self.wait_time, remainder))

            self.take_scan()
            print('failure')

    def scan_schedule(self):

        for i in range(self.max_scans):

            if SC.check_stop() or self.scan_index > self.max_scans:
                return 0

            start_time = time()

            self.take_scan()

            delta_time =  time() - start_time

            # print(delta_time)
            
            remainder = max(self.wait_time - delta_time, 0)
            sleep(min(self.wait_time, remainder))

    def get_filename(self):
        return self.out_path + self.name_conv + str(self.scan_index) + self.filetype

    def check_stop():
        stop_scan = Util.bin_read('stop_scan')
        if stop_scan:
            Util.bin_write('stop_scan', False)
        return stop_scan
