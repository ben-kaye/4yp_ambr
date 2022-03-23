from time import sleep, time
import twain
import Util
import os
import re

class SC:
    scan_index = 0  # current scan index
    # out_path = './Automated-scan/Scans/'
    out_path = './Experiment-data/'
    name_conv = 'scan_'
    filetype = '.bmp'
    wait_time = 60  # secondsx
    sc_name = b'EPSON Perfection V200'
    # frame = (3.1, 4, 5.8, 6.8) #

    #(3.0, 4, 5.4, 7.5)


    
    x0 = 2.9
    y0 = 5.1
    frame = (x0, y0, x0+2.6, y0+2.6) # x0 y0, x1 y1 from top left
    # frame = (0, 0, 8, 11) # A4 FRAME
    dpi = 300

    max_scans = 2*1440

    def __init__(self, overwrite = False, t_scan=60, start_index=0, out_dir='./Experiment-data/'):

        if not overwrite:
            max_index = -1
            expr = re.compile(r'\d+')

            for (dirpath, dirnames, filenames) in os.walk(out_dir):

                for f in filenames:
                    res = int(expr.search(f).group(0))
                    if res > max_index:
                        max_index = res

                # indices.extend(ints)
                break
            self.scan_index = max_index + 1
            if self.scan_index != start_index:
                print('WARNING starting at ' + str(self.scan_index) +', NOT '+str(start_index))
            
        else: 
            self.scan_index = start_index
        

        self.out_path = out_dir
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
