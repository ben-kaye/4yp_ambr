from time import sleep
import twain

import os

class SC:
    scan_index = 0  # current scan index
    # out_path = './Automated-scan/Scans/'
    out_path = 'Experiment-data/'
    name_conv = 'scan_'
    filetype = '.bmp'
    wait_time = 60  # seconds
    sc_name = b'EPSON Perfection V200'
    frame = (0, 0, 8.17551, 11.45438)
    dpi = 300

    def take_scan(self):
        success = False
        # next_index = self.scan_index + 1

        try:
            result = twain.acquire(self.get_filename(), ds_name=self.sc_name, dpi=self.dpi, frame=self.frame, pixel_type='color')

            if result is not None:
                success = True
        except Exception as e:
            print(e)

        if success:
            self.scan_index += 1
        else:
            # try and diagnose problem?
            print('failure')

        

    def scan_schedule(self):
        if (SC.check_stop()):
            return 0
        
        self.take_scan()
        sleep(self.wait_time)
        self.scan_schedule()
        

    def get_filename(self):
        return self.out_path + self.name_conv + str(self.scan_index) + self.filetype

    def check_stop():
        stop_scan = False
        with open('./Automated-scan/bin/stop_scan.bin', 'rb') as x:
            stop_scan = x.read()
        if stop_scan:
            with open('./Automated-scan/bin/stop_scan.bin', 'wb') as g:
                g.write(bytes([0b0]))
        return stop_scan
