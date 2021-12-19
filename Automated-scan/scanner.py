import twain
from time import sleep
import json


# 32 bit process

# TODO set the file to something else:


class sc_process():

    # default params: overidden with settings.json
    source_name = b'EPSON Perfection V200'
    dsm = twain.SourceManager(0)
    output_path = './Scans/'
    scan_interval = 60  # secs

    def sc_process(self):
        with open('./Automated-scan/settings.json') as json_file:
            settings = json.load(json_file)
        if settings:
            self.scan_interval = settings["scan_interval"]

    def acquire_scan(self, im_name):
        scanner_source = self.dsm.open_source(self.source_name)
        scanner_source.RequestAcquire(0, 0)  # request acquire without UI
        rv = scanner_source.XferImageNatively()
        if rv:
            file = self.output_path + im_name

            (handle, count) = rv
            twain.DIBToBMFile(handle, file)
        scanner_source.close()

        # write to ready flag
        with open('./Automated-scan/bin/scan_ready.bin', 'wb') as ready_flag:
            ready_flag.write(bytes([0b1]))

    def scan_schedule(self):

        self.acquire_scan('scan001.bmp')

        sleep(self.scan_interval)

        # break exec if stop_flag is true
        with open('./Automated-scan/bin/stop_scan', 'rb') as x:
            stop_scan = x.read()
            if stop_scan:
                with open('./Automated-scan/bin/stop_scan', 'wb') as g:
                    g.write(bytes([0b0]))
                return

        self.scan_schedule()

    def start_scan_loop(self):
        # set stop flag to false
        with open('./Automated-scan/bin/stop_scan', 'wb') as g:
            g.write(bytes([0b0]))
        self.scan_schedule()

    def acquire_calibration_scan(self):

        file_name = 'scan_setup.bmp'

        self.acquire_scan(file_name)


# command scanner to start processing
def main():
    scanner = sc_process()
    scanner.acquire_calibration_scan()
    scanner.start_scan_loop()
