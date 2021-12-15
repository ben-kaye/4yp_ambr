import twain
from time import sleep

# 32 bit process

class sc_process():

    source_name = b'EPSON Perfection V200'
    dsm = twain.SourceManager(0)
    output_path = './Scans/'
    wait_time = 60  # secs

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
        ready_flag = open('./Automated-scan/scan_ready.bin', 'wb')
        ready_flag.write(bytes([0b1]))
        ready_flag.close()

    def scan_schedule(self):

        self.acquire_scan('scan001.bmp')

        sleep(self.wait_time)

        self.scan_schedule()

    def set_up_wells(self):

        file_name = 'scan_setup.bmp'

        self.acquire_scan(file_name)

