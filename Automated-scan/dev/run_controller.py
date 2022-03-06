from Control import Controller

CT = Controller()
CT.recover_wells('./Automated-scan/dev/scan_44.bmp')
CT.next_scan(abort=False)  