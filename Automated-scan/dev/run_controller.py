from Control import Controller
from Detect import Well_Detector


# WD = Well_Detector()
# Well_Detector.store_well_loc(WD.return_wells('./Automated-scan/dev/scan_44.bmp'))

CT = Controller()
CT.recover_wells()

CT.next_scan(abort=False)  