from Control import Controller
from Detect import Well_Detector


# WD = Well_Detector()
# Well_Detector.store_well_loc(WD.return_wells('./Automated-scan/dev/scan_19.bmp'))

CT = Controller(start_index=0,out_folder='../Unit tests/TEST', data_folder='../Unit tests/Exp-03-14')
CT.recover_wells()
CT.process_loop(overwrite=True)  
