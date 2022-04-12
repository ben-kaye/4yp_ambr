from Control import Controller
from Detect import Well_Detector


# WD = Well_Detector()
# Well_Detector.store_well_loc(WD.return_wells('./Automated-scan/dev/scan_19.bmp'))

CT = Controller(start_index=0,out_folder='../Unit tests/Exp-22-03-ANALYSIS', alt_out_folder='../Unit tests/Exp-22-03-ANALYSIS2')

CT.recover_cropped() 
