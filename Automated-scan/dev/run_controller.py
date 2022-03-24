from Control import Controller
from Detect import Well_Detector


# WD = Well_Detector()
# Well_Detector.store_well_loc(WD.return_wells('./Automated-scan/dev/scan_19.bmp'))

CT = Controller(start_index=1410,out_folder='../Exp-22-03-ANALYSIS', data_folder='../Exp-22-03')
CT.recover_wells()

CT.run_control(overwrite=True)  
