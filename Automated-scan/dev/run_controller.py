from Control import Controller
from Detect import Well_Detector


# WD = Well_Detector()
# Well_Detector.store_well_loc(WD.return_wells('./Automated-scan/dev/scan_19.bmp'))

CT = Controller(start_index=6,out_folder='Experiment-processed', data_folder='Experiment-data')
CT.recover_wells()

CT.run_control(overwrite=True)  
