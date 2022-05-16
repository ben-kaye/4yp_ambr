from time import time
from Detect import Well_Detector
# import time

WD = Well_Detector(write_ims=False, debugging=True)
# tick = time()
# Well_Detector.store_well_loc(WD.return_wells('./Automated-scan/dev/scan_58.bmp'))
wells = WD.return_wells('../Unit tests/Exp-C/Part A/scan_121.bmp')

print('accept locations? type 0')
inp = input()
if inp == '0':
    Well_Detector.store_well_loc(wells)

# print(time()-tick)