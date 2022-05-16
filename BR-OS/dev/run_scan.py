import Flatbed_Scan

import os

scanner = Flatbed_Scan.SC(overwrite=False,out_dir='./../Exp-22-03/')

scanner.scan_schedule()
