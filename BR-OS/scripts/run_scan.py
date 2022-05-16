from libs.Flatbed_Scan import SC

import os

scanner = SC(overwrite=False,out_dir='./../Exp-22-03/')

scanner.scan_schedule()
