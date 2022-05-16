from libs.Control import Controller

CT = Controller(start_index=0,out_folder='../Unit tests/Exp-C ALS p2', data_folder='../Unit tests/Exp-C/Part B')
CT.recover_wells()
CT.process_loop(overwrite=True, offline=True)  
