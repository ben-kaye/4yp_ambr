from libs.Control import Controller

CT = Controller(start_index=0,out_folder='../Unit tests/TEST', data_folder='../Unit tests/Exp-03-14')
CT.recover_wells()
CT.run_online_controller() 