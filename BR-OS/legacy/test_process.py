from typing import Tuple
from time import sleep

pass


def check_scanner():
    im = None

    x = open('./Automated-scan/test.bin', 'rb')
    z = x.read()
    x.close()

    ready_flag = int(z[-1])

    if ready_flag:
        # TODO retrive im
        # im =

        x = open('./Automated-scan/test.bin', 'wb')
        x.write(bytes([0b0]))
        x.close()

        return True, im
    else:
        return False, None


def poll_scanner():
    im_received = False
    while(~im_received):
        im_received, im = check_scanner()
        sleep(1)

    # TODO do stuff with picture

    poll_scanner()
