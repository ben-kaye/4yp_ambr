import json

def load_settings():
    settings = {}
    with open('./Automated-scan/settings.json') as json_file:
        settings = json.load(json_file)
    return settings

def bin_read(file):
    res = None
    with open('./Automated-scan/bin/'+file+'.bin', 'rb') as x:
        res = x.read()
    return res

def bin_write(file, val):
    if val:
        write = bytes([0b1])
    else:
        write = bytes([0b0])
    with open('./Automated-scan/bin/'+file+'.bin', 'wb') as g:
        g.write(write)
