import twain

# example automated scan on Perfection V200

dsm = twain.SourceManager(0)
ss = dsm.open_source(b'EPSON Perfection V200')

ss.RequestAcquire(0,0)
rv = ss.XferImageNatively()
if rv:
    (handle, count) = rv
    twain.DIBToBMFile(handle, 'scan_01.bmp')

ss.close()