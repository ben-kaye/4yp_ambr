import twain

# example automated scan on Perfection V200

dsm = twain.SourceManager(0)
scanner_source = dsm.open_source(b'EPSON Perfection V200')


# scanner_source.set

scanner_source.RequestAcquire(0,0)
rv = scanner_source.XferImageNatively()
if rv:
    (handle, count) = rv
    twain.DIBToBMFile(handle, 'scan_01.bmp')

scanner_source.close()