import twain
import sys 
# example automated scan on Perfection V200

def scan():
    try:
        result = twain.acquire(outpath,
                                ds_name=name_bin,
                               dpi=300,
                               frame=(0, 0, 8.17551, 11.45438), # A4
                               pixel_type='color'
                               )
        
    except:
        # scanner_source.close()
        sys.exit(1)
        

    else:
        sys.exit(0 if result else 1)

dsm = twain.SourceManager(0)

name_bin = b'EPSON Perfection V200'

# scanner_source = dsm.open_source(b'EPSON Perfection V200')


# scanner_source.set

# scanner_source.RequestAcquire(0,0)
# rv = scanner_source.XferImageNatively()
# if rv:
#     (handle, count) = rv
#     twain.DIBToBMFile(handle, 'scan_01.bmp')

# twain.acquire('scan.bmp',ds_name=b'EPSON Perfection V200',dpi=300)

# scanner_source.close()

outpath = 'scan.bmp'

scan()



