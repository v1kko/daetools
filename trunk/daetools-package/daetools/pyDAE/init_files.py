import struct

def readInitializationFile(initFile):
    res = []
    f = open(initFile, 'rb')
    # Read the header: 4 bytes [Number of variables]
    NoVars, = struct.unpack('I', f.read(4))

    # Read actual data: 8 bytes double values x NoVars
    for i in xrange(NoVars):
        val, = struct.unpack('d', f.read(8))
        res.append(val)
    
    return res