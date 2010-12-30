from pyCore import *
from pyActivity import *
from pyDataReporting import *
from pyIDAS import *
try:
    from pyBONMIN import *
except ImportError, e:
    print 'Cannot load pyBONMIN module', str(e)

from daetools.daeSimulator import *
from daeLogs import daePythonStdOutLog

#try:
    #from pyCore import *
#except ImportError, e:
    #print 'Cannot load pyCore module', str(e)

#try:
    #from pyActivity import *
#except ImportError, e:
    #print 'Cannot load pyActivity module:', str(e)

#try:
    #from pySolver import *
#except ImportError, e:
    #print 'Cannot load pySolver module', str(e)

#try:
    #from pyDataReporting import *
#except ImportError, e:
    #print 'Cannot load pyDataReporting module', str(e)

#try:
    #from daetools.daeSimulator import *
#except ImportError, e:
    #print 'Cannot load daeSimulator module', str(e)

#try:
    #from daeLogs import daePythonStdOutLog
#except ImportError, e:
    #print 'Cannot load daeLogs module', str(e)


"""
dae_supported_solvers   = []
dae_unsupported_solvers = []

try:
    import pyIntelMKL
    dae_supported_solvers.append('pyIntelMKL')
except ImportError:
    dae_unsupported_solvers.append('pyIntelMKL')

try:
    import pyAmdACML
    dae_supported_solvers.append('pyAmdACML')
except ImportError:
    dae_unsupported_solvers.append('pyAmdACML')

try:
    import pyIntelPardiso
    dae_supported_solvers.append('pyIntelPardiso')
except ImportError:
    dae_unsupported_solvers.append('pyIntelPardiso')

try:
    import pyTrilinosAmesos
    dae_supported_solvers.append('pyTrilinosAmesos')
except ImportError:
    unsupported_solvers.append('pyTrilinosAmesos')

try:
    import pyAtlas
    dae_supported_solvers.append('pyAtlas')
except ImportError:
    unsupported_solvers.append('pyAtlas')

def daeLASolversInfo():
    print "Supported solvers:", dae_supported_solvers
    print "Unsupported solvers:", dae_unsupported_solvers
"""
