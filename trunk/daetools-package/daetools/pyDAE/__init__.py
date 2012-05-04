import sys, platform

PYTHON_MAJOR = str(sys.version_info[0])
PYTHON_MINOR = str(sys.version_info[1])

# System := {'Linux', 'Windows', 'Darwin'}
DAE_SYSTEM   = str(platform.system())

# Machine := {'i386', ..., 'i686', 'AMD64'}
DAE_MACHINE  = str(platform.machine())

TAG = '_{0}_{1}_py{2}{3}'.format(DAE_SYSTEM, DAE_MACHINE, PYTHON_MAJOR, PYTHON_MINOR)

# Platform-dependent modules
pyUnits         = __import__('pyUnits%s' % TAG,         globals(), locals(), ['base_unit', 'unit', 'quantity'], -1)
pyCore          = __import__('pyCore%s' % TAG,          globals(), locals(), ['*'], -1)
pyActivity      = __import__('pyActivity%s' % TAG,      globals(), locals(), ['*'], -1)
pyDataReporting = __import__('pyDataReporting%s' % TAG, globals(), locals(), ['*'], -1)
pyIDAS          = __import__('pyIDAS%s' % TAG,          globals(), locals(), ['*'], -1)

#from pyUnits import base_unit, unit, quantity
#from pyCore import *
#from pyActivity import *
#from pyDataReporting import *
#from pyIDAS import *

# Platform-independent modules
from daeLogs import daePythonStdOutLog
from daeVariableTypes import *

try:
    from daetools.daeSimulator import *
except ImportError as e:
    print('Cannot import daeSimulator module. Error: {0}'.format(str(e)))
