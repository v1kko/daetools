# Platform-dependent extension modules
import pyCore
import pyActivity
import pyDataReporting
import pyIDAS
import pyUnits

from pyUnits import base_unit, unit, quantity
from pyCore import *
from pyActivity import *
from pyDataReporting import *
from pyIDAS import *

# Platform-independent modules
from daeLogs import daePythonStdOutLog
from daeVariableTypes import *

#from daeModelicaExport import daeModelicaExport

try:
    from daetools.daeSimulator import *
except ImportError as e:
    print('Cannot import daeSimulator module. Error: {0}'.format(str(e)))
