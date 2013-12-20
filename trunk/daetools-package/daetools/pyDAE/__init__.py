import sys

python_major = sys.version_info[0]
python_minor = sys.version_info[1]

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
from .logs import daePythonStdOutLog
from .variable_types import *
    
try:
    from daetools.dae_simulator.simulator import daeSimulator
except ImportError as e:
    print('Cannot import daeSimulator module. Error: {0}'.format(str(e)))
