"""********************************************************************************
                               __init__.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2016
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
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
    from daetools.dae_simulator.simulation_explorer import daeSimulationExplorer
    from daetools.dae_simulator.simulation_inspector import daeSimulationInspector
except ImportError as e:
    print('Cannot import daeSimulator module. Error: {0}'.format(str(e)))
