"""********************************************************************************
                               __init__.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
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
from .simulation_loader_aux import *
from .thermo_packages import *
from .hr_upwind_scheme import daeHRUpwindSchemeEquation

def daeCreateQtApplication(sys_argv):
    # An auxiliary function to create a qt application with no reference to the qt version
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys_argv)
    return app

def daeQtMessage(windowTitle, message):
    # An auxiliary function to show a message box with no reference to the qt version
    try:
        from PyQt5 import QtCore, QtWidgets
        app = QtCore.QCoreApplication.instance()
        if not app:
            app = daeCreateQtApplication(sys.argv)
        QtWidgets.QMessageBox.warning(None, windowTitle, message)
    except Exception as e:
        print(str(e))

try:
    from daetools.dae_simulator.simulator            import daeSimulator
    from daetools.dae_simulator.simulation_explorer  import daeSimulationExplorer
    from daetools.dae_simulator.simulation_inspector import daeSimulationInspector
except ImportError as e:
    print('Cannot import simulator modules. Error: {0}'.format(str(e)))

try:
    from daetools.dae_simulator.activity import daeActivity
except ImportError as e:
    print('Cannot import daeActivity module. Error: {0}'.format(str(e)))
