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

# This function can be used by daetools_mex, daetools_s and daetools_fmi_cs to load a simulation.
def create_simulation_for_cosimulation(simulation_class, relativeTolerance = 0.0, calculateSensitivities = False):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeNoOpDataReporter()
    simulation   = simulation_class()
    
    # Set the relative tolerance (may be changed in fmi2SetupExperiment)
    if relativeTolerance > 0:
        daesolver.RelativeTolerance = relativeTolerance
    
    from daetools.solvers.superlu import pySuperLU
    lasolver = pySuperLU.daeCreateSuperLUSolver()
    daesolver.SetLASolver(lasolver)
        
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the default time horizon and the reporting interval (will be changed in fmi2SetupExperiment)
    simulation.ReportingInterval = 1.0
    simulation.TimeHorizon       = 1.0

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = calculateSensitivities)

    # Nota bene: store the objects since they will be destroyed when they go out of scope
    simulation.__rt_objects__ = [daesolver, datareporter, log, lasolver]

    return simulation
