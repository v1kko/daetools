#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                         tutorial_deal_II.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2013
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
__doc__ = """
"""

import sys, numpy, json
from daetools.pyDAE import *
from daetools.solvers.deal_II import pyDealII
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.fe = pyDealII.daeConvectionDiffusion_2D('Helmholtz', self, 'Modified deal.II step-7 example (steady-state Helmholtz equation)')
        options = {}
        options["meshFilename"]    = "step-7.msh"
        options["polynomialOrder"] = 1
        options["diffusivity"]     = 1.0 
        options["velocity"]        = []
        options["generation"]      = 1.0
        options["dirichletBC"]     = {0 : 1.0},
        options["neumannBC"]       = {1 : 0.5}
        
        jsonInit = json.dumps(options)
        print jsonInit
        self.fe.InitializeModel(jsonInit)
        
    def InitializeModel(self, jsonInit):
        print 'modTutorial.InitializeModel:', jsonInit
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_deal_II")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        pass
    
# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    simulation   = simTutorial()
    datareporter = simulation.m.CreateDataReporter('uja') #daeTCPIPDataReporter()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 1000

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.fe.SaveModelReport(simulation.m.fe.Name + ".xml")
    simulation.m.fe.SaveRuntimeModelReport(simulation.m.fe.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
