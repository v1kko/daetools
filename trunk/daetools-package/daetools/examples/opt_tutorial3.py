#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            opt_tutorial3.py
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
************************************************************************************
"""
__doc__ = """
This tutorial introduces NLOPT NLP solver, its setup and options.
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.nlopt import pyNLOPT

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x1 = daeVariable("x1", no_t, self)
        self.x2 = daeVariable("x2", no_t, self)
        self.x3 = daeVariable("x3", no_t, self)
        self.x4 = daeVariable("x4", no_t, self)

        self.dummy = daeVariable("dummy", no_t, self, "A dummy variable to satisfy the condition that there should be at least one-state variable and one equation in a model")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("Dummy")
        eq.Residual = self.dummy()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial3")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.x1.AssignValue(1)
        self.m.x2.AssignValue(5)
        self.m.x3.AssignValue(5)
        self.m.x4.AssignValue(1)

    def SetUpOptimization(self):
        # Set the objective function (min)
        self.ObjectiveFunction.Residual = self.m.x1() * self.m.x4() * (self.m.x1() + self.m.x2() + self.m.x3()) + self.m.x3()

        # Set the constraints (inequality, equality)
        # Constraints are in the following form:
        #  - Inequality: g(i) <= 0
        #  - Equality: h(i) = 0
        c1 = self.CreateInequalityConstraint("Constraint 1") # g(x) >= 25:  25 - x1*x2*x3*x4 <= 0
        c1.Residual = 25 - self.m.x1() * self.m.x2() * self.m.x3() * self.m.x4()

        c2 = self.CreateEqualityConstraint("Constraint 2") # h(x) == 40
        c2.Residual = self.m.x1() * self.m.x1() + self.m.x2() * self.m.x2() + self.m.x3() * self.m.x3() + self.m.x4() * self.m.x4() - 40

        # Set the optimization variables, their lower/upper bounds and the starting point
        self.SetContinuousOptimizationVariable(self.m.x1, 1, 5, 2);
        self.SetContinuousOptimizationVariable(self.m.x2, 1, 5, 2);
        self.SetContinuousOptimizationVariable(self.m.x3, 1, 5, 2);
        self.SetContinuousOptimizationVariable(self.m.x4, 1, 5, 2);

def chooseAlgorithm():
    from PyQt5 import QtCore, QtGui, QtWidgets
    algorithms = ['NLOPT_GN_DIRECT','NLOPT_GN_DIRECT_L','NLOPT_GN_DIRECT_L_RAND','NLOPT_GN_DIRECT_NOSCAL','NLOPT_GN_DIRECT_L_NOSCAL',
                  'NLOPT_GN_DIRECT_L_RAND_NOSCAL','NLOPT_GN_ORIG_DIRECT','NLOPT_GN_ORIG_DIRECT_L','NLOPT_GD_STOGO','NLOPT_GD_STOGO_RAND',
                  'NLOPT_LD_LBFGS_NOCEDAL','NLOPT_LD_LBFGS','NLOPT_LN_PRAXIS','NLOPT_LD_VAR1','NLOPT_LD_VAR2','NLOPT_LD_TNEWTON',
                  'NLOPT_LD_TNEWTON_RESTART','NLOPT_LD_TNEWTON_PRECOND','NLOPT_LD_TNEWTON_PRECOND_RESTART','NLOPT_GN_CRS2_LM',
                  'NLOPT_GN_MLSL','NLOPT_GD_MLSL','NLOPT_GN_MLSL_LDS','NLOPT_GD_MLSL_LDS','NLOPT_LD_MMA','NLOPT_LN_COBYLA',
                  'NLOPT_LN_NEWUOA','NLOPT_LN_NEWUOA_BOUND','NLOPT_LN_NELDERMEAD','NLOPT_LN_SBPLX','NLOPT_LN_AUGLAG','NLOPT_LD_AUGLAG',
                  'NLOPT_LN_AUGLAG_EQ','NLOPT_LD_AUGLAG_EQ','NLOPT_LN_BOBYQA','NLOPT_GN_ISRES',
                  'NLOPT_AUGLAG','NLOPT_AUGLAG_EQ','NLOPT_G_MLSL','NLOPT_G_MLSL_LDS','NLOPT_LD_SLSQP']
    # Show the input box to choose the algorithm (the default is len(algorithms)-1 that is: NLOPT_LD_SLSQP)
    algorithm, ok = QtWidgets.QInputDialog.getItem(None, "NLOPT Algorithm", "Choose the NLOPT algorithm:", algorithms, len(algorithms)-1, False)
    if ok:
        return str(algorithm)
    else:
        return 'NLOPT_LD_SLSQP'

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    opt = daeOptimization()
    algorithm = chooseAlgorithm()
    nlp = pyNLOPT.daeNLOPT(algorithm)
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 1
    sim.TimeHorizon       = 5
    simulator = daeSimulator(app, simulation = sim,
                                  optimization = opt,
                                  nlpsolver = nlp)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()
    optimization = daeOptimization()

    # NLOPT algorithm must be set in its constructor
    nlpsolver = pyNLOPT.daeNLOPT('NLOPT_LD_SLSQP')

    # Do no print progress
    log.PrintProgress = False
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 5

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    optimization.Initialize(simulation, nlpsolver, daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Run
    optimization.Run()
    optimization.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
