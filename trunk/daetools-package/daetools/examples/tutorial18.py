#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial18.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2014
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
__doc__ = """This tutorial demonstrates a solution of a discretized population balance using
high resolution upwind schemes with flux limiter.
Source:
Qamar et al. A comparative study of high resolution schemes for solving population balances
in crystallization, Computers and Chemical Engineering, 2006.
It shows a comparison between the analytical results and various discretization schemes:

- I order upwind scheme
- II order upwind scheme
- II order central scheme
- High resolution Van Leer k-interpolation (k = 1/3) upwind scheme with Koren (1993) flux limiter

The problem is from the section 3.1 Size-independent growth. Could be also found in:
Motz et al., Comparison of numerical methods for the simulation of 
dispersed phase systems, Chem. Eng. Sci., 2002

Results:

.. image:: _static/dpb-zoomed.png
   :height: 300px
                         
"""

import sys, numpy
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, mol, kmol, J, um

pbm_number_density_t = daeVariableType("pbm_number_density_t", m**(-1), 0.0, 1.0e+40,  0.0, 1e-0)

# Koren scheme (1993) using k = 1/3 for k-interpolation van Leer interpolation scheme
# Achtung, Achtung!!
# van Leer uses an inverse r in its scheme!!
def r(i, ni, epsilon = 1e-10):
    return (ni(i+1) - ni(i) + Constant(epsilon * m**(-1))) / (ni(i) - ni(i-1) + Constant(epsilon * m**(-1)))

def Phi(r):
    return Max(0, Min(2*r, Min(1.0/3 + 2.0*r/3, 2.0)))

def ni_edge_plus(i, ni, nL):
    if i == 0:      # Right face of the first cell: central interpolation (k=1)
        return 0.5 * (ni(0) + ni(1))
    elif i == nL-1: # Right face of the last cell: one-sided upwind scheme (k=-1)
        return ni(i) + 0.5 * (ni(i) - ni(i-1))
    else:           # Other cells: k=1/3
        return ni(i) + 0.5 * Phi(r(i, ni)) * (ni(i) - ni(i-1))

class modelMoC(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.L  = daeDomain("L",  self, um, "Characteristic dimension (size) of crystals")

        self.ni_an_60      = daeVariable("ni_analytical",  pbm_number_density_t, self, "Analytical solution at t = 60s",            [self.L])
        self.ni_I          = daeVariable("ni_I_upwind",    pbm_number_density_t, self, "I order upwind scheme",                     [self.L])
        self.ni_II         = daeVariable("ni_II_upwind",   pbm_number_density_t, self, "II order upwind scheme",                    [self.L])
        self.ni_II_central = daeVariable("ni_II_central",  pbm_number_density_t, self, "II order ceneter finite difference",        [self.L])
        self.ni_k_int      = daeVariable("ni_VanLeer",     pbm_number_density_t, self, "Van Leer k-interpolation scheme (k = 1/3)", [self.L])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        G  = 1 * um/s
        L  = self.L.Points
        nL = self.L.NumberOfPoints

        # Create an intial CSD array
        self.ni_0 = numpy.zeros(nL)
        for i in range(0, nL):
            if L[i] > 10 and L[i] < 20:
                self.ni_0[i] = 1e10

        # Analytical solution is ni(L,t) = ni_0(L-Gt) so we need to shift an initial CSD for Gt (in this case for 60 seconds)
        # Create shifted CSD array
        ni_60 = numpy.zeros(nL)
        for i in range(0, nL):
            if L[i] > (10 + G.value*60) and L[i] < (20 + G.value*60):
                ni_60[i] = 1e10
        # Actual equations
        for i in range(0, nL):
            eq = self.CreateEquation("ni_an_60(%d)" % i, "")
            eq.Residual = self.ni_an_60(i) - Constant(ni_60[i] * (1/m))

        # I order upwind scheme
        for i in range(0, nL):
            if i == 0:
                eq = self.CreateEquation("ni_I(%d)" % i, "")
                eq.Residual = self.ni_I(0) # Boundary conditions: here G*ni = 0
            else:
                eq = self.CreateEquation("ni_I(%d)" % i, "")
                eq.Residual = self.ni_I.dt(i) + Constant(G) * (self.ni_I(i) - self.ni_I(i-1)) / (self.L(i) - self.L(i-1))

        # II order upwind scheme
        for i in range(0, nL):
            if i == 0:
                eq = self.CreateEquation("ni_II(%d)" % i, "")
                eq.Residual = self.ni_II(0)
            elif i == 1:
                eq = self.CreateEquation("ni_II(%d)" % i, "")
                eq.Residual = self.ni_II.dt(i) + Constant(G) * (self.ni_II(i) - self.ni_II(i-1)) / (self.L(i) - self.L(i-1))
            else:
                eq = self.CreateEquation("ni_II(%d)" % i, "")
                eq.Residual = self.ni_II.dt(i) + Constant(G) * (self.ni_II(i-2) - 4*self.ni_II(i-1) + 3*self.ni_II(i)) / (self.L(i) - self.L(i-2))

        # II order central scheme
        eq = self.CreateEquation("ni_II_central(0)", "")
        eq.Residual = self.ni_II_central(0)

        eq = self.CreateEquation("ni_II_central(n)", "")
        l = eq.DistributeOnDomain(self.L, eOpenClosed)
        eq.Residual = self.ni_II_central.dt(l) + Constant(G) * self.ni_II_central.d(self.L, l)

        # k-interpolation (van Leer 1985)
        for i in range(0, nL):
            if i == 0:
                eq = self.CreateEquation("ni_k_int(%d)" % i, "")
                eq.Residual = self.ni_k_int(0) # Boundary condition: here G*ni = 0
            else:
                eq = self.CreateEquation("ni_k_int(%d)" % i, "")
                eq.Residual = self.ni_k_int.dt(i) + Constant(G) * (ni_edge_plus(i,self.ni_k_int,nL) - ni_edge_plus(i-1,self.ni_k_int,nL)) / (self.L(i) - self.L(i-1))

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modelMoC("tutorial18")

    def SetUpParametersAndDomains(self):
        self.m.L.CreateStructuredGrid(eCFDM, 2, 200, 0, 100)

    def SetUpVariables(self):
        # Initial conditions
        L = self.m.L.Points
        nL = self.m.L.NumberOfPoints

        # Initial conditions (the first item is not differential)
        ni_0 = self.m.ni_0.copy()
        ni_0[0] = None # unset

        self.m.ni_I.SetInitialConditions(ni_0)
        self.m.ni_II.SetInitialConditions(ni_0)
        self.m.ni_II_central.SetInitialConditions(ni_0)
        self.m.ni_k_int.SetInitialConditions(ni_0)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Umfpack", "")
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 5
    sim.TimeHorizon       = 60
    simulator  = daeSimulator(app, simulation=sim, lasolver = lasolver)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Umfpack", "")
    daesolver.SetLASolver(lasolver)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 5
    simulation.TimeHorizon = 60

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

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
