#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial18.py
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
This tutorial introduces several new concepts:


"""

import sys, numpy
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, m, "X axis domain")
        self.y = daeDomain("y", self, m, "Y axis domain")

        self.Qb = daeParameter("Q_b",         W/(m**2), self, "Heat flux at the bottom edge of the plate")
        self.Qt = daeParameter("Q_t",         W/(m**2), self, "Heat flux at the top edge of the plate")
        self.ro = daeParameter("&rho;",      kg/(m**3), self, "Density of the plate")
        self.cp = daeParameter("c_p",         J/(kg*K), self, "Specific heat capacity of the plate")
        self.k  = daeParameter("&lambda;_p",   W/(m*K), self, "Thermal conductivity of the plate")
       
        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate", [self.x, self.y])
        
        self.Tsum = daeVariable("Tsum", temperature_t, self, "Sum of all plate temperatures")
        self.Tave = daeVariable("Tave", temperature_t, self, "Average temperature of the plate")
        self.Tstd = daeVariable("Tstd", temperature_t, self, "Standard deviation of the temperature")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.ro() * self.cp() * self.T.dt(x, y) - self.k() * \
                     (self.T.d2(self.x, x, y) + self.T.d2(self.y, x, y))

        eq = self.CreateEquation("BC_bottom", "Boundary conditions for the bottom edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = - self.k() * self.T.d(self.y, x, y) - self.Qb()

        eq = self.CreateEquation("BC_top", "Boundary conditions for the top edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = - self.k() * self.T.d(self.y, x, y) - self.Qt()

        eq = self.CreateEquation("BC_left", "Boundary conditions at the left edge")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

        eq = self.CreateEquation("BC_right", "Boundary conditions for the right edge")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)
        
        numpy.set_printoptions(linewidth=1e10)
        Nx = self.x.NumberOfPoints
        Ny = self.y.NumberOfPoints
        
        Tndarray = numpy.empty((Nx,Ny), dtype=object)
        for x in range(Nx):
            for y in range(Ny):
                Tndarray[x,y] = self.T(x,y)
        #print 'T as Numpy array:'
        #print Tndarray
        
        
        Tmatrix = numpy.matrix(Tndarray, dtype=object)
        print('T as Numpy matrix:')
        print(Tmatrix)

        #print 'T matrix transposed:'
        #print Tmatrix.transpose()
        
        b = numpy.zeros(Ny)
        b[:] = [i for i in range(Ny)]
        print(b)
        
        print(Tndarray * b)
        
        a = numpy.eye(2)
        print(numpy.outer(Tndarray, b))
        
        #print 'Sum of all temperatures:'
        #print Tmatrix.sum()
        eq = self.CreateEquation("Tsum", "Sum of all temperatures")
        eq.BuildJacobianExpressions = True
        eq.Residual = self.Tsum() - Tmatrix.sum()
        
        #print 'Average temperature:'
        #print Tmatrix.sum()/(Nx*Ny)
        eq = self.CreateEquation("Tave", "Average temperature")
        eq.BuildJacobianExpressions = True
        eq.Residual = self.Tave() - Tmatrix.sum()/(Nx*Ny)
        
        #print 'Standard deviation for T:'
        #print Tmatrix.std()
        eq = self.CreateEquation("Tstd", "Standard deviation")
        #eq.BuildJacobianExpressions = True
        eq.Residual = self.Tstd() - Tmatrix.std()
        
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial18")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(eCFDM, 2, 4, 0, 0.1)
        self.m.y.CreateStructuredGrid(eCFDM, 2, 4, 0, 0.1)

        self.m.k.SetValue(401 * W/(m*K))
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.ro.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1e6 * W/(m**2))
        self.m.Qt.SetValue(0 * W/(m**2))

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300 * K)

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
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

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
