
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_cv_6.py
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
Code verification using the Method of Exact Solutions.

Reference (section 3.3):

- B. Koren. A robust upwind discretization method for advection, diffusion and source terms. 
  Department of Numerical Mathematics. Report NM-R9308 (1993). 
  `PDF <http://oai.cwi.nl/oai/asset/5293/05293D.pdf>`_
  
The problem in this tutorial is 1D *transient convection-diffusion* equation:
    
.. code-block:: none

   dc_dt + u*dc/dx - D*d2c/dc2 = 0

The equation is solved using the high resolution cell-centered finite volume upwind scheme
with flux limiter described in the article.
  
Numerical vs. exact solution plots (Nx = [20, 40, 80]):

.. image:: _static/tutorial_cv_6-results.png
   :width: 800px
"""

import sys, math, numpy
from time import localtime, strftime
import matplotlib.pyplot as plt
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, mol, kmol, J, um
    
c_t = daeVariableType("c_t", dimless, -1.0e+20, 1.0e+20, 0.0, 1e-07)

u  = 1.0
D  = 0.002
L  = 1.0
dt = 0.3
pi = numpy.pi

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x",  self, m, "")

        self.c = daeVariable("c", c_t, self, "c using high resolution upwind scheme", [self.x])
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        xp = self.x.Points
        Nx = self.x.NumberOfPoints
        t  = Time()
        hr = daeHRUpwindSchemeEquation(self.c,  self.x, daeHRUpwindSchemeEquation.Phi_Koren, 1e-10)
        
        c = lambda i: self.c(i)
        
        # Convection-diffusion equation
        for i in range(1, Nx):
            eq = self.CreateEquation("c(%d)" % i, "")
            eq.Residual = hr.dc_dt(i) + u * hr.dc_dx(i) - D * hr.d2c_dx2(i)
            eq.CheckUnitsConsistency = False
        
        # BCs
        eq = self.CreateEquation("c(%d)" % 0, "")
        eq.Residual = c(0) - 0.0

class simTutorial(daeSimulation):
    def __init__(self, Nx):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_cv_6(%d)" % Nx)
        self.m.Description = __doc__
        
        self.Nx = Nx

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(self.Nx, 0.0, L)

    def SetUpVariables(self):
        Nx = self.m.x.NumberOfPoints
        xp = self.m.x.Points
        for i in range(1, Nx):
            self.m.c.SetInitialCondition(i, numpy.sin(pi*xp[i]))

# Setup everything manually and run in a console
def simulate(Nx):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial(Nx)

    # Do no print progress
    log.PrintProgress = False

    daesolver.RelativeTolerance = 1e-7
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.02
    simulation.TimeHorizon = dt

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())

    # 1. TCP/IP
    tcpipDataReporter = daeTCPIPDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    if not tcpipDataReporter.Connect("", simName):
        sys.exit()

    # 2. Data
    dr = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr)

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()
    
    ###########################################
    #  Data                                   #
    ###########################################
    results = dr.Process.dictVariables
    
    cvar  = results[simulation.m.Name + '.c']
    c       = cvar.Values[-1, :]        # 2D array [t,x]
    
    return simulation.m.x.Points, c

def run(guiRun = False, qtApp = None):
    Nxs = numpy.array([20, 40, 80])
    n = len(Nxs)
    hs = L / Nxs
    c  = []
    
    # Run simulations
    for i,Nx in enumerate(Nxs):
        nx, c_ = simulate(int(Nx))
        # Exact solution:
        cexact_ = []
        for xk in nx:
            if xk >= u*dt:
                ce = numpy.exp(-D*dt) * numpy.sin(pi*(xk-u*dt))
            else:
                ce = 0
            cexact_.append(ce)
        c.append((nx, c_, cexact_))
    
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(12,4), facecolor='white')
    
    ax = plt.subplot(131)
    plt.figure(1, facecolor='white')
    plt.plot(c[0][0], c[0][1], 'ro', linewidth=1.0, label='c (Nx=20)')
    plt.plot(c[0][0], c[0][2], 'b-', linewidth=1.0, label='c_exact (Nx=20)')
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('c', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.0, 1.0))
    #plt.ylim((0.0, 1.0))
        
    ax = plt.subplot(132)
    plt.figure(1, facecolor='white')
    plt.plot(c[1][0], c[1][1], 'ro', linewidth=1.0, label='c (Nx=40)')
    plt.plot(c[1][0], c[1][2], 'b-', linewidth=1.0, label='c_exact (Nx=40)')
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('c', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.0, 1.0))
    #plt.ylim((0.0, 1.0))
    
    ax = plt.subplot(133)
    plt.figure(1, facecolor='white')
    plt.plot(c[2][0], c[2][1], 'ro', linewidth=1.0, label='c (Nx=80)')
    plt.plot(c[2][0], c[2][2], 'b-', linewidth=1.0, label='c_exact (Nx=80)')
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('c', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.0, 1.0))
    #plt.ylim((0.0, 1.0))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
