
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_cv_8.py
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
Code verification using the Method of Manufactured Solutions.

Reference (page 64):

- W. Hundsdorfer. Numerical Solution of Advection-Diffusion-Reaction Equations. 
  Lecture notes (2000), Thomas Stieltjes Institute. 
  `PDF <http://homepages.cwi.nl/~willem/Coll_AdvDiffReac/notes.pdf>`_
  
The problem in this tutorial is 1D *transient convection-reaction* equation:
    
.. code-block:: none

   dc/dt + dc/dx = c**2

The exact solution is:
    
.. code-block:: none

   c(x,t) = sin(pi*(x-t))**2 / (1 - t*sin(pi*(x-t))**2)

The equation is solved using the high resolution cell-centered finite volume upwind scheme
with flux limiter described in the article. The boundary and initial conditions are obtained 
from the exact solution.

The consistent discretisation of the convection and the source terms cannot be done
since the constant C1 in the integral of the source term: 
    
.. code-block:: none

    S(x) = 1/u * Integral s(x)*dx = u**/3 + C1 

is not known. Therefore, the numerical cell average is used: 

.. code-block:: none

    Snum(x) = Integral s(x)*dx = s(i) * (x[i]-x[i-1]).
    
Numerical vs. manufactured solution plot (Nx=80):

.. image:: _static/tutorial_cv_8-results.png
   :width: 500px

The normalised global errors and the order of accuracy plots 
for the Koren flux limiter (grids 40, 60, 80, 120):

.. image:: _static/tutorial_cv_8-results-koren.png
   :width: 800px
"""

import sys, math, numpy
from time import localtime, strftime
import matplotlib.pyplot as plt
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, mol, kmol, J, um
    
c_t = daeVariableType("c_t", dimless, -1.0e+20, 1.0e+20, 0.0, 1e-07)

L    = 1.0
pi   = numpy.pi
tend = 0.5
t    = Time()

def c_exact(x,time):
    return numpy.sin(pi*(x-time))**2 / (1 - time*numpy.sin(pi*(x-time))**2)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x",  self, m, "")

        self.c_exact = daeVariable("c_exact", c_t, self, "c using the analytical solution",       [self.x])
        self.c       = daeVariable("c",       c_t, self, "c using high resolution upwind scheme", [self.x])
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        xp = self.x.Points
        Nx = self.x.NumberOfPoints
        hr = daeHRUpwindSchemeEquation(self.c,  self.x, daeHRUpwindSchemeEquation.Phi_Koren, 1e-10)
        
        c = lambda i: self.c(i)
        
        # Manufactured (exact) solution
        for i in range(0, Nx):
            eq = self.CreateEquation("c_exact(%d)" % i, "")
            eq.Residual =  self.c_exact(i) - c_exact(xp[i],t)
            eq.CheckUnitsConsistency = False
            
        # The source function
        def s(i):
            return c(i)**2
        
        # The analytical source term integral for consistent discretisation of convection and source terms: 
        #   S = 1/u * Integral(s(x)*dx)  
        def S(i):
            C1 = 0.0
            return c(i)**2 / 3 + C1
        
        # Convection-diffusion equation
        for i in range(1, Nx):
            eq = self.CreateEquation("c(%d)" % i, "")
            eq.Residual = hr.dc_dt(i) + hr.dc_dx(i) - hr.source(s,i)
            eq.CheckUnitsConsistency = False
        
        # BCs
        eq = self.CreateEquation("c(0)", "")
        eq.Residual = c(0) - c_exact(xp[0], t)
        eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self, Nx):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_cv_8(%d)" % Nx)
        self.m.Description = __doc__
        
        self.Nx = Nx

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(self.Nx, 0.0, L)

    def SetUpVariables(self):
        Nx = self.m.x.NumberOfPoints
        xp = self.m.x.Points
        for i in range(1, Nx):
            self.m.c.SetInitialCondition(i, c_exact(xp[i], 0.0))

# Setup everything manually and run in a console
def simulate(Nx):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial(Nx)

    daesolver.RelativeTolerance = 1e-7
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.05
    simulation.TimeHorizon = tend

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
    
    cvar        = results[simulation.m.Name + '.c']
    c_var_exact = results[simulation.m.Name + '.c_exact']
    
    c       = cvar.Values[-1, :]        # 2D array [t,x]
    c_exact = c_var_exact.Values[-1, :] # 2D array [t,x]
    
    return c, c_exact

def run():
    Nxs = numpy.array([40, 60, 80, 120])
    n = len(Nxs)
    hs = L / Nxs
    E = numpy.zeros(n)
    C = numpy.zeros(n)
    p = numpy.zeros(n)
    E2 = numpy.zeros(n)
    
    # The normalised global errors
    for i,Nx in enumerate(Nxs):
        numerical_sol, manufactured_sol = simulate(int(Nx))
        E[i] = numpy.sqrt((1.0/Nx) * numpy.sum((numerical_sol-manufactured_sol)**2))

    # Order of accuracy
    for i,Nx in enumerate(Nxs):
        p[i] = numpy.log(E[i]/E[i-1]) / numpy.log(hs[i]/hs[i-1])
        C[i] = E[i] / hs[i]**p[i]
        
    C2 = 17 # constant for the second order slope line (to get close to the actual line)
    E2 = C2 * hs**2 # E for the second order slope
    
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(10,4), facecolor='white')
    fig.canvas.set_window_title('The Normalised global errors and the Orders of accuracy (Nelems = %s)' % Nxs.tolist())
    
    ax = plt.subplot(121)
    plt.figure(1, facecolor='white')
    plt.loglog(hs, E,  'ro', label='E(h)')
    plt.loglog(hs, E2, 'b-', label='2nd order slope')
    plt.xlabel('h', fontsize=fontsize)
    plt.ylabel('||E||', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.04, 0.11))
        
    ax = plt.subplot(122)
    plt.figure(1, facecolor='white')
    plt.semilogx(hs[1:], p[1:],  'rs-', label='Order of Accuracy (p)')
    plt.xlabel('h', fontsize=fontsize)
    plt.ylabel('p', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    #plt.xlim((0.04, 0.075))
    #plt.ylim((2.0, 2.04))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
