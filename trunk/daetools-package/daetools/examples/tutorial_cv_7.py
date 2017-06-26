
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_cv_7.py
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

Reference (section 4.2):

- B. Koren. A robust upwind discretization method for advection, diffusion and source terms. 
  Department of Numerical Mathematics. Report NM-R9308 (1993). 
  `PDF <http://oai.cwi.nl/oai/asset/5293/05293D.pdf>`_
  
The problem in this tutorial is 1D *steady-state convection-diffusion* (Burger's) equation:
    
.. code-block:: none

   u*dc/dx - D*d2c/dc2 = 0

The manufactured solution is:
    
.. code-block:: none

   c(x) = 0.5 * (1 - cos(2*pi*(x-a)/(b-a))), x in [a,b]
   c(x) = 0, otherwise

The new source term is:
    
.. code-block:: none

   s(x) = pi/(b-a) * u * sin(2*pi*(x-a)/(b-a)) - 2*(pi/(b-a))**2 * D * cos(2*pi*(x-a)/(b-a)), x in [a,b]
   s(x) = 0, otherwise
    
The modified equation:
    
.. code-block:: none

   u*dc/dx - D*d2c/dc2 = s(x)

is solved using the high resolution cell-centered finite volume upwind scheme
with flux limiter described in the article.

In order to obtain the consistent discretisation of the convection and the source terms 
an integral of the source term: S(x) = 1/u * Integral s(x)*dx must be calculated.
The result of integration is given as:
    
.. code-block:: none

   S(x) = 0.5 * (1 - cos(2*pi*(x-a)/(b-a)))) - pi/(b-a) * D/u * sin(2*pi*(x-a)/(b-a))), x in [a,b]
   S(x) = 0, otherwise
    
Numerical vs. manufactured solution plot (Nx=80):

.. image:: _static/tutorial_cv_7-results.png
   :width: 500px

The normalised global errors and the order of accuracy plots 
for the Koren flux limiter (grids 40, 60, 80, 120):

.. image:: _static/tutorial_cv_7-results-koren.png
   :width: 800px

The normalised global errors and the order of accuracy plots 
for the Superbee flux limiter (grids 40, 60, 80, 120):

.. image:: _static/tutorial_cv_7-results-superbee.png
   :width: 800px
"""

import sys, math, numpy
from time import localtime, strftime
import matplotlib.pyplot as plt
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, mol, kmol, J, um
    
c_t = daeVariableType("c_t", dimless, -1.0e+20, 1.0e+20, 0.0, 1e-07)

a = 0.2
b = 0.6
D = 0.01
u = 1.0
L = 1.0
pi = numpy.pi

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x",  self, m, "")

        self.c_exact = daeVariable("c_exact", c_t, self, "c using the analytical solution",       [self.x])
        self.c       = daeVariable("c",       c_t, self, "c using high resolution upwind scheme", [self.x])
        
        self.hr = daeHRUpwindScheme(self.c,  self.x, daeHRUpwindScheme.Phi_Koren, 1e-10)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        hr = self.hr
        xp = self.x.Points
        Nx = self.x.NumberOfPoints
        term1 = pi/(b-a)
        term2 = lambda x: 2*pi * (x-a) / (b-a)   
        
        c = lambda i: self.c(i)
        
        # Manufactured (exact) solution
        def c_exact(i):
            x = xp[i]
            if x >= a and x <= b:
                return 0.5 * (1 - numpy.cos(term2(x)))
            else:
                return 0.0
        for i in range(0, Nx):
            eq = self.CreateEquation("c_exact(%d)" % i, "")
            eq.Residual =  self.c_exact(i) - c_exact(i)
            eq.CheckUnitsConsistency = False
            
        # The source term integral    
        def S(i):
            # Analytical integral S = 1/u * Integral(s(x)*dx)
            x = xp[i]
            C1 = 0.5
            if x >= a and x <= b:
                # The continuous source integral:
                res = -0.5 * numpy.cos(term2(xp[i])) - term1 * D/u * numpy.sin(term2(xp[i])) + C1
            else:
                res = 0.0
            return res
        
        # Convection-diffusion equation
        for i in range(1, Nx):
            eq = self.CreateEquation("c(%d)" % i, "")
            eq.Residual = u * hr.dc_dx(i, S) - D * hr.d2c_dx2(i) #- S_num(i)
            eq.CheckUnitsConsistency = False
        
        # BCs
        eq = self.CreateEquation("c(%d)" % 0, "")
        eq.Residual = c(0) - 0

class simTutorial(daeSimulation):
    def __init__(self, Nx):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_cv_7(%d)" % Nx)
        self.m.Description = __doc__
        
        self.Nx = Nx

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(self.Nx, 0.0, L)

    def SetUpVariables(self):
        pass

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
    simulation.ReportingInterval = 1.0
    simulation.TimeHorizon = 1.0

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
        
    C2 = 3.0 # constant for the second order slope line (to get close to the actual line)
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
