#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_cv_3.py
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

References: 

1. G. Tryggvason. Method of Manufactured Solutions, Lecture 33: Predictivity-I, 2011.
   `PDF <http://www3.nd.edu/~gtryggva/CFD-Course/2011-Lecture-33.pdf>`_
2. K. Salari and P. Knupp. Code Verification by the Method of Manufactured Solutions. 
   SAND2000 – 1444 (2000).
   `doi:10.2172/759450 <https://doi.org/10.2172/759450>`_
3. P.J. Roache. Fundamentals of Verification and Validation. Hermosa, 2009.
   `ISBN-10:0913478121 <http://www.isbnsearch.org/isbn/0913478121>`_

The problem in this tutorial is identical to tutorial_cv_3. The only difference is that 
the Neumann boundary conditions are applied:
    
.. code-block:: none

   df(x=0)/dx   = dq(x=0)/dx   = cos(0 + Ct)
   df(x=2pi)/dx = dq(x=2pi)/dx = cos(2pi + Ct)

Numerical vs. manufactured solution plot (no. elements = 60, t = 1.0s):

.. image:: _static/tutorial_cv_3-results.png
   :width: 500px

The normalised global errors and the order of accuracy plots (no. elements = [60, 90, 120, 150], t = 1.0s):

.. image:: _static/tutorial_cv_3-results2.png
   :width: 800px
"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *
import matplotlib.pyplot as plt

no_t = daeVariableType("no_t", dimless, -1.0e+20, 1.0e+20, 0.0, 1e-6)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, m, "X axis domain")

        self.A = 1.0
        self.C = 1.0
        self.D = 0.05

        self.f = daeVariable("f", no_t, self, "", [self.x])
        self.q = daeVariable("q", no_t, self, "", [self.x])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Create some auxiliary functions to make equations more readable 
        A       = self.A
        C       = self.C
        D       = self.D
        t       = Time()
        f       = lambda x:    self.f(x)
        df_dt   = lambda x: dt(self.f(x))
        df_dx   = lambda x:  d(self.f(x), self.x, eCFDM)
        d2f_dx2 = lambda x: d2(self.f(x), self.x, eCFDM)
        q       = lambda x: A + numpy.sin(x() + C*t)
        dq_dt   = lambda x: C * numpy.cos(x() + C*t)
        dq_dx   = lambda x: numpy.cos(x() + C*t)
        d2q_dx2 = lambda x: -numpy.sin(x() + C*t)
        g       = lambda x: dq_dt(x) + q(x) * dq_dx(x) - D * d2q_dx2(x)

        # Numerical solution
        eq = self.CreateEquation("f", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        eq.Residual = df_dt(x) + f(x) * df_dx(x) - D * d2f_dx2(x) - g(x)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("f(0)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        eq.Residual = df_dx(x) - dq_dx(x)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("f(2pi)", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        eq.Residual = df_dx(x) - dq_dx(x)
        eq.CheckUnitsConsistency = False

        # Manufactured solution
        eq = self.CreateEquation("q", "Manufactured solution")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        eq.Residual = self.q(x) - q(x)
        eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self, Nx):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_cv_3(%d)" % Nx)
        self.m.Description = __doc__
        
        self.Nx = Nx

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(self.Nx, 0, 2*numpy.pi)
        
    def SetUpVariables(self):
        Nx = self.m.x.NumberOfPoints
        xp = self.m.x.Points
        for x in range(1, Nx-1):
            self.m.f.SetInitialCondition(x, self.m.A + numpy.sin(xp[x]))
                
# Setup everything manually and run in a console
def simulate(Nx):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial(Nx)

    # Do no print progress
    log.PrintProgress = False

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.05
    simulation.TimeHorizon = 1

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
    fvar = results[simulation.m.Name + '.f']
    qvar = results[simulation.m.Name + '.q']
    times = fvar.TimeValues
    q = qvar.Values[-1, :] # 2D array [t,x]
    f = fvar.Values[-1, :] # 2D array [t,x]
    #print(times,f,q)
    
    return times,f,q

def run(guiRun = False, qtApp = None):
    Nxs = numpy.array([60, 90, 120, 150])
    n = len(Nxs)
    L = 2*numpy.pi
    hs = L / Nxs
    E = numpy.zeros(n)
    C = numpy.zeros(n)
    p = numpy.zeros(n)
    E2 = numpy.zeros(n)
    
    # The normalised global errors
    for i,Nx in enumerate(Nxs):
        times, numerical_sol, manufactured_sol = simulate(int(Nx))
        E[i] = numpy.sqrt((1.0/Nx) * numpy.sum((numerical_sol-manufactured_sol)**2))

    # Order of accuracy
    for i,Nx in enumerate(Nxs):
        p[i] = numpy.log(E[i]/E[i-1]) / numpy.log(hs[i]/hs[i-1])
        C[i] = E[i] / hs[i]**p[i]
        
    C2 = 0.18 # constant for the second order slope line (to get close to the actual line)
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
    plt.xlim((0.04, 0.11))
        
    ax = plt.subplot(122)
    plt.figure(1, facecolor='white')
    plt.semilogx(hs[1:], p[1:],  'rs-', label='Order of Accuracy (p)')
    plt.xlabel('h', fontsize=fontsize)
    plt.ylabel('p', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.xlim((0.04, 0.075))
    plt.ylim((2.0, 2.04))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
