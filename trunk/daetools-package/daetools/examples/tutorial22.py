#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial22.py
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
This tutorial illustrates the code verification method using the Method of Manufactured 
Solutions:

Reference: G. Tryggvason. Method of Manufactured Solutions, Lecture 33: Predictivity-I, 2011.
`PDF link <http://www3.nd.edu/~gtryggva/CFD-Course/2011-Lecture-33.pdf>`_

The procedure for the sample problem:
    
.. code-block:: none

   L(u) = d2u/dx2 = 0

is the following:
    
1. Pick a function (q, the analytical solution): 
    
   .. code-block:: none
    
      q = 1 - x**2

2. Compute the new source term (g) for the original problem:
    
   .. code-block:: none
      
      g = d2q/dx2

3. Solve the original problem with the new source term:
    
   .. code-block:: none

      d2u/dx2 = g

Since L(u) = g and g = L(q), consequently we have: u = q.
Therefore, the computed numerical solution (f) should be equal to the analytical one (q).

The terms in the source g term are:

.. code-block:: none

   dq/dx   = -2x
   d2q/dx2 = -2

The model in this example is very similar to the model used in the tutorial 2.

The solution plot (t = 1.0 s):

.. image:: _static/tutorial22-results.png
   :width: 500px
"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *

no_t = daeVariableType("no_t", dimless, -1.0e+20, 1.0e+20, 0.0, 1e-6)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, m, "X axis domain")

        self.u = daeVariable("u", no_t, self, "", [self.x])
        self.q = daeVariable("q", no_t, self, "", [self.x])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Create some auxiliary functions to make equations more readable 
        u       = lambda x: self.u(x)
        d2u_dx2 = lambda x: d2(self.u(x), self.x, eCFDM)
        q       = lambda x: 1 - x()**2
        d2q_dx2 = lambda x: -2
        g       = lambda x: d2q_dx2(x)

        # Numerical solution
        eq = self.CreateEquation("f", "Numerical solution")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        eq.Residual = d2u_dx2(x) - g(x)
        eq.CheckUnitsConsistency = False

        # BC at x = 0
        eq = self.CreateEquation("u(0)", "BC at x = 0")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        eq.Residual = u(x) - 1

        # BC at x = 1
        eq = self.CreateEquation("u(1)", "BC at x = 01")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        eq.Residual = u(x) - 0

        # Analytical solution
        eq = self.CreateEquation("q", "Analytical solution")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        eq.Residual = self.q(x) - q(x)
        eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial22")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(50, 0, 1)
        
    def SetUpVariables(self):
        pass
                
# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportTimeDerivatives = True
    sim.ReportingInterval = 1
    sim.TimeHorizon       = 1
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

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 1

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
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
