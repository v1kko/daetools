#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_opt_6.py
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
COPS optimisation test 14: Catalyst Mixing.

Determine the optimal mixing policy of two catalysts along the length of a tubular
plug flow reactor involving several reactions.

Reference: Benchmarking Optimization Software with COPS 3.0, Mathematics and Computer
Science Division, Argonne National Laboratory, Technical Report ANL/MCS-273, 2004.
`PDF <http://www.mcs.anl.gov/~more/cops/cops3.pdf>`_

In DAE Tools numerical solution of dynamic optimisation problems is obtained using
the Direct Sequential Approach where, given a set of values for the decision variables,
the system of ODEs are accurately integrated over the entire time interval using specific
numerical integration formulae so that the objective functional can be evaluated.
Therefore, the differential equations are satisfied at each iteration of the
optimisation procedure.

In the COPS test, the problem is solved using the Direct Simultaneous Approach where
the equations that result from a discretisation of an ODE model using orthogonal
collocation on finite elements (OCFE), are incorporated directly into the optimisation
problem, and the combined problem is then solved using a large-scale optimisation strategy.

The results: fobj = -4.79479E-2 (for Nh = 100) and -4.78676E-02 (for Nh = 200).

The control variables plot (for Nh = 100):

.. image:: _static/tutorial_che_opt_6-results.png
   :width: 500px

The control variables plot (for Nh = 200):

.. image:: _static/tutorial_che_opt_6-results2.png
   :width: 500px
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
from daetools.solvers.ipopt import pyIPOPT
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l

x_t  = daeVariableType("x_t", unit(), -1.0e+20, 1.0e+20, 0.0, 1e-07)

class modCatalystMixing(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Ni = daeDomain("Ni", self, unit(), "Number of time intervals")

        # Control variables at specific time intervals
        self.uc = daeVariable("uc", no_t, self, "Control variable at the specified time interval", [self.Ni])

        # Control variable in the current time interval (used in equations)
        self.u = daeVariable("u",  no_t, self, "The mixing ratio of the catalysts")

        # State variables
        self.x1 = daeVariable("x1", x_t, self, "Catalyst 1")
        self.x2 = daeVariable("x2", x_t, self, "Catalyst 2")

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        x1 = self.x1()
        x2 = self.x2()
        u  = self.u()
        uc = lambda i: self.uc(i)

        # Derivatives
        dx1_dt = self.x1.dt()
        dx2_dt = self.x2.dt()

        # Switch to different control variables at different time intervals
        Ni = self.Ni.NumberOfPoints
        self.uc_STN = self.STN('uc')
        for i in range(Ni):
            self.STATE('u_%d' % i)
            eq = self.CreateEquation("u_%d" % i, "")
            eq.Residual = u - uc(i)
        self.END_STN()

        # x1
        eq = self.CreateEquation("x1", "")
        eq.Residual = dx1_dt - u*(10*x2 - x1)
        eq.CheckUnitsConsistency = False

        # x2
        eq = self.CreateEquation("x2", "")
        eq.Residual = dx2_dt - ( u*(x1 - 10*x2) - (1-u)*x2 )
        eq.CheckUnitsConsistency = False

class simCatalystMixing(daeSimulation):
    def __init__(self, Ni, dt):
        daeSimulation.__init__(self)
        self.m = modCatalystMixing("tutorial_che_opt_6")
        self.m.Description = __doc__
        self.Ni = Ni
        self.dt = dt

    def SetUpParametersAndDomains(self):
        self.m.Ni.CreateArray(self.Ni)

    def SetUpVariables(self):
        for i in range(self.m.Ni.NumberOfPoints):
            self.m.uc.AssignValue(i, 0.0)

        self.m.x1.SetInitialCondition(1.0)
        self.m.x2.SetInitialCondition(0.0)

    def Run(self):
        t = 0.0
        for i in range(self.Ni):
            tn = self.CurrentTime+self.dt
            if tn > self.TimeHorizon:
                tn = self.TimeHorizon
            self.m.uc_STN.ActiveState = 'u_%d' % i
            self.Reinitialize()
            self.Log.Message('Interval %d (u=%f): integrating from %f to %f ...' % (i, self.m.uc.GetValue(i), self.CurrentTime, tn), 0)
            self.IntegrateUntilTime(tn, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))

    def SetUpOptimization(self):
        # Yield of component B (mol)
        self.ObjectiveFunction.Residual = -1 + self.m.x1() + self.m.x2()

        # Set the inequality constraints.
        # Nota bene:
        #  Not required here since the bounds can be enforced in the continuous optimization variables.
        #
        #for i in range(self.Ni):
        #    c1 = self.CreateInequalityConstraint("umax") # u - 1 <= 0
        #    c1.Residual = self.m.uc(i) - Constant(1.0)
        #    c2 = self.CreateInequalityConstraint("umin") # 0 - u <= 0
        #    c2.Residual = -self.m.uc(i)

        self.u_opt = []
        for i in range(self.Ni):
            self.u_opt.append( self.SetContinuousOptimizationVariable(self.m.uc(i), 0.0, 1.0, 0.0) )

def setOptions(nlpsolver):
    nlpsolver.SetOption('print_level', 0)
    nlpsolver.SetOption('tol', 5e-5)
    nlpsolver.SetOption('mu_strategy', 'adaptive')
    #nlpsolver.SetOption('obj_scaling_factor', 100.0)
    #nlpsolver.SetOption('nlp_scaling_method', 'none') #'user-scaling')

def run(guiRun = False, qtApp = None):
    simulation = simCatalystMixing(100, 1.0/100)
    nlpsolver  = pyIPOPT.daeIPOPT()
    lasolver   = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    relativeTolerance = 1e-6
    daeActivity.optimize(simulation, reportingInterval       = 1, 
                                     timeHorizon             = 1,
                                     lasolver                = lasolver,
                                     nlpsolver               = nlpsolver,
                                     nlpsolver_setoptions_fn = setOptions,
                                     relativeTolerance       = relativeTolerance,
                                     guiRun                  = guiRun,
                                     qtApp                   = qtApp)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun)
