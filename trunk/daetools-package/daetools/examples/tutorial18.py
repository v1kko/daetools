#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial18.py
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
This tutorial shows one more problem solved using the NumPy arrays that operate on
DAE Tools variables. The model is taken from the Sundials ARKODE (ark_analytic_sys.cpp).
The ODE system is defined by the following system of equations:

.. code-block:: none

   dy/dt = A*y

where:

.. code-block:: none

    A = V * D * Vi
    V = [1 -1 1; -1 2 1; 0 -1 2];
    Vi = 0.25 * [5 1 -3; 2 2 -2; 1 1 1];
    D = [-0.5 0 0; 0 -0.1 0; 0 0 lam];

lam is a large negative number.

The analytical solution to this problem is:

.. code-block:: none

    Y(t) = V*exp(D*t)*Vi*Y0

for t in the interval [0.0, 0.05], with initial condition y(0) = [1,1,1]'.

The stiffness of the problem is directly proportional to the value of "lamda".
The value of lamda should be negative to result in a well-posed ODE;
for values with magnitude larger than 100 the problem becomes quite stiff.

In this example, we choose lamda = -100.

The solution:

.. code-block:: none

   lamda = -100
   reltol = 1e-06
   abstol = 1e-10

   --------------------------------------
      t        y0        y1        y2
   --------------------------------------
    0.0050   0.70327   0.70627   0.41004
    0.0100   0.52267   0.52865   0.05231
    0.0150   0.41249   0.42145  -0.16456
    0.0200   0.34504   0.35696  -0.29600
    0.0250   0.30349   0.31838  -0.37563
    0.0300   0.27767   0.29551  -0.42383
    0.0350   0.26138   0.28216  -0.45296
    0.0400   0.25088   0.27459  -0.47053
    0.0450   0.24389   0.27053  -0.48109
    0.0500   0.23903   0.26858  -0.48740
   --------------------------------------

The plot of the 'y0', 'y1', 'y2' variables:

.. image:: _static/tutorial18-results.png
   :width: 500px
"""

import sys, numpy, scipy.linalg
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

typeNone = daeVariableType("typeNone", unit(), 0, 1E10,   0, 1e-10)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, unit(), "")

        self.y = daeVariable("y", typeNone, self, "", [self.x])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Input data:
        lamda = -100
        V = numpy.array([[1, -1, 1], [-1, 2, 1], [0, -1, 2]])
        Vi = 0.25 * numpy.array([[5, 1, -3], [2, 2, -2], [1, 1, 1]])
        D = numpy.array([[-0.5, 0, 0], [0, -0.1, 0], [0, 0, lamda]])
        self.y0 = numpy.array([1.0, 1.0, 1.0])

        # Create a vector of y:
        y = numpy.empty(3, dtype=object)
        y[:] = [self.y(i) for i in range(3)]

        # Create a vector of dy/dt:
        dydt = numpy.empty(3, dtype=object)
        dydt[:] = [dt(self.y(i)) for i in range(3)]

        # Create the ODE system: dy/dt = A*y
        # Use dot product (numpy arrays don't behave as matrices)
        # or use numpy.matrix where the operator * performs the dot product.
        Ay = V.dot(D).dot(Vi).dot(y)
        #print(Ay)
        for i in range(3):
            eq = self.CreateEquation("y(%d)" % i, "")
            eq.Residual = dydt[i] - Ay[i]
            eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial18")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.x.CreateArray(3)

    def SetUpVariables(self):
        self.m.y.SetInitialConditions(self.m.y0)

def run(**kwargs):
    simulation = simTutorial()
    relativeTolerance = 1E-6
    return daeActivity.simulate(simulation, reportingInterval = 0.005, 
                                            timeHorizon       = 0.050,
                                            relativeTolerance = relativeTolerance,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
