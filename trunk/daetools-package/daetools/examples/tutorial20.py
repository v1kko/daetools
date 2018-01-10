#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial20.py
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
This tutorial illustrates the support variable constraints available in Sundials IDA solver.
Benchmarks are available from `Matlab documentation 
<https://www.mathworks.com/help/matlab/math/nonnegative-ode-solution.html>`_.

In DAE Tools contraints follow the Sundials IDA solver implementation and can be 
specified using the valueConstraint argument of the daeVariableType class __init__
function:

 - eNoConstraint (default)
 - eValueGTEQ: imposes >= 0 constraint
 - eValueLTEQ: imposes <= 0 constraint
 - eValueGT:   imposes > 0 constraint
 - eValueLT:   imposes < 0 constraint

and changed for individual variables using daeVariable.SetValueConstraint functions. 
"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *

# Impose >= constraint on y value using the eValueGTEQ flag. 
type_y = daeVariableType("type_y", unit(), 0, 1E10, 0, 1e-5, eValueGTEQ)

class modTutorial1(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.y = daeVariable("y", type_y, self, "")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Auxiliary objects to make equations more readable 
        y     = self.y()
        dy_dt = dt(self.y())

        eq = self.CreateEquation("y")
        eq.Residual = dy_dt + numpy.fabs(y)
        eq.CheckUnitsConsistency = False

class modTutorial2(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.y = daeVariable("y", type_y, self, "")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # 0 < eps << 1
        epsilon = 1e-6
        
        # Auxiliary objects to make equations more readable 
        t     = Time()
        y     = self.y()
        dy_dt = dt(self.y())
        
        eq = self.CreateEquation("y")
        eq.Residual = epsilon * dy_dt - ((1-t)*y - y**2)
        eq.CheckUnitsConsistency = False

class simTutorial1(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial1("tutorial20(1)")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.y.SetInitialCondition(1.0)

class simTutorial2(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial2("tutorial20(2)")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.y.SetInitialCondition(1.0)

def run1(**kwargs):
    simulation = simTutorial1()
    return daeActivity.simulate(simulation, reportingInterval = 1.0, 
                                            timeHorizon       = 40.0,
                                            **kwargs)

def run2(**kwargs):
    simulation = simTutorial2()
    return daeActivity.simulate(simulation, reportingInterval = 0.05, 
                                            timeHorizon       = 2.0,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run1(guiRun = guiRun)
    run2(guiRun = guiRun)
