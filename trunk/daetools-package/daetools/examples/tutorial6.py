#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial6.py
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
This tutorial introduces the following concepts:

- Ports
- Port connections
- Units (instances of other models)

A simple port type 'portSimple' is defined which contains only one variable 't'.
Two models 'modPortIn' and 'modPortOut' are defined, each having one port of type 'portSimple'.
The wrapper model 'modTutorial' instantiate these two models as its units and connects them
by connecting their ports.
"""

import sys, tempfile
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

# Ports, like models, consist of domains, parameters and variables. Parameters and variables
# can be distributed as well. Here we define a very simple port, with only one variable.
# The process of defining ports is analogous to defining models. Domains, parameters and
# variables are declared in the constructor __init__ and their constructor accepts ports as
# the 'Parent' argument.
class portSimple(daePort):
    def __init__(self, Name, PortType, Model, Description = ""):
        daePort.__init__(self, Name, PortType, Model, Description)

        self.t = daeVariable("t", time_t, self, "Time elapsed in the process")

# Here we define two models, 'modPortIn' and 'modPortOut' each having one port of type portSimple.
# The model 'modPortIn' contains inlet port Pin while the model 'modPortOut' contains outlet port Pout.
class modPortIn(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Pin = portSimple("P_in", eInletPort, self, "The simple port")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

class modPortOut(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Pout = portSimple("P_out", eOutletPort, self, "The simple port")
        self.time = daeVariable("time", time_t, self, "Time elapsed in the process")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("time", "Differential equation to calculate the time elapsed in the process.")
        eq.Residual = self.time.dt() - 1.0

        eq = self.CreateEquation("Port_t", "")
        eq.Residual = self.Pout.t() - self.time()

# Model 'modTutorial' declares two units mpin of type 'modPortIn' and 'mpout' of type 'modPortOut'.
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.mpin  = modPortIn("Port_In", self)
        self.mpout = modPortOut("Port_Out", self)

        # Ports can be connected by using the function ConnectPorts from daeModel class. Apparently,
        # ports dont have to be of the same type but must contain the same number of parameters and variables.
        self.ConnectPorts(self.mpout.Pout, self.mpin.Pin)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
   
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial6")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.mpout.time.SetInitialCondition(0)

def run(**kwargs):
    simulation = simTutorial()
    daeActivity.simulate(simulation, reportingInterval = 10, 
                                     timeHorizon       = 100,
                                     **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
