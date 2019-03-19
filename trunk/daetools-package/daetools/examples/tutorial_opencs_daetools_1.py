#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_opencs_daetools_1.py
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
DAE Tools version of the tutorial_opencs_dae_1.py example (IDAS idasAkzoNob_dns model).
"""

import os, sys, json
from time import localtime, strftime
from daetools.pyDAE import *
from tutorial_opencs_dae_1 import ChemicalKinetics

y_type = daeVariableType("y_type", unit(), 0, 1E10,   0, 1e-10)

class modTutorial(daeModel):
    def __init__(self, Name, cs_model, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.cs_model = cs_model
        
        self.N = daeDomain("N", self, unit(), "")
        self.y = daeVariable("y", y_type, self, "", [self.N])
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        
        Neq = self.cs_model.Nequations
        
        # Create a vector of variables:
        y = numpy.empty(Neq, dtype=object)
        y[:] = [self.y(i) for i in range(Neq)]

        # Create a vector of time derivatives:
        dydt = numpy.empty(Neq, dtype=object)
        dydt[:] = [dt(self.y(i)) for i in range(Neq)]
        
        equations = self.cs_model.CreateEquations(y, dydt)    
        for i in range(Neq):
            eq = self.CreateEquation("y(%d)" % i, "")
            eq.Residual = equations[i]
            eq.CheckUnitsConsistency = False

class simTutorial(daeSimulation):
    def __init__(self, cs_model):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_opencs_dae_1", cs_model)
        self.m.Description = __doc__
        
        self.cs_model = cs_model
    
    def SetUpParametersAndDomains(self):
        self.m.N.CreateArray(self.cs_model.Nequations)

    def SetUpVariables(self):
        ics = numpy.array(self.cs_model.GetInitialConditions())
        for i in range(self.cs_model.Nequations-1): # y6 is not differential variable
            self.m.y.SetInitialCondition(i, ics[i])
        
def run(**kwargs):
    cs_model = ChemicalKinetics()
    simulation = simTutorial(cs_model)
    return daeActivity.simulate(simulation, reportingInterval = 1, 
                                            timeHorizon       = 180,
                                            **kwargs)
           
if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
