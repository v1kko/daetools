#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial15.py
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

- Nested state transitions

In this example we use the same model as in the tutorial 4 with the more complex STN:

.. code-block:: none

   IF t < 200
     IF 0 <= t < 100
        IF 0 <= t < 50
          Qin = 1600 W
        ELSE
          Qin = 1500 W
     ELSE
       Qin = 1400 W

   ELSE IF 200 <= t < 300
     Qin = 1300 W

   ELSE
     Qin = 0 W

The plot of the 'Qin' variable:

.. image:: _static/tutorial15-results.png
   :width: 500px

The temperature plot:

.. image:: _static/tutorial15-results2.png
   :width: 500px
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kW

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       kg,           self, "Mass of the copper plate")
        self.cp    = daeParameter("c_p",     J/(kg*K),     self, "Specific heat capacity of the plate")
        self.alpha = daeParameter("&alpha;", W/((m**2)*K), self, "Heat transfer coefficient")
        self.A     = daeParameter("A",       m**2,         self, "Area of the plate")
        self.Tsurr = daeParameter("T_surr",  K,            self, "Temperature of the surroundings")

        self.Qin  = daeVariable("Q_in",  power_t,       self, "Power of the heater")
        self.T    = daeVariable("T",     temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        # IF-1
        self.IF(Time() < Constant(200*s)) # --------------------------------------------------> 1

        # IF-1.1
        self.IF((Time() >= Constant(0*s)) & (Time() < Constant(100*s))) # ------------> 1.1

        # IF-1.1.1
        self.IF((Time() >= Constant(0*s)) & (Time() < Constant(50*s))) # ---> 1.1.1
        eq = self.CreateEquation("Q_111", "The heater is on")
        eq.Residual = self.Qin() - Constant(1600 * W)

        self.ELSE() # ----------------------------------------------------> 1.1.1
        eq = self.CreateEquation("Q_112", "The heater is on")
        eq.Residual = self.Qin() - Constant(1500 * W)

        self.END_IF() # --------------------------------------------------> 1.1.1

        self.ELSE() # --------------------------------------------------------------> 1.1
        eq = self.CreateEquation("Q_12", "The heater is on")
        eq.Residual = self.Qin() - Constant(1400 * W)

        self.END_IF() # ------------------------------------------------------------>1.1

        self.ELSE_IF((Time() >= Constant(200*s)) & (Time() < Constant(300*s))) # ------------->1

        eq = self.CreateEquation("Q_2", "The heater is on")
        eq.Residual = self.Qin() - Constant(1300 * W)

        self.ELSE() # ------------------------------------------------------------------------>1

        eq = self.CreateEquation("Q_3", "The heater is off")
        eq.Residual = self.Qin()

        self.END_IF() # ---------------------------------------------------------------------->1

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial15")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(283 * K)

def run(**kwargs):
    simulation = simTutorial()
    daeActivity.simulate(simulation, reportingInterval = 10, 
                                     timeHorizon       = 500,
                                     **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
