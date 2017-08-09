#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial7.py
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

- Quasi steady state initial condition mode (eQuasiSteadyState flag)
- User-defined schedules (operating procedures)
- Resetting of degrees of freedom
- Resetting of initial conditions

In this example we use the same heat transfer problem as in the tutorial 4.
The input power of the heater is defined as a variable. Since there is
no equation defined to calculate the value of the input power, the system
contains N variables but only N-1 equations. To create a well-posed DAE system
one of the variable needs to be "fixed". However the choice of variables is not
arbitrary and in this example the only variable that can be fixed is Qin. Thus,
the Qin variable represents a degree of freedom (DOF). Its value will be fixed
at the beginning of the simulation and later manipulated in the user-defined
schedule in the overloaded function daeSimulation.Run().

The default daeSimulation.Run() function (re-implemented in Python) is:
    
.. code-block:: python

    def Run(self):
        # Python implementation of daeSimulation::Run() C++ function.
        
        import math
        while self.CurrentTime < self.TimeHorizon:
            # Get the time step (based on the TimeHorizon and the ReportingInterval).
            # Do not allow to get past the TimeHorizon.
            t = self.NextReportingTime
            if t > self.TimeHorizon:
                t = self.TimeHorizon

            # If the flag is set - a user tries to pause the simulation, therefore return.
            if self.ActivityAction == ePauseActivity:
                self.Log.Message("Activity paused by the user", 0)
                return

            # If a discontinuity is found, loop until the end of the integration period.
            # The data will be reported around discontinuities!
            while t > self.CurrentTime:
                self.Log.Message("Integrating from [%f] to [%f] ..." % (self.CurrentTime, t), 0)
                self.IntegrateUntilTime(t, eStopAtModelDiscontinuity, True)
            
            # After the integration period, report the data. 
            self.ReportData(self.CurrentTime)
            
            # Set the simulation progress.
            newProgress = math.ceil(100.0 * self.CurrentTime / self.TimeHorizon)
            if newProgress > self.Log.Progress:
                self.Log.Progress = newProgress
                
In this example the following schedule is specified:

1. Run the simulation for 100s using the daeSimulation.IntegrateForTimeInterval() function
   and report the data using the daeSimulation.ReportData() function.

2. Re-assign the value of Qin to 2000W. After re-assigning DOFs or re-setting initial conditions
   the function daeSimulation.Reinitialize() has to be called to reinitialise the DAE system.
   Use the function daeSimulation.IntegrateUntilTime() to run until the time reaches 200s
   and report the data.

3. Re-assign the variable Qin to a new value 1500W, re-initialise the temperature again to 300K
   re-initialise the system, run the simulation until the TimeHorizon is reached using the function
   daeSimulation.Integrate() and report the data.

The plot of the inlet power:

.. image:: _static/tutorial7-results.png
   :width: 500px

The temperature plot:

.. image:: _static/tutorial7-results2.png
   :width: 500px
"""

import sys, math
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, J, W

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
        eq.BuildJacobianExpressions = True
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial7")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.Qin.AssignValue(0 * W)
        
        # Here we can manually set the initial temperature to the temperature of the surroundings (283 * K).
        # However, here we use the eQuasiSteadyState initial condition mode available in the Sundials IDA solver
        # It assumes all time derivatives are initially equal to zero and calculates the non-derivative parts.
        # As a result, the initial temperature will be equal to the temperature of the surroundings (283 K).
        self.InitialConditionMode = eQuasiSteadyState

    # daeSimulation class provides the function Run() which is called after successful initialisation
    # to run the simulation. By default, it runs for time period defined by the TimeHorizon property,
    # stopping after each period of time defined by the ReportInterval property to report the data.
    # However, the default behaviour can be changed by re-implementing the function Run().
    # The functions Integrate(), IntegrateUntilTime(), and IntegrateForTimeInterval() from the
    # daeSimulation class can be used to advance in time, while functions ReAssignValue() and
    # ReSetInitialCondition() from daeVariable class can be used to alter the values of variables.
    # In this example we specify the following schedule:
    #  1. Run the simulation for 100s using the function daeSimulation.IntegrateForTimeInterval()
    #     and report the data using the function daeSimulation.ReportData().
    #  2. Re-assign the value of Qin to 2000W. After re-assigning DOFs or re-setting initial conditions
    #     the function daeSimulation.Reinitialize() has to be called to reinitialise the DAE system.
    #     Use the function daeSimulation.IntegrateUntilTime() to run until the time reaches 200s
    #     and report the data.
    #  3. Re-assign the variable Qin to a new value 1500W, re-initialise the temperature again to 300K
    #     re-initialise the system, run the simulation until the TimeHorizon is reached using the function
    #     daeSimulation.Integrate() and report the data.
    # Nota bene:
    #  a) The daeLog object (accessed through the simulation.Log property) can be used to print the messages
    #     and to set the simulation progress (in percents) using the function log.SetProgress().
    #  b) Integration functions require a flag as a second argument that specifies how to perform the
    #     integration. It can be one of:
    #      - eDoNotStopAtDiscontinuity (integrate and do not return even if one of the conditions have been satisfied)
    #      - eStopAtDiscontinuity (integrate and return if some conditions have been satisfied); in this case,
    #        the integration has to be performed in a loop until the required time is reached.
    def Run(self):
        # 1. Set Qin=500W and integrate for 100s
        self.m.Qin.ReAssignValue(500 * W)
        self.Reinitialize()
        self.ReportData(self.CurrentTime)
        self.Log.Message("OP: Integrating for 100 seconds ... ", 0)
        time = self.IntegrateForTimeInterval(100, eDoNotStopAtDiscontinuity)
        self.ReportData(self.CurrentTime)
        self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon));   

        # 2. Set Qin=750W and integrate until time = 200s
        self.m.Qin.ReAssignValue(750 * W)
        self.Reinitialize()
        self.ReportData(self.CurrentTime)
        self.Log.Message("OP: Integrating until time = 200 seconds ... ", 0)
        time = self.IntegrateUntilTime(200, eDoNotStopAtDiscontinuity)
        self.ReportData(self.CurrentTime)
        self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon));   

        # 3. Set Qin=1000W and integrate until the specified TimeHorizon is reached
        self.m.Qin.ReAssignValue(1000 * W)
        self.m.T.ReSetInitialCondition(300 * K)
        self.Reinitialize()
        self.ReportData(self.CurrentTime)
        self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))  
        self.Log.Message("OP: Integrating from " + str(time) + " to the time horizon (" + str(self.TimeHorizon) + ") ... ", 0)
        time = self.Integrate(eDoNotStopAtDiscontinuity)
        self.ReportData(self.CurrentTime)
        self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon));   
        self.Log.Message("OP: Finished", 0)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 500
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

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 500

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
