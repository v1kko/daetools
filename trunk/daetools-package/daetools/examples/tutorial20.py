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
This tutorial illustrates the sensitivity analysis features in DAE Tools.

This model has one state variable (T) and one degree of freedom (Qin).
Qin is set as a parameter for sensitivity analysis.

The sensitivity analysis is enabled and the sensitivities can be reported to the 
data reporter like any ordinary variable by setting the boolean property 
simulation.ReportSensitivities to True.

Raw sensitivity matrices can be saved into a specified directory using the 
simulation.SensitivityDataDirectory property (before a call to Initialize).
The sensitivity matrics are saved in .mmx coordinate format where the first
dimensions is Nparameters and second Nvariables: S[Np, Nvars].

The plot of the sensitivity of T per Qin:
    
.. image:: _static/tutorial20-results.png
   :width: 500px
"""

import os, sys, tempfile, numpy, scipy, scipy.io
from time import localtime, strftime
from daetools.pyDAE import *
import matplotlib.pyplot as plt

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       kg,           self, "Mass of the copper plate")
        self.cp    = daeParameter("c_p",     J/(kg*K),     self, "Specific heat capacity of the plate")
        self.alpha = daeParameter("&alpha;", W/((m**2)*K), self, "Heat transfer coefficient")
        self.A     = daeParameter("A",       m**2,         self, "Area of the plate")
        self.Tsurr = daeParameter("T_surr",  K,            self, "Temperature of the surroundings")

        self.Qin  = daeVariable("Qin",  power_t,       self, "Power of the heater")
        self.T    = daeVariable("T",    temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.BuildJacobianExpressions = True
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial20")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.Qin.AssignValue(1000 * W)
        self.m.T.SetInitialCondition(283 * K)
        
    def SetUpSensitivityAnalysis(self):
        # SetSensitivityParameter is a handy alias for SetContinuousOptimizationVariable(variable, LB=0.0, UB=1.0, defaultValue=1.0)
        # In this scenario, the lower bound, the upper bound and the default value are unused.
        # If required, the optimisation functions can be added using the simulation.SetNumberOfObjectiveFunctions(n) function.
        self.SetSensitivityParameter(self.m.Qin)

def run():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    dr_tcpip     = daeTCPIPDataReporter()
    dr_data      = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_data)
    simulation   = simTutorial()

    # Do no print progress
    log.PrintProgress = False

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True

    # Enable reporting of sensitivities for all reported variables
    simulation.ReportSensitivities = True

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 5
    simulation.TimeHorizon = 200

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if not dr_tcpip.Connect("", simName):
        sys.exit()

    # Initialize the simulation
    # The .mmx files with the sensitivity matrices will be saved in the temporary folder
    sensitivity_folder = tempfile.mkdtemp(suffix = '-sensitivities', prefix = 'tutorial20-')
    simulation.SensitivityDataDirectory = sensitivity_folder
    simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = True)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()
    
    # Run
    simulation.Run()
    simulation.Finalize()

    #############################################################
    # Plot the sensitivity of T per Qin using the data reporter #
    #############################################################
    # Get a dictionary with the reported variables
    variables = dr_data.Process.dictVariables

    # Get the daeDataReceiverVariable objects from the dictionary.
    # This class has properties such as TimeValues (ndarray with times) and Values (ndarray with values).
    dT_dQin_var = variables['tutorial20.sensitivities.d(T)_d(Qin)']

    # Time points can be taken from any variable (x axis)
    times = dT_dQin_var.TimeValues

    # Sensitivities (y axis)
    dT_dQin = dT_dQin_var.Values

    fontsize = 14
    fontsize_legend = 11
    
    plt.figure(figsize=(8,6), facecolor='white')
    plt.plot(times, dT_dQin, label=r'$\frac{\partial T(t)}{\partial Q_{in}}$')
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylabel('dT/dQin', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    run()
