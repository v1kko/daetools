#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                            tutorial_sa_2.py
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
********************************************************************************"""
__doc__ = """
This tutorial illustrates the local derivative-based sensitivity analysis method 
available in DAE Tools.

The problem is adopted from the section 2.1 of the following article:
    
- A. Saltelli, M. Ratto, S. Tarantola, F. Campolongo.
  Sensitivity Analysis for Chemical Models. Chem. Rev. (2005), 105(7):2811-2828.
  `doi:10.1021/cr040659d <http://dx.doi.org/10.1021/cr040659d>`_
  
The model is very simple and describes a simple reversible chemical reaction A <-> B, 
with reaction rates k1 and k_1 for the direct and inverse reactions, respectively.
The reaction rates are uncertain and are described by continuous random variables 
with known probability density functions. The standard deviation is 0.3 for k1 and
1 for k_1. The standard deviation of the concentration of the species A is
approximated using the following expression defined in the article:
    
.. code-block:: none  

   stddev(Ca)**2 = stddev(k1)**2 * (dCa/dk1)**2 + stddev(k_1)**2 * (dCa/dk_1)**2

The following derivative-based measures are used in the article:

- Derivatives dCa/dk1 and dCa/dk_1 calculated using the forward sensitivity method
- Sigma normalised derivatives:
    
  .. code-block:: none  
  
     S(k1)  = stddev(k1) / stddev(Ca) * dCa/dk1
     S(k_1) = stddev(k_1)/ stddev(Ca) * dCa/dk_1
    
The plot of the concentrations, derivatives and sigma normalised derivatives:
    
.. image:: _static/tutorial_sa_2-results.png
   :width: 800px
"""

import sys, numpy
from daetools.pyDAE import *
from time import localtime, strftime
# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l
import matplotlib.pyplot as plt

k_t                  = daeVariableType("k_t",                    s**(-1),  0.0, 1E20, 0, 1e-05)
mass_concentration_t = daeVariableType("mass_concentration_t", kg/(m**3),  0.0, 1E20, 0, 1e-05)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.stddev_k1  = 0.3
        self.stddev_k_1 = 1.0

        self.Ca0 = daeVariable("Ca0", mass_concentration_t, self, "Initial concentration of the reactant A")
        self.Cb0 = daeVariable("Cb0", mass_concentration_t, self, "Initial concentration of the reactant B")

        self.Ca = daeVariable("Ca", mass_concentration_t, self, "Concentration of the reactant A")
        self.Cb = daeVariable("Cb", mass_concentration_t, self, "Concentration of the reactant B")

        self.k1  = daeVariable("k1",  k_t,  self, "Reaction rate constant")
        self.k_1 = daeVariable("k_1", k_t,  self, "Reverse reaction rate constant")
        
        # Dummy variable to make the model dynamic
        self.tau = daeVariable("&tau;", time_t, self, "Time elapsed in the process")

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        k1  = self.k1()
        k_1 = self.k_1()
        Ca  = self.Ca()
        Ca0 = self.Ca0()
        Cb  = self.Cb()
        t   = Time()
        
        # Reaction rate constants
        eq = self.CreateEquation("Ca", "")
        eq.Residual = Ca - (Ca0 / (k1 + k_1) * (k1 * Exp(-(k1 + k_1)*t) + k_1))

        eq = self.CreateEquation("Cb", "")
        eq.Residual = Cb - (Ca0 - Ca)
        
        # Dummy equation to make the model dynamic
        eq = self.CreateEquation("Time", "")
        eq.Residual = dt(self.tau()) - 1.0

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_sa_2")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass
    
    def SetUpVariables(self):
        self.m.k1.AssignValue(3.0)
        self.m.k_1.AssignValue(3.0)
        self.m.Ca0.AssignValue(1.0)
        self.m.Cb0.AssignValue(0.0)
        self.m.tau.SetInitialCondition(0.0)

    def SetUpSensitivityAnalysis(self):
        # order matters
        self.SetSensitivityParameter(self.m.k1)
        self.SetSensitivityParameter(self.m.k_1)

def run():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()

    # Do no print progress
    log.PrintProgress = True

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True
    
    # Enable reporting of sensitivities for all reported variables
    simulation.ReportSensitivities = True

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.025 # 1.5 min
    simulation.TimeHorizon       = 0.5   # 0.5 hour

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
    simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = True)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()
    
    ###########################################
    #  Data                                   #
    ###########################################
    # Get a dictionary with the reported variables
    variables = dr.Process.dictVariables
    
    # Auxiliary functions
    def sensitivity(variableName, parameterName): 
        return variables['tutorial_sa_2.sensitivities.d(%s)_d(%s)' % (variableName, parameterName)]
    def variable(variableName):
        return variables['tutorial_sa_2.%s' % variableName]

    Ca_var       = variable('Ca')
    Cb_var       = variable('Cb')
    dCa_dk1_var  = sensitivity('Ca', 'k1')
    dCa_dk_1_var = sensitivity('Ca', 'k_1')

    times      = Ca_var.TimeValues
    Ca         = Ca_var.Values[:]
    Cb         = Cb_var.Values[:]
    
    # Absolute values of the derivative based sensitivities:
    dCa_dk1    = numpy.abs(dCa_dk1_var.Values[:])
    dCa_dk_1   = numpy.abs(dCa_dk_1_var.Values[:])
    
    # Standard deviations of k1, k_1 and Ca:
    stddev_k1  = simulation.m.stddev_k1
    stddev_k_1 = simulation.m.stddev_k_1
    stddev_Ca  = numpy.sqrt(stddev_k1**2 * dCa_dk1**2 + stddev_k_1**2 * dCa_dk_1**2)
    stddev_Ca[0] = 1e-20 # to avoid division by zero
    
    # A dimensionless version of the derivative based sensitivities (sigma normalised):
    Sk1  = (stddev_k1 /stddev_Ca) * dCa_dk1
    Sk_1 = (stddev_k_1/stddev_Ca) * dCa_dk_1
    
    # Plot Ca, Cb and sensitivities
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(10,6), facecolor='white')
    
    ax = plt.subplot(221)
    plt.figure(1, facecolor='white')
    plt.plot(times, Ca, 'b-', label='Ca')
    plt.plot(times, Cb, 'r-', label='Cb')
    plt.xlabel('time', fontsize=fontsize)
    plt.ylabel('C', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
        
    ax = plt.subplot(223)
    plt.figure(1, facecolor='white')
    plt.plot(times[1:], dCa_dk1[1:],  'b-', label='d(Ca)/d(k1)')
    plt.plot(times[1:], dCa_dk_1[1:], 'b:', label='d(Ca)/d(k-1)')
    plt.xlabel('time', fontsize=fontsize)
    plt.ylabel('dC/dk', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
        
    ax = plt.subplot(224)
    plt.figure(1, facecolor='white')
    plt.plot(times[1:], Sk1[1:],  'g-', label='Sk1')
    plt.plot(times[1:], Sk_1[1:], 'g:', label='Sk-1')
    plt.xlabel('time', fontsize=fontsize)
    plt.ylabel('Sk', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
