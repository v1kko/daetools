#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial21.py
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
This tutorial illustrates the additional sensitivity analysis features.
Here, the numerical sensitivities for the Constant coefficient first order equations 
are compared to the available analytical solution.

The sensitivity analysis is enabled and the sensitivities are reported to the data reporter.
The sensitivity data can be obtained in two ways:
    
- Directly from the DAE solver in the user-defined Run function using the
  DAESolver.SensitivityMatrix property.
- From the data reporter as any ordinary variable.

The comparison between the numerical and the analytical sensitivities:
    
.. image:: _static/tutorial21-results.png
   :width: 800px
"""
import os, sys, numpy, scipy, scipy.io
from time import localtime, strftime
from daetools.pyDAE import *
import matplotlib.pyplot as plt

class modTutorial(daeModel):
    def __init__(self,Name,Parent=None,Description=""):
        daeModel.__init__(self,Name,Parent,Description)

        self.p1      = daeVariable("p1",      no_t,   self,   "parameter1")
        self.p2      = daeVariable("p2",      no_t,   self,   "parameter2")
        self.y1      = daeVariable("y1",      no_t,   self,   "variable1")
        self.y2      = daeVariable("y2",      no_t,   self,   "variable2")
        self.y1a     = daeVariable("y1a",     no_t,   self,   "variable1 analytical")
        self.y2a     = daeVariable("y2a",     no_t,   self,   "variable2 analytical")
        self.dy1_dp1 = daeVariable("dy1_dp1", no_t,   self,   "dy1_dp1 analytical")
        self.dy1_dp2 = daeVariable("dy1_dp2", no_t,   self,   "dy1_dp2 analytical")
        self.dy2_dp1 = daeVariable("dy2_dp1", no_t,   self,   "dy2_dp1 analytical")
        self.dy2_dp2 = daeVariable("dy2_dp2", no_t,   self,   "dy2_dp2 analytical")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        a1 = 1.0
        a2 = 2.0
        a3 = 3.0
        a4 = 4.0

        c1 = a1*self.p1() + a2*self.p2()
        c2 = a3*self.p1() + a4*self.p2()
        exp_t = Exp(-Time())

        eq = self.CreateEquation("y1")
        eq.CheckUnitsConsistency = False
        eq.Residual = dt(self.y1()) + self.y1() + c1

        eq = self.CreateEquation("y2")
        eq.CheckUnitsConsistency = False
        eq.Residual = dt(self.y2()) + self.y2() + c2

        eq = self.CreateEquation("y1a")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.y1a() + c1 * (1 - exp_t)

        eq = self.CreateEquation("y2a")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.y2a() + c2 * (1 - exp_t)

        eq = self.CreateEquation("dy1_dp1")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy1_dp1() + a1 * (1 - exp_t)

        eq = self.CreateEquation("dy1_dp2")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy1_dp2() + a2 * (1 - exp_t)

        eq = self.CreateEquation("dy2_dp1")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy2_dp1() + a3 * (1 - exp_t)

        eq = self.CreateEquation("dy2_dp2")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy2_dp2() + a4 * (1 - exp_t)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial21")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.p1.AssignValue(1)
        self.m.p2.AssignValue(1)
        self.m.y1.SetInitialCondition(0)
        self.m.y2.SetInitialCondition(0)

    def SetUpSensitivityAnalysis(self):
        # order matters
        self.SetSensitivityParameter(self.m.p1)
        self.SetSensitivityParameter(self.m.p2)
        
    def Run(self):
        # The user-defined Run function can be used to access the sensitivites from the DAESolver.SensitivityMatrix
        
        # Concentrations block indexes required to access the data in the sensitivity matrix.
        # The property variable.BlockIndexes is ndarray with block indexes for all points in the variable.
        # If the variable is not distributed on domains then the BlockIndexes returns an integer.
        y1_bi = self.m.y1.BlockIndexes
        y2_bi = self.m.y2.BlockIndexes
        #print('Variable %s: overallIndex = %d, blockIndex = %d' % ('y1', self.m.y1.OverallIndex, y1_bi))
        #print('Variable %s: overallIndex = %d, blockIndex = %d' % ('y2', self.m.y2.OverallIndex, y2_bi))
        
        # Sensitivity parameters indexes
        p1_i = 0
        p2_i = 1
        
        times = []
        dy1_dp1 = []
        dy1_dp2 = []
        dy2_dp1 = []
        dy2_dp2 = []

        dy1_dp1_analytical = []
        dy1_dp2_analytical = []
        dy2_dp1_analytical = []
        dy2_dp2_analytical = []
        
        def addSensitivityData():
            # Sensitivity matrix as numpy array, which is 2D numpy array [Nparams, Nvariables]
            # Also the __call__ function from the sensitivity matrix could be used 
            # which is faster since it avoids copying the matrix data (i.e. see du1_dk2 below).
            sm   = self.DAESolver.SensitivityMatrix
            ndsm = sm.npyValues
            
            # Append the current time
            times.append(self.CurrentTime)
            
            # Append the sensitivities
            dy1_dp1.append(sm(p1_i, y1_bi))
            dy1_dp2.append(sm(p2_i, y1_bi))
            dy2_dp1.append(sm(p1_i, y2_bi))
            dy2_dp2.append(sm(p2_i, y2_bi))
            
            dy1_dp1_analytical.append(self.m.dy1_dp1.GetValue())
            dy1_dp2_analytical.append(self.m.dy1_dp2.GetValue())
            dy2_dp1_analytical.append(self.m.dy2_dp1.GetValue())
            dy2_dp2_analytical.append(self.m.dy2_dp2.GetValue())

        # Add sensitivities for time = 0
        addSensitivityData()
        
        # The default Run() function is re-implemented here (just the very basic version)
        # to be able to obtain the sensitivity matrix (faster than saving it to .mmx files and re-loading it)
        while self.CurrentTime < self.TimeHorizon:
            dt = self.ReportingInterval
            if self.CurrentTime+dt > self.TimeHorizon:
                dt = self.TimeHorizon - self.CurrentTime
            self.Log.Message('Integrating from [%.2f] to [%.2f] ...' % (self.CurrentTime, self.CurrentTime+dt), 0)
            self.IntegrateForTimeInterval(dt, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))
            
            # Add sensitivities for the current time
            addSensitivityData()
            
        self.Log.Message('The simulation has finished succesfully!', 0)
        
        fontsize = 14
        fontsize_suptitle = 16
        fontsize_legend = 11
        
        plt.figure(figsize=(8,4), facecolor='white')
        plt.suptitle('Sensitivities from DAESolver.SensitivityMatrix', fontsize=fontsize_suptitle)
        ax = plt.subplot(121)
        ax.set_title('Numerical sensitivities')
        plt.plot(times, dy1_dp1, label=r'$\frac{\partial y_1(t)}{\partial p_1}$')
        plt.plot(times, dy1_dp2, label=r'$\frac{\partial y_1(t)}{\partial p_2}$')
        plt.plot(times, dy2_dp1, label=r'$\frac{\partial y_2(t)}{\partial p_1}$')
        plt.plot(times, dy2_dp2, label=r'$\frac{\partial y_2(t)}{\partial p_2}$')
        plt.xlabel('Time (s)', fontsize=fontsize)
        plt.ylabel('dy/dp (-)', fontsize=fontsize)
        plt.legend(loc = 0, fontsize=fontsize_legend)
        plt.grid(b=True, which='both', color='0.65',linestyle='-')

        ax = plt.subplot(122)
        ax.set_title('Analytical sensitivities')
        plt.plot(times, dy1_dp1_analytical, label=r'$\frac{\partial y_1(t)}{\partial p_1}$')
        plt.plot(times, dy1_dp2_analytical, label=r'$\frac{\partial y_1(t)}{\partial p_2}$')
        plt.plot(times, dy2_dp1_analytical, label=r'$\frac{\partial y_2(t)}{\partial p_1}$')
        plt.plot(times, dy2_dp2_analytical, label=r'$\frac{\partial y_2(t)}{\partial p_2}$')
        plt.xlabel('Time (s)', fontsize=fontsize)
        plt.ylabel('dy/dp (-)', fontsize=fontsize)
        plt.legend(loc = 0, fontsize=fontsize_legend)
        plt.grid(b=True, which='both', color='0.65',linestyle='-')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

# Setup everything manually and run in a console
def run():
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    simulation   = simTutorial()
    datareporter = daeDelegateDataReporter()
    dr_tcpip     = daeTCPIPDataReporter()
    dr_data      = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_data)

    # Do no print progress
    log.PrintProgress = False

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True
    
    # Enable reporting of sensitivities for all reported variables
    simulation.ReportSensitivities = True
    
    simulation.ReportingInterval = 0.25
    simulation.TimeHorizon = 10

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if not dr_tcpip.Connect("", simName):
        sys.exit()

    # The .mmx files with the sensitivity matrices will not be saved in this example.
    #simulation.SensitivityDataDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sensitivities')
    simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = True)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    simulation.SolveInitial()

    simulation.Run()
    simulation.Finalize()
    
    ##############################################################################################
    # Plot the comparison between numerical and analytical sensitivities using the data reporter #
    ##############################################################################################
    # Get a dictionary with the reported variables
    variables = dr_data.Process.dictVariables

    # Auxiliary functions to get a variable or a sensitivity from the data reporter (as a daeDataReceiverVariable object).
    # daeDataReceiverVariable class has properties such as TimeValues (ndarray with times) and Values (ndarray with values).
    def sensitivity(variableName, parameterName): 
        return variables['tutorial21.sensitivities.d(%s)_d(%s)' % (variableName, parameterName)]
    def variable(variableName):
        return variables['tutorial21.%s' % variableName]

    # Time points can be taken from any variable (x axis)
    times = sensitivity('y1', 'p1').TimeValues

    # Get the daeDataReceiverVariable objects from the dictionary.
    # This class has properties such as TimeValues (ndarray with times) and Values (ndarray with values)
    dy1_dp1 = sensitivity('y1', 'p1').Values
    dy1_dp2 = sensitivity('y1', 'p2').Values
    dy2_dp1 = sensitivity('y2', 'p1').Values
    dy2_dp2 = sensitivity('y2', 'p2').Values

    dy1_dp1_analytical = variable('dy1_dp1').Values
    dy1_dp2_analytical = variable('dy1_dp2').Values
    dy2_dp1_analytical = variable('dy2_dp1').Values
    dy2_dp2_analytical = variable('dy2_dp2').Values
    
    fontsize = 14
    fontsize_suptitle = 16
    fontsize_legend = 11
    
    plt.figure(figsize=(8,4), facecolor='white')
    plt.suptitle('Sensitivities from DataReporter', fontsize=fontsize_suptitle)
    ax = plt.subplot(121)
    ax.set_title('Numerical sensitivities')
    plt.plot(times, dy1_dp1, label=r'$\frac{\partial y_1(t)}{\partial p_1}$')
    plt.plot(times, dy1_dp2, label=r'$\frac{\partial y_1(t)}{\partial p_2}$')
    plt.plot(times, dy2_dp1, label=r'$\frac{\partial y_2(t)}{\partial p_1}$')
    plt.plot(times, dy2_dp2, label=r'$\frac{\partial y_2(t)}{\partial p_2}$')
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylabel('dy/dp (-)', fontsize=fontsize)
    plt.legend(loc = 0, fontsize=fontsize_legend)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')

    ax = plt.subplot(122)
    ax.set_title('Analytical sensitivities')
    plt.plot(times, dy1_dp1_analytical, label=r'$\frac{\partial y_1(t)}{\partial p_1}$')
    plt.plot(times, dy1_dp2_analytical, label=r'$\frac{\partial y_1(t)}{\partial p_2}$')
    plt.plot(times, dy2_dp1_analytical, label=r'$\frac{\partial y_2(t)}{\partial p_1}$')
    plt.plot(times, dy2_dp2_analytical, label=r'$\frac{\partial y_2(t)}{\partial p_2}$')
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylabel('dy/dp (-)', fontsize=fontsize)
    plt.legend(loc = 0, fontsize=fontsize_legend)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__ == "__main__":
    run()
