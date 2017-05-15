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
Here, the sensitivity matrix for the Constant coefficient first order equations 
is obtained directly from the DAE solver and compared to the analytical solution.

The plot of numerical and analytical sensitivities:
    
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

        self.p1     = daeVariable("p1",     no_t,   self,   "parameter1")
        self.p2     = daeVariable("p2",     no_t,   self,   "parameter2")
        self.y1     = daeVariable("y1",     no_t,   self,   "variable1")
        self.y2     = daeVariable("y2",     no_t,   self,   "variable2")
        self.y1a    = daeVariable("y1a",    no_t,   self,   "variable1 analytical")
        self.y2a    = daeVariable("y2a",    no_t,   self,   "variable2 analytical")
        self.dy1_p1 = daeVariable("dy1_p1", no_t,   self,   "dy1_p1 analytical")
        self.dy1_p2 = daeVariable("dy1_p2", no_t,   self,   "dy1_p2 analytical")
        self.dy2_p1 = daeVariable("dy2_p1", no_t,   self,   "dy2_p1 analytical")
        self.dy2_p2 = daeVariable("dy2_p2", no_t,   self,   "dy2_p2 analytical")

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

        eq = self.CreateEquation("dy1_p1")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy1_p1() + a1 * (1 - exp_t)

        eq = self.CreateEquation("dy1_p2")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy1_p2() + a2 * (1 - exp_t)

        eq = self.CreateEquation("dy2_p1")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy2_p1() + a3 * (1 - exp_t)

        eq = self.CreateEquation("dy2_p2")
        eq.CheckUnitsConsistency = False
        eq.Residual = self.dy2_p2() + a4 * (1 - exp_t)

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
        oi_bi_map = self.m.OverallIndex_BlockIndex_VariableNameMap
        #print('Dictionary OverallIndex_BlockIndex_VariableNameMap:')
        #print(str(oi_bi_map))
        
        # Concentrations block indexes
        y1_bi  = oi_bi_map[self.m.y1.OverallIndex][0]
        y2_bi  = oi_bi_map[self.m.y2.OverallIndex][0]
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
            # Sensitivity matix as numpy array, which is 2D numpy array [Nparams, Nvariables]
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
            
            dy1_dp1_analytical.append(self.m.dy1_p1.GetValue())
            dy1_dp2_analytical.append(self.m.dy1_p2.GetValue())
            dy2_dp1_analytical.append(self.m.dy2_p1.GetValue())
            dy2_dp2_analytical.append(self.m.dy2_p2.GetValue())

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
        fontsize_legend = 11
        
        plt.figure(figsize=(8,4), facecolor='white')
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
        plt.show()

# Setup everything manually and run in a console
def run():
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    simulation   = simTutorial()
    datareporter = daeTCPIPDataReporter()

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
    if(datareporter.Connect("",simName)==False):
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

if __name__ == "__main__":
    run()
