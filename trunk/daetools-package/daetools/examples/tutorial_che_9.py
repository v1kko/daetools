#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_9.py
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
Chemical reaction network from the Dow Chemical Company described in the following article:

- Caracotsios M., Stewart W.E. (1985) Sensitivity analysis of initial value problems 
  with mixed odes and algebraic equations. Computers & Chemical Engineering 9(4):359-365.
  `doi:10.1016/0098-1354(85)85014-6 <https://doi.org/10.1016/0098-1354(85)85014-6>`_
  
The sensitivity analysis is enabled and the sensitivities are reported to the data reporter.
The sensitivity data can be obtained in two ways:
    
- Directly from the DAE solver in the user-defined Run function using the
  DAESolver.SensitivityMatrix property.
- From the data reporter as any ordinary variable.

The concentrations plot (u1, u3, u4):

.. image:: _static/tutorial_che_9-results1.png
   :width: 500px
  
The concentrations plot (u6, u8):

.. image:: _static/tutorial_che_9-results2.png
   :width: 500px
  
The sensitivities plot (k2*du1/dk2, k2*du2/dk2, k2*du3/dk2, k2*du4/dk2, k2*du5/dk2):

.. image:: _static/tutorial_che_9-results3.png
   :width: 700px
"""
import os, sys, math, numpy, scipy, scipy.io
from time import localtime, strftime
from daetools.pyDAE import *
import matplotlib.pyplot as plt

# Note the absolute tolerance for u_t (the variabe type for u(t) variables)!!
# Note the units for k_t and kn_t (seconds - not hours)
u_t   = daeVariableType("u_t",      mol/kg,    -1.0e+20, 1.0e+20,  0.0, 1e-10)
k_t   = daeVariableType("k_t",  kg/(mol*s),    -1.0e+20, 1.0e+20,  0.0, 1e-5)
kn_t  = daeVariableType("kn_t",    s**(-1),    -1.0e+20, 1.0e+20,  0.0, 1e-5)
Keq_t = daeVariableType("u_t",      mol/kg,    -1.0e+20, 1.0e+20,  0.0, 1e-5)

class modTutorial(daeModel):
    def __init__(self,Name,Parent=None,Description=""):
        daeModel.__init__(self,Name,Parent,Description)

        # Rate constants, kg/(mol*s) | 1/s
        self.k1     = daeVariable("k1",  k_t,   self,   "k1")
        self.k1n    = daeVariable("k1n", kn_t,  self,   "k1n")
        self.k2     = daeVariable("k2",  k_t,   self,   "k2")
        self.k3     = daeVariable("k3",  k_t,   self,   "k3")
        self.k3n    = daeVariable("k3n", kn_t,  self,   "k3n")

        # Equilibrium constants, mol/kg
        self.Keq1   = daeVariable("Keq1", Keq_t,   self,   "Keq1")
        self.Keq2   = daeVariable("Keq2", Keq_t,   self,   "Keq2")
        self.Keq3   = daeVariable("Keq3", Keq_t,   self,   "Keq3")

        # Concentrations, mol/kg
        self.u1     = daeVariable("u1", u_t,   self,   "u1")
        self.u2     = daeVariable("u2", u_t,   self,   "u2")
        self.u3     = daeVariable("u3", u_t,   self,   "u3")
        self.u4     = daeVariable("u4", u_t,   self,   "u4")
        self.u5     = daeVariable("u5", u_t,   self,   "u5")
        self.u6     = daeVariable("u6", u_t,   self,   "u6")
        self.u7     = daeVariable("u7", u_t,   self,   "u7")
        self.u8     = daeVariable("u8", u_t,   self,   "u8")
        self.u9     = daeVariable("u9", u_t,   self,   "u9")
        self.u10    = daeVariable("u10",u_t,   self,   "u10")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Create some auxiliary adouble objects to make equations more readable 
        k1  = self.k1()
        k1n = self.k1n()
        k2  = self.k2()
        k3  = self.k3()
        k3n = self.k3n()
        
        Keq1 = self.Keq1()
        Keq2 = self.Keq2()
        Keq3 = self.Keq3()

        u1 = self.u1()
        u2 = self.u2()
        u3 = self.u3()
        u4 = self.u4()
        u5 = self.u5()
        u6 = self.u6()
        u7 = self.u7()
        u8 = self.u8()
        u9 = self.u9()
        u10 = self.u10()

        eq = self.CreateEquation("E1")
        eq.Residual = dt(u1) - (-k2*u2*u8)

        eq = self.CreateEquation("E2")
        eq.Residual = dt(u2) - (-k1*u2*u6 + k1n*u10 - k2*u2*u8)

        eq = self.CreateEquation("E3")
        eq.Residual = dt(u3) - (k2*u2*u8 + k3*u4*u6 - k3n*u9)

        eq = self.CreateEquation("E4")
        eq.Residual = dt(u4) - (-k3*u4*u6 + k3n*u9)

        eq = self.CreateEquation("E5")
        eq.Residual = dt(u5) - (k1*u2*u6 - k1n*u10)

        eq = self.CreateEquation("E6")
        eq.Residual = dt(u6) - (-k1*u2*u6 - k3*u4*u6 + k1n*u10 + k3n*u9)

        eq = self.CreateEquation("E7")
        eq.Residual = u7 - (Constant(-0.0131*mol/kg) + u6 + u8 + u9 + u10)

        eq = self.CreateEquation("E8")
        eq.Residual = u8 - Keq2*u1 / (Keq2 + u7)

        eq = self.CreateEquation("E9")
        eq.Residual = u9 - Keq3*u3 / (Keq3 + u7)

        eq = self.CreateEquation("E10")
        eq.Residual = u10 - Keq1*u5 / (Keq1 + u7)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_che_9")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        u1 = 1.5776
        Keq1 = 7.65e-18
        Keq2 = 4.03e-11
        Keq3 = 5.32e-18
        u7 = 0.5 * (-Keq2 + math.sqrt(Keq2**2 + 4*Keq2*u1))
        #print('u7(t=0) = %f\n' % u7)

        # Note: the original values are in kg/(mol*hour) and 1/hour
        # Units can be omitted but it is better to use quantities and not simple floats
        self.m.k1.AssignValue(21.893  / 3600 * kg/(mol*s))
        self.m.k1n.AssignValue(2.14e9 / 3600 * s**(-1))
        self.m.k2.AssignValue(32.318  / 3600 * kg/(mol*s))
        self.m.k3.AssignValue(21.893  / 3600 * kg/(mol*s))
        self.m.k3n.AssignValue(1.07e9 / 3600 * s**(-1))
        
        self.m.Keq1.AssignValue(Keq1 * mol/kg)
        self.m.Keq2.AssignValue(Keq2 * mol/kg)
        self.m.Keq3.AssignValue(Keq3 * mol/kg)
        
        self.m.u1.SetInitialCondition(u1     * mol/kg)
        self.m.u2.SetInitialCondition(8.32   * mol/kg)
        self.m.u3.SetInitialCondition(0      * mol/kg)
        self.m.u4.SetInitialCondition(0      * mol/kg)
        self.m.u5.SetInitialCondition(0      * mol/kg)
        self.m.u6.SetInitialCondition(0.0131 * mol/kg)
        
        # Initial guesses to help the solver
        self.m.u7.SetInitialGuess(u7 * mol/kg)
        self.m.u8.SetInitialGuess(u7 * mol/kg)
        self.m.u9.SetInitialGuess(0  * mol/kg)
        self.m.u10.SetInitialGuess(0 * mol/kg)

    def SetUpSensitivityAnalysis(self):
        # order matters
        self.SetSensitivityParameter(self.m.k1)
        self.SetSensitivityParameter(self.m.k1n)
        self.SetSensitivityParameter(self.m.k2)
        self.SetSensitivityParameter(self.m.k3)
        self.SetSensitivityParameter(self.m.k3n)
     
    
    def Run(self):
        # The user-defined Run function can be used to access the sensitivites from the DAESolver.SensitivityMatrix
        # The default Run() function is re-implemented here (just the very basic version)
        # to be able to obtain the sensitivity matrix at every reporting interval.
        
        # Concentrations block indexes required to access the data in the sensitivity matrix.
        # The property variable.BlockIndexes is ndarray with block indexes for all points in the variable.
        # If the variable is not distributed on domains then the BlockIndexes returns an integer.
        u1_bi  = self.m.u1.BlockIndexes
        u2_bi  = self.m.u2.BlockIndexes
        u3_bi  = self.m.u3.BlockIndexes
        u4_bi  = self.m.u4.BlockIndexes
        u5_bi  = self.m.u5.BlockIndexes
        #print('Variable %s: overallIndex = %d, blockIndex = %d' % ('u1', self.m.u1.OverallIndex, u1_bi))
        #print('Variable %s: overallIndex = %d, blockIndex = %d' % ('u2', self.m.u2.OverallIndex, u2_bi))
        #print('Variable %s: overallIndex = %d, blockIndex = %d' % ('u3', self.m.u3.OverallIndex, u3_bi))
        #print('Variable %s: overallIndex = %d, blockIndex = %d' % ('u4', self.m.u4.OverallIndex, u4_bi))
        #print('Variable %s: overallIndex = %d, blockIndex = %d' % ('u5', self.m.u5.OverallIndex, u5_bi))
        
        # Sensitivity parameters indexes
        k1_i  = 0
        k1n_i = 1
        k2_i  = 2
        k3_i  = 3
        k3n_i = 4
        
        times = []
        du1_dk2 = []
        du2_dk2 = []
        du3_dk2 = []
        du4_dk2 = []
        du5_dk2 = []
        
        while self.CurrentTime < self.TimeHorizon:
            dt = self.ReportingInterval
            if self.CurrentTime+dt > self.TimeHorizon:
                dt = self.TimeHorizon - self.CurrentTime
            self.Log.Message('Integrating from [%.1f] to [%.1f] ...' % (self.CurrentTime, self.CurrentTime+dt), 0)
            self.IntegrateForTimeInterval(dt, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))
            
            # Get the system's full sensitivity matrix [Nparams, Nvariables] as a daeDenseMatrix object.
            sm   = self.DAESolver.SensitivityMatrix
            # Sensitivity matrix as a 2D numpy array [Nparams, Nvariables].
            ndsm = sm.npyValues
            
            # Append the current time
            times.append(self.CurrentTime)
            
            # Append the sensitivities per k2 (Figure 6. in the article)
            # Items are accessed using the sensitivity parameters' indexes and variables' block indexes
            # using the __call__ function from the daeDenseMatrix class or from the numpy array.
            # Nota bene:
            #   Using daeDenseMatrix is faster since it does not copy the matrix data into the numpy array.
            du1_dk2.append(sm(k2_i, u1_bi))   # use daeDenseMatrix.__call__ function
            du2_dk2.append(ndsm[k2_i, u2_bi]) # use numpy array
            du3_dk2.append(ndsm[k2_i, u3_bi])
            du4_dk2.append(ndsm[k2_i, u4_bi])
            du5_dk2.append(ndsm[k2_i, u5_bi])

        self.Log.Message('The simulation has finished succesfully!', 0)
        
        # Transform time into hours
        times = numpy.array(times) / 3600
        
        # Multiply all results with k2
        k2 = self.m.k2.GetValue()
        du1_dk2 = k2 * numpy.array(du1_dk2)
        du2_dk2 = k2 * numpy.array(du2_dk2)
        du3_dk2 = k2 * numpy.array(du3_dk2)
        du4_dk2 = k2 * numpy.array(du4_dk2)
        du5_dk2 = k2 * numpy.array(du5_dk2)

        fontsize = 14
        fontsize_suptitle = 16
        fontsize_legend = 11
        
        plt.figure(figsize=(8,6), facecolor='white')
        plt.suptitle('Sensitivities from the DAESolver.SensitivityMatrix', fontsize=fontsize_suptitle)
        plt.plot(times, du1_dk2, label=r'$k_2 \frac{\partial u_1(t)}{\partial k_2}$')
        plt.plot(times, du2_dk2, label=r'$k_2 \frac{\partial u_2(t)}{\partial k_2}$')
        plt.plot(times, du3_dk2, label=r'$k_2 \frac{\partial u_3(t)}{\partial k_2}$')
        plt.plot(times, du4_dk2, label=r'$k_2 \frac{\partial u_4(t)}{\partial k_2}$')
        plt.plot(times, du5_dk2, label=r'$k_2 \frac{\partial u_5(t)}{\partial k_2}$')
        plt.xlabel('Time (hr)', fontsize=fontsize)
        plt.ylabel('k2*du/dk', fontsize=fontsize)
        plt.legend(fontsize=fontsize_legend)
        plt.grid(b=True, which='both', color='0.65',linestyle='-')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    
def run():
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    simulation   = simTutorial()
    datareporter = daeDelegateDataReporter()
    dr_tcpip     = daeTCPIPDataReporter()
    dr_data      = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_data)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Enable reporting of time derivatives for all reported variables
    simulation.ReportTimeDerivatives = True

    # Enable reporting of sensitivities for all reported variables
    simulation.ReportSensitivities = True

    simulation.ReportingInterval =  1*60   #  1 min
    simulation.TimeHorizon       = 10*3600 # 10 hr

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if not dr_tcpip.Connect("",simName):
        sys.exit()

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

    ###############################################################################################################
    # Plot the Figure 6. from the article using the data reporter (sensitivities of u1, u2, u3, u4 and u5 per k2) #
    ###############################################################################################################
    # Get a dictionary with the reported variables
    variables = dr_data.Process.dictVariables

    # An auxiliary function to format a name of a sensitivity variable
    def sensitivity_variable_name(variable, parameter): 
        return 'tutorial_che_9.sensitivities.d(%s)_d(%s)' % (variable, parameter)
    k2_name = 'tutorial_che_9.k2'

    # Get the daeDataReceiverVariable objects from the dictionary.
    # This class has properties such as TimeValues (ndarray with times) and Values (ndarray with values)
    k2_var      = variables[k2_name]
    du1_dk2_var = variables[sensitivity_variable_name('u1', 'k2')]
    du2_dk2_var = variables[sensitivity_variable_name('u2', 'k2')]
    du3_dk2_var = variables[sensitivity_variable_name('u3', 'k2')]
    du4_dk2_var = variables[sensitivity_variable_name('u4', 'k2')]
    du5_dk2_var = variables[sensitivity_variable_name('u5', 'k2')]
    
    # Transform time points into hours (x axis)
    times = k2_var.TimeValues / 3600
    
    # Get sensitivities du/dk2 and multiply them with k2 (y axis)
    k2      = k2_var.Values[0]
    du1_dk2 = k2 * du1_dk2_var.Values
    du2_dk2 = k2 * du2_dk2_var.Values
    du3_dk2 = k2 * du3_dk2_var.Values
    du4_dk2 = k2 * du4_dk2_var.Values
    du5_dk2 = k2 * du5_dk2_var.Values

    fontsize = 14
    fontsize_suptitle = 16
    fontsize_legend = 11
    
    plt.figure(figsize=(8,6), facecolor='white')
    plt.suptitle('Sensitivities from the DataReporter', fontsize=fontsize_suptitle)
    plt.plot(times, du1_dk2, label=r'$k_2 \frac{\partial u_1(t)}{\partial k_2}$')
    plt.plot(times, du2_dk2, label=r'$k_2 \frac{\partial u_2(t)}{\partial k_2}$')
    plt.plot(times, du3_dk2, label=r'$k_2 \frac{\partial u_3(t)}{\partial k_2}$')
    plt.plot(times, du4_dk2, label=r'$k_2 \frac{\partial u_4(t)}{\partial k_2}$')
    plt.plot(times, du5_dk2, label=r'$k_2 \frac{\partial u_5(t)}{\partial k_2}$')
    plt.xlabel('Time (hr)', fontsize=fontsize)
    plt.ylabel('k2*du/dk', fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    run()
