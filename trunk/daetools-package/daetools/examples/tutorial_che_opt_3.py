#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_opt_3.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2016
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
COPS test 6: Marine Population Dynamics.

Given estimates of the abundance of the population of a marine species at each stage
(for example, nauplius, juvenile, adult) as a function of time, determine stage specific
growth and mortality rates.

Reference: Benchmarking Optimization Software with COPS 3.0, Mathematics and Computer
Science Division, Argonne National Laboratory, Technical Report ANL/MCS-273, 2004.
`PDF <http://www.mcs.anl.gov/~more/cops/cops3.pdf>`_

Experimental data generated following the procedure described in the COPS test.

Run options:

- Simulation with optimal parameters: python tutorial_che_opt_3.py simulation
- Parameter estimation console run:   python tutorial_che_opt_3.py console
- Parameter estimation GUI run:       python tutorial_che_opt_3.py gui

Currently, the parameter estimation results are (solver options/scaling should be tuned):

.. code-block:: none

   Fobj = e+7

The distribution moments 1,2,5,6 plots (for optimal results from the literature):

.. image:: _static/tutorial_che_opt_3-results.png
   :width: 500px

The distribution moments 3,4,7,8 plots (for optimal results from the literature):

.. image:: _static/tutorial_che_opt_3-results2.png
   :width: 500px
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
from daetools.solvers.ipopt import pyIPOPT
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l

y_t  = daeVariableType("y_t",  unit(), -1.0e+20, 1.0e+20, 0.0, 1e-06)
L2_t = daeVariableType("L2_t", unit(), -1.0e+20, 1.0e+20, 0.0, 1e-06)

#########################################################
#             Marine Population Dynamics
#########################################################
# Mathematical model
class modMarinePopulation(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Ns = daeDomain("Ns", self, unit(), "Number of species in the population")

        # Population growth/mortality rates
        self.m = daeVariable("m",  no_t, self, "Mortality rate", [self.Ns])
        self.g = daeVariable("g",  no_t, self, "Growth rate",    [self.Ns])

        # State variables
        self.y = daeVariable("y",  y_t, self, "Population moments", [self.Ns])

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        m = lambda i: self.m(i)
        g = lambda i: self.g(i)
        y = lambda i: self.y(i)

        # Derivatives
        dy_dt = lambda i: dt(self.y(i))

        # y[0]
        eq = self.CreateEquation("y0", "")
        eq.Residual = dy_dt(0) + (m(0) + g(0)) * y(0)
        eq.CheckUnitsConsistency = False

        Ns = self.Ns.NumberOfPoints
        # y[1-6]
        for j in range(1,Ns-1):
            eq = self.CreateEquation("y%d" % j, "")
            eq.Residual = dy_dt(j) - (g(j-1)*y(j-1) - (m(j) + g(j)) * y(j))
            eq.CheckUnitsConsistency = False

        # y[Ns-1]
        j = Ns-1
        eq = self.CreateEquation("y%d" % j, "")
        eq.Residual = dy_dt(j) - (g(j-1)*y(j-1) - m(j)*y(j))
        eq.CheckUnitsConsistency = False

# Simulation (can be run independently from optimisation)
class simMarinePopulation(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modMarinePopulation("tutorial_che_opt_3")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.Ns.CreateArray(8)

    def SetUpVariables(self):
        self.m.m.AssignValues(m_opt)
        self.m.g.AssignValues(g_opt)
        self.m.y.SetInitialConditions(y_t0)

#########################################################
#               Parameter Estimation Part
#########################################################
# We need some additional variables to determine reaction coefficients.
# Derive a new class from modMarinePopulation and add extra data.
# Nota Bene:
#   modMarinePopulation_Opt inherits all parameters/variables from the base class.
class modMarinePopulation_Opt(modMarinePopulation):
    def __init__(self, Name, Parent = None, Description = ""):
        modMarinePopulation.__init__(self, Name, Parent, Description)

        # Observed values at the specific time interval
        self.y1_obs = daeVariable("y1_obs", no_t, self, "Observed value 1 at the specified time interval")
        self.y2_obs = daeVariable("y2_obs", no_t, self, "Observed value 2 at the specified time interval")
        self.y3_obs = daeVariable("y3_obs", no_t, self, "Observed value 3 at the specified time interval")
        self.y4_obs = daeVariable("y4_obs", no_t, self, "Observed value 4 at the specified time interval")
        self.y5_obs = daeVariable("y5_obs", no_t, self, "Observed value 5 at the specified time interval")
        self.y6_obs = daeVariable("y6_obs", no_t, self, "Observed value 6 at the specified time interval")
        self.y7_obs = daeVariable("y7_obs", no_t, self, "Observed value 7 at the specified time interval")
        self.y8_obs = daeVariable("y8_obs", no_t, self, "Observed value 8 at the specified time interval")

        # This L2 norm sums all L2 norms in the previous time intervals
        self.L2      = daeVariable("L2",      L2_t, self, "Current L2 norm: ||yi(t) - yi_obs(t)||^2")
        self.L2_prev = daeVariable("L2_prev", L2_t, self, "L2 norm in previous time intrvals")

    def DeclareEquations(self):
        modMarinePopulation.DeclareEquations(self)

        # L2-norm ||yi(t) - yi_obs(t)||^2
        # L2 norm is a sum of the L2 norm in the previous time steps (L2_prev)
        # and the current norm: s1 + s2.
        # L2_prev will be reset after every time interval where we have observed values.
        s1 = (self.y(0) - self.y1_obs())**2
        s2 = (self.y(1) - self.y2_obs())**2
        s3 = (self.y(2) - self.y3_obs())**2
        s4 = (self.y(3) - self.y4_obs())**2
        s5 = (self.y(4) - self.y5_obs())**2
        s6 = (self.y(5) - self.y6_obs())**2
        s7 = (self.y(6) - self.y7_obs())**2
        s8 = (self.y(7) - self.y8_obs())**2
        eq = self.CreateEquation("L2", "")
        eq.Residual = self.L2() - (self.L2_prev() + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8)
        eq.CheckUnitsConsistency = False

# Simulation class that will be used by the optimisation.
class simMarinePopulation_opt(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modMarinePopulation_Opt("tutorial_che_opt_3")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.Ns.CreateArray(8)

    def SetUpVariables(self):
        # modMarinePopulation part
        self.m.m.AssignValues(m_opt)
        self.m.g.AssignValues(g_opt)
        self.m.y.SetInitialConditions(y_t0)

        # Initialise variables required for parameter estimation.
        # Notate bene:
        #   Observed values should match initial conditions at t = 0
        #   L2_prev should be 0.0 initially
        self.m.y1_obs.AssignValue(y_t0[0])
        self.m.y2_obs.AssignValue(y_t0[1])
        self.m.y3_obs.AssignValue(y_t0[2])
        self.m.y4_obs.AssignValue(y_t0[3])
        self.m.y5_obs.AssignValue(y_t0[4])
        self.m.y6_obs.AssignValue(y_t0[5])
        self.m.y7_obs.AssignValue(y_t0[6])
        self.m.y8_obs.AssignValue(y_t0[7])
        self.m.L2_prev.AssignValue(0.0)

    def Run(self):
        for t, tn in enumerate(times):
            # Reset L2_prev value to the current L2
            if t == 0:
                self.m.L2_prev.ReAssignValue(0.0)
            else:
                L2 = self.m.L2.GetValue()
                self.m.L2_prev.ReAssignValue(L2)

            # Reset observed values to match the current interval end time
            self.m.y1_obs.ReAssignValue(y_obs[t,0])
            self.m.y2_obs.ReAssignValue(y_obs[t,1])
            self.m.y3_obs.ReAssignValue(y_obs[t,2])
            self.m.y4_obs.ReAssignValue(y_obs[t,3])
            self.m.y5_obs.ReAssignValue(y_obs[t,4])
            self.m.y6_obs.ReAssignValue(y_obs[t,5])
            self.m.y7_obs.ReAssignValue(y_obs[t,6])
            self.m.y8_obs.ReAssignValue(y_obs[t,7])

            # Reinitialise the DAE system after all changes made above
            self.Reinitialize()

            # Integrate, report data and set progress
            self.Log.Message('Integrating from %f to %f ...' % (self.CurrentTime, tn), 0)
            self.IntegrateUntilTime(tn, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))

    def SetUpOptimization(self):
        # Minimise L2-norm ||yi(t) - yi_obs(t)||^2
        self.ObjectiveFunction.Residual = self.m.L2() * 1e-8 # scale Fobj
        #self.ObjectiveFunction.AbsTolerance = 1e-6

        m_lb   =  0.0
        m_ub   = 10.0
        m_init =  1.0
        m1 = self.SetContinuousOptimizationVariable(self.m.m(0), m_lb, m_ub, m_init)
        m2 = self.SetContinuousOptimizationVariable(self.m.m(1), m_lb, m_ub, m_init)
        m3 = self.SetContinuousOptimizationVariable(self.m.m(2), m_lb, m_ub, m_init)
        m4 = self.SetContinuousOptimizationVariable(self.m.m(3), m_lb, m_ub, m_init)
        m5 = self.SetContinuousOptimizationVariable(self.m.m(4), m_lb, m_ub, m_init)
        m6 = self.SetContinuousOptimizationVariable(self.m.m(5), m_lb, m_ub, m_init)
        m7 = self.SetContinuousOptimizationVariable(self.m.m(6), m_lb, m_ub, m_init)
        m8 = self.SetContinuousOptimizationVariable(self.m.m(7), m_lb, m_ub, m_init)

        g_lb   =  0.0
        g_ub   = 10.0
        g_init =  1.0
        g1 = self.SetContinuousOptimizationVariable(self.m.g(0), g_lb, g_ub, g_init)
        g2 = self.SetContinuousOptimizationVariable(self.m.g(1), g_lb, g_ub, g_init)
        g3 = self.SetContinuousOptimizationVariable(self.m.g(2), g_lb, g_ub, g_init)
        g4 = self.SetContinuousOptimizationVariable(self.m.g(3), g_lb, g_ub, g_init)
        g5 = self.SetContinuousOptimizationVariable(self.m.g(4), g_lb, g_ub, g_init)
        g6 = self.SetContinuousOptimizationVariable(self.m.g(5), g_lb, g_ub, g_init)
        g7 = self.SetContinuousOptimizationVariable(self.m.g(6), g_lb, g_ub, g_init)
        #g8 = self.SetContinuousOptimizationVariable(self.m.g(7), g_lb, g_ub, g_init)

        def constraint(p, LB, UB, name):
            return
            c1 = self.CreateInequalityConstraint("%smax" % name) # p - UB <= 0
            c1.Residual = p - UB
            c2 = self.CreateInequalityConstraint("%smin" % name) # LB - p <= 0
            c2.Residual = LB - p

        for i in range(8):
            constraint(self.m.g(i), g_lb, g_ub, 'g')
        for i in range(7):
            constraint(self.m.m(i), m_lb, m_ub, 'm')

# Experimental data (21 measurements) generated by the simulation below
times  = numpy.array([1e-10, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                      6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
# yobs[species,time]
y_obs = numpy.array([
    20000, 17000, 10000, 15000, 12000, 9000, 7000, 3000,
    12445, 15411, 13040, 13338, 13484, 8426, 6615, 4022,
     7705, 13074, 14623, 11976, 12453, 9272, 6891, 5020,
     4664,  8579, 12434, 12603, 11738, 9710, 6821, 5722,
     2977,  7053, 11219, 11340, 13665, 8534, 6242, 5695,
     1769,  5054, 10065, 11232, 12112, 9600, 6647, 7034,
      943,  3907,  9473, 10334, 11115, 8826, 6842, 7348,
      581,  2624,  7421, 10297, 12427, 8747, 7199, 7684,
      355,  1744,  5369,  7748, 10057, 8698, 6542, 7410,
      223,  1272,  4713,  6869,  9564, 8766, 6810, 6961,
      137,   821,  3451,  6050,  8671, 8291, 6827, 7525,
       87,   577,  2649,  5454,  8430, 7411, 6423, 8388,
       49,   337,  2058,  4115,  7435, 7627, 6268, 7189,
       32,   228,  1440,  3790,  6474, 6658, 5859, 7467,
       17,   168,  1178,  3087,  6524, 5880, 5562, 7144,
       11,    99,   919,  2596,  5360, 5762, 4480, 7256,
        7,    65,   647,  1873,  4556, 5058, 4944, 7538,
        4,    44,   509,  1571,  4009, 4527, 4233, 6649,
        2,    27,   345,  1227,  3677, 4229, 3805, 6378,
        1,    20,   231,   934,  3197, 3695, 3159, 6454,
        1,    12,   198,   707,  2562, 3163, 3232, 5566], dtype=float).reshape((21,8))
# Initial conditions
y_t0 = numpy.array([2.0e4, 1.7e4, 1.0e4, 1.5e4, 1.2e4, 0.9e4, 0.7e4, 0.3e4])
# Approximate optimal rates
m_opt = numpy.array([0.28, 0.10, 0.25, 0.12, 1e-3, 1e-9, 0.32, 0.43])
g_opt = numpy.array([0.70, 0.81, 0.47, 0.48, 0.49, 0.65, 0.54, 0.0])

# Use daeSimulator class
def guiRun(app):
    sim = simMarinePopulation_opt()
    opt = daeOptimization()
    nlp = pyIPOPT.daeIPOPT()
    sim.m.SetReportingOn(True)
    sim.ReportingTimes = times.tolist()
    simulator = daeSimulator(app, simulation = sim,
                                  optimization = opt,
                                  nlpsolver = nlp,
                                  nlpsolver_setoptions_fn = setOptions)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    nlpsolver    = pyIPOPT.daeIPOPT()
    datareporter = daeNoOpDataReporter()
    simulation   = simMarinePopulation_opt()
    optimization = daeOptimization()
    lasolver     = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    daesolver.SetLASolver(lasolver)

    daesolver.RelativeTolerance = 1e-6

    # Do no print progress
    log.PrintProgress = True
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingTimes = times.tolist()

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the optimization
    optimization.Initialize(simulation, nlpsolver, daesolver, datareporter, log)

    # Achtung! Achtung! NLP solver options can only be set after optimization.Initialize()
    # Otherwise seg. fault occurs for some reasons.
    nlpsolver.SetOption('print_level', 0)
    nlpsolver.SetOption('tol', 1e-5)
    nlpsolver.SetOption('mu_strategy', 'adaptive')
    nlpsolver.SetOption('obj_scaling_factor', 1e3)
    nlpsolver.SetOption('nlp_scaling_method', 'none') #'user-scaling')

    # Run
    optimization.Run()
    optimization.Finalize()

def consoleSimulation():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simMarinePopulation_opt()

    dr_tcpip     = daeTCPIPDataReporter()
    dr_data      = daeNoOpDataReporter()
    datareporter.AddDataReporter(dr_tcpip)
    datareporter.AddDataReporter(dr_data)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10.0/20
    simulation.TimeHorizon       = 10.0

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(dr_tcpip.Connect("", simName) == False):
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
    elif len(sys.argv) > 1 and (sys.argv[1] == 'simulation'):
        consoleSimulation()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
