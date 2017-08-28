#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_opt_2.py
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
COPS test 5: Isomerization of α-pinene (parameter estimation of a dynamic system).

Very slow convergence.

Determine the reaction coefficients in the thermal isometrization of α-pinene (y1) to
dipentene (y2) and allo-ocimen (y3) which in turn produces α- and β-pyronene (y4)
and a dimer (y5).

Reference: Benchmarking Optimization Software with COPS 3.0, Mathematics and Computer
Science Division, Argonne National Laboratory, Technical Report ANL/MCS-273, 2004.
`PDF <http://www.mcs.anl.gov/~more/cops/cops3.pdf>`_

Experimental data taken from:  Rocha A.M.A.C., Martins M.C., Costa M.F.P.,
Fernandes, E.M.G.P. (2016) Direct sequential based firefly algorithm for the α-pinene
isomerization problem. 16th International Conference on Computational Science and Its
Applications, ICCSA 2016, Beijing, China.
`doi:10.1007/978-3-319-42085-1_30 <http://doi.org/10.1007/978-3-319-42085-1_30>`_

Run options:

- Simulation with optimal parameters: python tutorial_che_opt_2.py simulation
- Parameter estimation console run:   python tutorial_che_opt_2.py console
- Parameter estimation GUI run:       python tutorial_che_opt_2.py gui

Currently, the parameter estimation results are (solver options/scaling should be tuned):

.. code-block:: none

    Fobj  57.83097
    p1    5.63514e-05
    p2    2.89711e-05
    p3    1.39979e-05
    p4   18.67874e-05
    p5    2.23770e-05

The concentration plots (for optimal 'p' from the literature):

.. image:: _static/tutorial_che_opt_2-results.png
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
#                 Isomerization of α-pinene
#########################################################
# Mathematical model
class modAlphaPinene(daeModel):
    p_scaling = 1e5

    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        # Reaction coefficients
        self.p1 = daeVariable("p1",  no_t, self, "Reaction coefficient 1")
        self.p2 = daeVariable("p2",  no_t, self, "Reaction coefficient 2")
        self.p3 = daeVariable("p3",  no_t, self, "Reaction coefficient 3")
        self.p4 = daeVariable("p4",  no_t, self, "Reaction coefficient 4")
        self.p5 = daeVariable("p5",  no_t, self, "Reaction coefficient 5")

        # State variables
        self.y1 = daeVariable("y1", y_t, self, "α-pinene concentration")
        self.y2 = daeVariable("y2", y_t, self, "Dipentene concentration")
        self.y3 = daeVariable("y3", y_t, self, "Allo-ocimen concentration")
        self.y4 = daeVariable("y4", y_t, self, "α- and β-pyronene concentration")
        self.y5 = daeVariable("y5", y_t, self, "Dimer concentration")

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        y1 = self.y1()
        y2 = self.y2()
        y3 = self.y3()
        y4 = self.y4()
        y5 = self.y5()
        p1 = self.p1()
        p2 = self.p2()
        p3 = self.p3()
        p4 = self.p4()
        p5 = self.p5()

        # Derivatives
        dy1_dt = self.y1.dt()
        dy2_dt = self.y2.dt()
        dy3_dt = self.y3.dt()
        dy4_dt = self.y4.dt()
        dy5_dt = self.y5.dt()

        # y1
        eq = self.CreateEquation("y1", "")
        eq.Residual = dy1_dt + (p1+p2)*y1 / modAlphaPinene.p_scaling
        eq.CheckUnitsConsistency = False

        # y2
        eq = self.CreateEquation("y2", "")
        eq.Residual = dy2_dt - p1*y1  / modAlphaPinene.p_scaling
        eq.CheckUnitsConsistency = False

        # y3
        eq = self.CreateEquation("y3", "")
        eq.Residual = dy3_dt - (p2*y1 - (p3+p4)*y3 + p5*y5) / modAlphaPinene.p_scaling
        eq.CheckUnitsConsistency = False

        # y4
        eq = self.CreateEquation("y4", "")
        eq.Residual = dy4_dt - p3*y3 / modAlphaPinene.p_scaling
        eq.CheckUnitsConsistency = False

        # y5
        eq = self.CreateEquation("y5", "")
        eq.Residual = dy5_dt - (p4*y3 - p5*y5) / modAlphaPinene.p_scaling
        eq.CheckUnitsConsistency = False

# Simulation (can be run independently from optimisation)
class simAlphaPinene(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modAlphaPinene("tutorial_che_opt_2")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # The reaction coefficients below are optimal results found in the literature.
        # They should produce L2 norm of 19.872393107.
        self.m.p1.AssignValue( 5.9256e-5 * modAlphaPinene.p_scaling)
        self.m.p2.AssignValue( 2.9632e-5 * modAlphaPinene.p_scaling)
        self.m.p3.AssignValue( 2.0450e-5 * modAlphaPinene.p_scaling)
        self.m.p4.AssignValue(27.4730e-5 * modAlphaPinene.p_scaling)
        self.m.p5.AssignValue( 4.0073e-5 * modAlphaPinene.p_scaling)

        self.m.y1.SetInitialCondition(y1_t0)
        self.m.y2.SetInitialCondition(y2_t0)
        self.m.y3.SetInitialCondition(y3_t0)
        self.m.y4.SetInitialCondition(y4_t0)
        self.m.y5.SetInitialCondition(y5_t0)

#########################################################
#               Parameter Estimation Part
#########################################################
# We need some additional variables to determine reaction coefficients.
# Derive a new class from modAlphaPinene and add extra data.
# Nota Bene:
#   modAlphaPinene_Opt inherits all parameters/variables from the base class.
class modAlphaPinene_Opt(modAlphaPinene):
    def __init__(self, Name, Parent = None, Description = ""):
        modAlphaPinene.__init__(self, Name, Parent, Description)

        # Observed values at the specific time interval
        self.y1_obs = daeVariable("y1_obs", no_t, self, "Observed value 1 at the specified time interval")
        self.y2_obs = daeVariable("y2_obs", no_t, self, "Observed value 2 at the specified time interval")
        self.y3_obs = daeVariable("y3_obs", no_t, self, "Observed value 3 at the specified time interval")
        self.y4_obs = daeVariable("y4_obs", no_t, self, "Observed value 4 at the specified time interval")
        self.y5_obs = daeVariable("y5_obs", no_t, self, "Observed value 5 at the specified time interval")

        # This L2 norm sums all L2 norms in the previous time intervals
        self.L2      = daeVariable("L2",      L2_t, self, "Current L2 norm: ||yi(t) - yi_obs(t)||^2")
        self.L2_prev = daeVariable("L2_prev", L2_t, self, "L2 norm in previous time intrvals")

    def DeclareEquations(self):
        modAlphaPinene.DeclareEquations(self)

        # L2-norm ||yi(t) - yi_obs(t)||^2
        # L2 norm is a sum of the L2 norm in the previous time steps (L2_prev)
        # and the current norm: s1 + s2 + s3 + s4 + s5.
        # L2_prev will be reset after every time interval where we have observed values.
        s1 = (self.y1() - self.y1_obs())**2
        s2 = (self.y2() - self.y2_obs())**2
        s3 = (self.y3() - self.y3_obs())**2
        s4 = (self.y4() - self.y4_obs())**2
        s5 = (self.y5() - self.y5_obs())**2
        eq = self.CreateEquation("L2", "")
        eq.Residual = self.L2() - (self.L2_prev() + s1 + s2 + s3 + s4 + s5)
        eq.CheckUnitsConsistency = False

# Simulation class that will be used by the optimisation.
class simAlphaPinene_opt(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modAlphaPinene_Opt("tutorial_che_opt_2")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # modAlphaPinene part
        self.m.p1.AssignValue( 5.9256e-5 * modAlphaPinene.p_scaling)
        self.m.p2.AssignValue( 2.9632e-5 * modAlphaPinene.p_scaling)
        self.m.p3.AssignValue( 2.0450e-5 * modAlphaPinene.p_scaling)
        self.m.p4.AssignValue(27.4730e-5 * modAlphaPinene.p_scaling)
        self.m.p5.AssignValue( 4.0073e-5 * modAlphaPinene.p_scaling)

        self.m.y1.SetInitialCondition(y1_t0)
        self.m.y2.SetInitialCondition(y2_t0)
        self.m.y3.SetInitialCondition(y3_t0)
        self.m.y4.SetInitialCondition(y4_t0)
        self.m.y5.SetInitialCondition(y5_t0)

        # Initialise variables required for parameter estimation.
        # Notate bene:
        #   Observed values should match initial conditions at t = 0
        #   L2_prev should be 0.0 initially
        self.m.y1_obs.AssignValue(y1_t0)
        self.m.y2_obs.AssignValue(y2_t0)
        self.m.y3_obs.AssignValue(y3_t0)
        self.m.y4_obs.AssignValue(y4_t0)
        self.m.y5_obs.AssignValue(y5_t0)
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
            self.m.y1_obs.ReAssignValue(y1_obs[t])
            self.m.y2_obs.ReAssignValue(y2_obs[t])
            self.m.y3_obs.ReAssignValue(y3_obs[t])
            self.m.y4_obs.ReAssignValue(y4_obs[t])
            self.m.y5_obs.ReAssignValue(y5_obs[t])

            # Reinitialise the DAE system after all changes made above
            self.Reinitialize()

            # Integrate, report data and set progress
            self.Log.Message('Integrating from %f to %f ...' % (self.CurrentTime, tn), 0)
            self.IntegrateUntilTime(tn, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))

    def SetUpOptimization(self):
        # Minimise L2-norm ||yi(t) - yi_obs(t)||^2
        self.ObjectiveFunction.Residual = self.m.L2()
        #self.ObjectiveFunction.AbsTolerance = 1e-6

        p_lb   = 0.0
        p_ub   = 5e-4 * modAlphaPinene.p_scaling
        p_init = 0.0

        p1 = self.SetContinuousOptimizationVariable(self.m.p1, p_lb, p_ub, p_init)
        p2 = self.SetContinuousOptimizationVariable(self.m.p2, p_lb, p_ub, p_init)
        p3 = self.SetContinuousOptimizationVariable(self.m.p3, p_lb, p_ub, p_init)
        p4 = self.SetContinuousOptimizationVariable(self.m.p4, p_lb, p_ub, p_init)
        p5 = self.SetContinuousOptimizationVariable(self.m.p5, p_lb, p_ub, p_init)
        """
        c1 = self.CreateInequalityConstraint("p1max") # p1 - UB <= 0
        c1.Residual = self.m.p1() - p_ub
        c2 = self.CreateInequalityConstraint("p1min") # LB - p1 <= 0
        c2.Residual = p_lb - self.m.p1()

        c1 = self.CreateInequalityConstraint("p2max") # p2 - UB <= 0
        c1.Residual = self.m.p2() - p_ub
        c2 = self.CreateInequalityConstraint("p2min") # LB - p2 <= 0
        c2.Residual = p_lb - self.m.p2()

        c1 = self.CreateInequalityConstraint("p3max") # p3 - UB <= 0
        c1.Residual = self.m.p3() - p_ub
        c2 = self.CreateInequalityConstraint("p3min") # LB - p3 <= 0
        c2.Residual = p_lb - self.m.p3()

        c1 = self.CreateInequalityConstraint("p4max") # p4 - UB <= 0
        c1.Residual = self.m.p4() - p_ub
        c2 = self.CreateInequalityConstraint("p4min") # LB - p4 <= 0
        c2.Residual = p_lb - self.m.p4()

        c1 = self.CreateInequalityConstraint("p5max") # p5 - UB <= 0
        c1.Residual = self.m.p5() - p_ub
        c2 = self.CreateInequalityConstraint("p5min") # LB - p5 <= 0
        c2.Residual = p_lb - self.m.p5()
        """
# Experimental data (8 measurements)
times  = numpy.array([1230.00, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0])
y1_obs = numpy.array([  88.35,   76.4,   65.1,   50.4,    37.5,    25.9,    14.0,     4.5])
y2_obs = numpy.array([   7.30,   15.6,   23.1,   32.9,    42.7,    49.1,    57.4,    63.1])
y3_obs = numpy.array([   2.30,    4.5,    5.3,    6.0,     6.0,     5.9,     5.1,     3.8])
y4_obs = numpy.array([   0.40,    0.7,    1.1,    1.5,     1.9,     2.2,     2.6,     2.9])
y5_obs = numpy.array([   1.75,    2.8,    5.8,    9.3,    12.0,    17.0,    21.0,    25.7])

# Initial conditions
y1_t0 = 100.0
y2_t0 =   0.0
y3_t0 =   0.0
y4_t0 =   0.0
y5_t0 =   0.0

def setOptions(nlpsolver):
    nlpsolver.SetOption('print_level', 0)
    nlpsolver.SetOption('tol', 1e-4)
    nlpsolver.SetOption('mu_strategy', 'adaptive')
    nlpsolver.SetOption('obj_scaling_factor', 1e-3)
    nlpsolver.SetOption('nlp_scaling_method', 'none') #'user-scaling')

def consoleSimulation():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simAlphaPinene()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingTimes = times.tolist()

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

def run(guiRun = False, qtApp = None):
    simulation = simAlphaPinene_opt()
    nlpsolver  = pyIPOPT.daeIPOPT()
    lasolver   = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    relativeTolerance = 1e-6
    reportingTimes = times.tolist()
    daeActivity.optimize(simulation, reportingInterval       = 1, 
                                     timeHorizon             = 1,
                                     reportingTimes          = reportingTimes,
                                     lasolver                = lasolver,
                                     nlpsolver               = nlpsolver,
                                     nlpsolver_setoptions_fn = setOptions,
                                     relativeTolerance       = relativeTolerance,
                                     guiRun                  = guiRun,
                                     qtApp                   = qtApp)

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'simulation'):
        consoleSimulation()
    else:
        guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
        run(guiRun)
