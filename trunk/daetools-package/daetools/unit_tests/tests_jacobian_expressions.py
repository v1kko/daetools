#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, numpy, unittest, pprint
daetools_root = os.path.abspath('../../')
if not daetools_root in sys.path:
    sys.path.insert(0, daetools_root)
from daetools.pyDAE import *
import tests_object

"""
Tests for:
"""

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m      = daeParameter("m",       kg,            self, "Mass of the copper plate")
        self.cp     = daeParameter("c_p",     J/(kg*K),      self, "Specific heat capacity of the plate")
        self.alpha  = daeParameter("&alpha;", W/((m**2)*K),  self, "Heat transfer coefficient")
        self.A      = daeParameter("A",       m**2,          self, "Area of the plate")
        self.Tsurr  = daeParameter("T_surr",  K,             self, "Temperature of the surroundings")

        self.Qin    = daeVariable("Q_in",     power_t,       self, "Power of the heater")
        self.T      = daeVariable("T",        temperature_t, self, "Temperature of the plate")

        self.t0     = daeVariable("t0",  no_t, self)
        self.t1     = daeVariable("t1",  no_t, self)
        self.t2     = daeVariable("t2",  no_t, self)
        self.t3     = daeVariable("t3",  no_t, self)
        self.t4     = daeVariable("t4",  no_t, self)
        self.t5     = daeVariable("t5",  no_t, self)
        self.t6     = daeVariable("t6",  no_t, self)
        self.t7     = daeVariable("t7",  no_t, self)
        self.t8     = daeVariable("t8",  no_t, self)
        self.t9     = daeVariable("t9",  no_t, self)
        self.t10    = daeVariable("t10", no_t, self)
        self.t11    = daeVariable("t11", no_t, self)
        self.t12    = daeVariable("t12", no_t, self)
        self.t13    = daeVariable("t13", no_t, self)
        self.t14    = daeVariable("t14", no_t, self)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        self.expectedJacobianExpressions = {}
        
        eq = self.CreateEquation("HeatBalance")
        eq.BuildJacobianExpressions = True
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['((tutorial.m() * tutorial.c_p()) * (InverseTimeStep())) + ((tutorial.alpha() * tutorial.A()) * Constant(1))']
        
        eq = self.CreateEquation("test_0")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t0() + self.T()
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1)', 'Constant(1)']

        eq = self.CreateEquation("test_1")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t1() - self.T()
        self.expectedJacobianExpressions[eq.CanonicalName] = ['(-Constant(1))', 'Constant(1)']

        eq = self.CreateEquation("test_2")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t2() * self.T()
        self.expectedJacobianExpressions[eq.CanonicalName] = ['tutorial.t2() * Constant(1)', 'tutorial.T() * Constant(1)']

        eq = self.CreateEquation("test_3")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t3() / self.T()
        self.expectedJacobianExpressions[eq.CanonicalName] = ['((-tutorial.t3() * Constant(1))) / (tutorial.T() * tutorial.T())', 'Constant(1) / tutorial.T()']

        eq = self.CreateEquation("test_4")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t4() + Sqrt(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1) / (Constant(2) * Sqrt(tutorial.T()))', 'Constant(1)']

        eq = self.CreateEquation("test_5")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t5() + Exp(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Exp(tutorial.T()) * Constant(1)', 'Constant(1)']

        eq = self.CreateEquation("test_6")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t6() + Log(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1) / tutorial.T()', 'Constant(1)']

        eq = self.CreateEquation("test_7")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t7() + Log10(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1) / (Log10(Constant(10)) * tutorial.T())', 'Constant(1)']

        eq = self.CreateEquation("test_8")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t8() + Pow(self.T(), 2)
        self.expectedJacobianExpressions[eq.CanonicalName] = ['(Constant(2) * (tutorial.T() ** (Constant(2) - Constant(1)))) * Constant(1)', 'Constant(1)']

        eq = self.CreateEquation("test_9")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t9() + Sin(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Cos(tutorial.T()) * Constant(1)', 'Constant(1)']

        eq = self.CreateEquation("test_10")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t10() + Cos(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['((-Sin(tutorial.T()))) * Constant(1)', 'Constant(1)']

        eq = self.CreateEquation("test_11")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t11() + Tan(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1) / (Cos(tutorial.T()) ** Constant(2))', 'Constant(1)']

        eq = self.CreateEquation("test_12")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t12() + ASin(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1) / Sqrt(Constant(1) - (ASin(tutorial.T()) ** Constant(2)))', 'Constant(1)']

        eq = self.CreateEquation("test_13")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t13() + ACos(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1) / ((-Sqrt(Constant(1) - (ACos(tutorial.T()) ** Constant(2)))))', 'Constant(1)']

        eq = self.CreateEquation("test_14")
        eq.BuildJacobianExpressions = True
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t14() + ATan(self.T())
        self.expectedJacobianExpressions[eq.CanonicalName] = ['Constant(1) / (Constant(1) + (ATan(tutorial.T()) ** Constant(2)))', 'Constant(1)']
            
        
        """
        Not supported functions:
          - Sinh
          - Cosh
          - Tanh
          - ASinh
          - ACosh
          - ATanh
          - ATan2
          - Abs
          - Ceil
          - Floor
          - Min, Max
        """

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial")

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(283 * K)
        self.m.Qin.AssignValue(1500 * W)

class case_JacobianExpressions(unittest.TestCase):
    def test_JacobianExpressions(self):
        # Setup a simulation and initialize it
        log          = daePythonStdOutLog()
        daesolver    = daeIDAS()
        datareporter = daeBlackHoleDataReporter()
        simulation   = simTutorial()

        log.Enabled = False
        simulation.m.SetReportingOn(False)

        simulation.ReportingInterval = 10
        simulation.TimeHorizon       = 100

        # Only initialize the simulation - do not run it
        simulation.Initialize(daesolver, datareporter, log)
        
        expressions = {}
        for eq in simulation.m.Equations:
            name = eq.CanonicalName
            expressions[name] = []
            eeis = eq.EquationExecutionInfos
            
            # Since the equations are not distributed the number of equation exec. infos must be 1
            self.assertTrue(len(eeis) == 1)
            
            eei = eeis[0]
            for i, expression in enumerate(eei.JacobianExpressions):
                expressions[name].append(str(expression))
                
                # Evaluated Jacobian expressions must be equal to the expected ones
                self.assertEqual(str(expression), simulation.m.expectedJacobianExpressions[name][i])
                
            #print name
            #print(expressions[name])
            #print(simulation.m.expectedJacobianExpressions[name])
        
if __name__ == "__main__":
    unittest.main()