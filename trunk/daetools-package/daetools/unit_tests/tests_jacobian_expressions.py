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
    def __init__(self, UseJacobianExpressions, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.UseJacobianExpressions = UseJacobianExpressions
        
        self.m      = daeParameter("m",       kg,            self, "Mass of the copper plate")
        self.cp     = daeParameter("c_p",     J/(kg*K),      self, "Specific heat capacity of the plate")
        self.alpha  = daeParameter("&alpha;", W/((m**2)*K),  self, "Heat transfer coefficient")
        self.A      = daeParameter("A",       m**2,          self, "Area of the plate")
        self.Tsurr  = daeParameter("T_surr",  K,             self, "Temperature of the surroundings")

        self.Qin    = daeVariable("Q_in",     power_t,       self, "Power of the heater")
        self.T      = daeVariable("T",        temperature_t, self, "Temperature of the plate")

        self.t0     = daeVariable("t00", no_t, self)
        self.t1     = daeVariable("t01", no_t, self)
        self.t2     = daeVariable("t02", no_t, self)
        self.t3     = daeVariable("t03", no_t, self)
        self.t4     = daeVariable("t04", no_t, self)
        self.t5     = daeVariable("t05", no_t, self)
        self.t6     = daeVariable("t06", no_t, self)
        self.t7     = daeVariable("t07", no_t, self)
        self.t8     = daeVariable("t08", no_t, self)
        self.t9     = daeVariable("t09", no_t, self)
        self.t10    = daeVariable("t10", no_t, self)
        self.t11    = daeVariable("t11", no_t, self)
        self.t12    = daeVariable("t12", no_t, self)
        self.t13    = daeVariable("t13", no_t, self)
        self.t14    = daeVariable("t14", no_t, self)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        self.expectedJacobianExpressions = {}
        # Not supprted functions: Abs, Min, Max, ExternalFunction (ACHTUNG, ACHTUNG!!)
        # Not implemented functions: Sinh, Cosh, Tanh, ASinh, ACosh, ATanh, ATan2 
        # Non-differentiable functions: Ceil, Floor
        
        eq = self.CreateEquation("HeatBalance")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())
        
        eq = self.CreateEquation("test_0")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t0() + self.T() * (self.Qin() - self.m() * self.cp() * self.T.dt()) / (self.m() * self.cp() * self.T() - self.T() ** 0.5)

        eq = self.CreateEquation("test_1")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t1() - self.T()

        eq = self.CreateEquation("test_2")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t2() * self.T()

        eq = self.CreateEquation("test_3")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t3() / self.T()

        eq = self.CreateEquation("test_4")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t4() + Sqrt(self.T())

        eq = self.CreateEquation("test_5")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t5() + Exp(self.T())

        eq = self.CreateEquation("test_6")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t6() + Log(self.T())

        eq = self.CreateEquation("test_7")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t7() + Log10(self.T())

        eq = self.CreateEquation("test_8")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t8() + Pow(self.T(), 2)

        eq = self.CreateEquation("test_9")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t9() + Sin(self.T())

        eq = self.CreateEquation("test_10")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t10() + Cos(self.T())

        eq = self.CreateEquation("test_11")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t11() #+ Tan(self.T())

        eq = self.CreateEquation("test_12")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t12() #+ ASin(self.T() / 283)

        eq = self.CreateEquation("test_13")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t13() #+ ACos(self.T() / 283)

        eq = self.CreateEquation("test_14")
        eq.BuildJacobianExpressions = self.UseJacobianExpressions
        eq.CheckUnitsConsistency    = False
        eq.Residual = self.t14() #+ ATan(self.T() / 283)
        
class simTutorial(daeSimulation):
    def __init__(self, UseJacobianExpressions):
        daeSimulation.__init__(self)
        self.m = modTutorial(UseJacobianExpressions, "tutorial")

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(283 * K)
        self.m.Qin.AssignValue(1500 * W)

def simulate(UseJacobianExpressions):
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeNoOpDataReporter()
    simulation   = simTutorial(UseJacobianExpressions)

    log.Enabled = False
    simulation.m.SetReportingOn(True)

    simulation.ReportingInterval = 10
    simulation.TimeHorizon       = 100

    # Only initialize the simulation - do not run it
    simulation.Initialize(daesolver, datareporter, log)
    simulation.SolveInitial()
    simulation.Run()
    simulation.Finalize()
    
    #for eq in simulation.m.Equations:
    #    name = eq.CanonicalName
    #    eei = eq.EquationExecutionInfos[0]
    #    if len(eei.JacobianExpressions) > 0:
    #        print name
    #        print eei.JacobianExpressions
    
    return datareporter
        
class case_JacobianExpressions(unittest.TestCase):
    def test_JacobianExpressions(self):
        # Do a simulation calculating a Jacobian in an ordinary way
        dr     = simulate(False)
        
        # Do the same simulation calculating a Jacobian using pre-built Jacobian expressions
        dr_jac = simulate(True)
        
        # Compare the results
        for v, v_jac in zip(dr.Process.Variables, dr_jac.Process.Variables):
            for val1, val2 in zip(v.Values, v_jac.Values):
                test = (abs(val1 - val2) < 1e-7)
                #if not test:
                #    print v.Name, abs(val1 - val2), val1, val2
                self.assertTrue(test)
        
if __name__ == "__main__":
    unittest.main()