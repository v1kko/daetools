#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, numpy, unittest, pprint
python_major  = sys.version_info[0]
daetools_root = os.path.abspath('../../')
if not daetools_root in sys.path:
    sys.path.insert(0, daetools_root)
from daetools.pyDAE import *
if python_major == 2:
    import tests_object
elif python_major == 3:
    from . import tests_object

"""
Tests for:
"""

class modTutorial(daeModel):
    def __init__(self, n, useNumpy, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.N        = n
        self.useNumpy = useNumpy

        self.x      = daeDomain("x", self, m, "")
        self.p      = daeParameter("p", unit(), self, "", [self.x])

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
        self.t15    = daeVariable("t15", no_t, self)
        self.t16    = daeVariable("t16", no_t, self)
        self.t17    = daeVariable("t17", no_t, self)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Supported functions:
        # - operators +, -, *, /, **
        # - sqrt, exp, log, log10, sin, cos, tan, arcsin, arccos, arctan, fabs, ceil, floor
        # - sum, cumsum[], prod, cumprod[], diff, ediff1d, cross, trapz, square
        """
        res = u ** 3
        print res
        res = numpy.sum(u)
        print res
        res = numpy.cumsum(u)
        print res
        res = numpy.prod(u)
        print res
        res = numpy.cumprod(u)
        print res
        res = numpy.diff(u)
        print res
        res = numpy.ediff1d(u)
        print res
        res = numpy.cross(u[0:3], u[0:3]) # dimensions of vectors must be 2 or 3
        print res
        res = numpy.trapz(u, [0.1, 0.2, 0.4, 0.7, 1.1])
        print res
        res = numpy.square(u)
        print res
        """

        du = self.p.array('*')
        
        u = numpy.empty(self.N, dtype=object)
        u[:] = [self.p(i) for i in range(self.N)]
        
        # sum
        eq = self.CreateEquation("test_0")
        if self.useNumpy:
            eq.Residual = self.t0() + numpy.sum(u)
        else:
            eq.Residual = self.t0() + Sum(du)

        eq = self.CreateEquation("test_1")
        if self.useNumpy:
            eq.Residual = self.t1() + numpy.prod(u)
        else:
            eq.Residual = self.t1() + Product(du)

        eq = self.CreateEquation("test_2")
        if self.useNumpy:
            eq.Residual = self.t2() + numpy.sum(numpy.sqrt(u))
        else:
            eq.Residual = self.t2() + Sum(Sqrt(du))

        eq = self.CreateEquation("test_3")
        if self.useNumpy:
            eq.Residual = self.t3() + numpy.sum(numpy.exp(u))
        else:
            eq.Residual = self.t3() + Sum(Exp(du))

        eq = self.CreateEquation("test_4")
        if self.useNumpy:
            eq.Residual = self.t4() + numpy.sum(numpy.log(u))
        else:
            eq.Residual = self.t4() + Sum(Log(du))

        eq = self.CreateEquation("test_5")
        if self.useNumpy:
            eq.Residual = self.t5() + numpy.sum(numpy.log10(u))
        else:
            eq.Residual = self.t5() + Sum(Log10(du))

        eq = self.CreateEquation("test_6")
        if self.useNumpy:
            eq.Residual = self.t6() + numpy.sum(numpy.sin(u))
        else:
            eq.Residual = self.t6() + Sum(Sin(du))

        eq = self.CreateEquation("test_7")
        if self.useNumpy:
            eq.Residual = self.t7() + numpy.sum(numpy.cos(u))
        else:
            eq.Residual = self.t7() + Sum(Cos(du))

        eq = self.CreateEquation("test_8")
        if self.useNumpy:
            eq.Residual = self.t8() + numpy.sum(numpy.tan(u))
        else:
            eq.Residual = self.t8() + Sum(Tan(du))

        eq = self.CreateEquation("test_9")
        if self.useNumpy:
            eq.Residual = self.t9() + numpy.sum(numpy.fabs(u))
        else:
            eq.Residual = self.t9() + Sum(Abs(du))

        eq = self.CreateEquation("test_10")
        if self.useNumpy:
            eq.Residual = self.t10() + numpy.sum(numpy.ceil(u))
        else:
            eq.Residual = self.t10() + Sum(Ceil(du))

        eq = self.CreateEquation("test_11")
        if self.useNumpy:
            eq.Residual = self.t11() + numpy.sum(numpy.floor(u))
        else:
            eq.Residual = self.t11() + Sum(Floor(du))

        eq = self.CreateEquation("test_12")
        if self.useNumpy:
            eq.Residual = self.t12() + numpy.sum(u + u)
        else:
            eq.Residual = self.t12() + Sum(du + du)

        eq = self.CreateEquation("test_13")
        if self.useNumpy:
            eq.Residual = self.t13() + numpy.sum(u - u)
        else:
            eq.Residual = self.t13() + Sum(du - du)

        eq = self.CreateEquation("test_14")
        if self.useNumpy:
            eq.Residual = self.t14() + numpy.sum(u * u)
        else:
            eq.Residual = self.t14() + Sum(du * du)

        eq = self.CreateEquation("test_15")
        if self.useNumpy:
            eq.Residual = self.t15() + numpy.sum(u / u)
        else:
            eq.Residual = self.t15() + Sum(du / du)

        eq = self.CreateEquation("test_16")
        if self.useNumpy:
            eq.Residual = self.t16() + numpy.sum(numpy.reciprocal(u))
        else:
            eq.Residual = self.t16() + Sum(1 / du)
        
        eq = self.CreateEquation("test_17")
        if self.useNumpy:
            eq.Residual = self.t17() + numpy.sum(numpy.negative(u))
        else:
            eq.Residual = self.t17() + Sum(-du)
        
class simTutorial(daeSimulation):
    def __init__(self, n, useNumpy):
        daeSimulation.__init__(self)
        self.m = modTutorial(n, useNumpy, "tutorial")

    def SetUpParametersAndDomains(self):
        self.m.x.CreateArray(self.m.N)
        for x in range(0, self.m.x.NumberOfPoints):
            self.m.p.SetValue(x, float(x + 0.1))

    def SetUpVariables(self):
        pass

def simulate(n, useNumpy):
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeNoOpDataReporter()
    simulation   = simTutorial(n, useNumpy)

    log.Enabled = False
    simulation.m.SetReportingOn(True)

    simulation.ReportingInterval = 0
    simulation.TimeHorizon       = 0

    # Only initialize the simulation - do not run it
    simulation.Initialize(daesolver, datareporter, log)
    simulation.SolveInitial()
    simulation.Run()
    simulation.Finalize()
    
    return datareporter
        
class case_numpy_ufuncs(unittest.TestCase):
    def test_numpy_ufuncs(self):
        # Do a simulation using daetools functions
        dr = simulate(5, False)
        
        # Do the same simulation using numpy ufuncs
        dr_np = simulate(5, True)
        
        # Compare the results
        for v, v_np in zip(dr.Process.Variables, dr_np.Process.Variables):
            for values1, values2 in zip(v.Values, v_np.Values):
                if isinstance(values1, numpy.float64):
                    test = (abs(val1 - val2) < 1e-7)
                    if not test:
                        print v.Name, abs(val1 - val2), val1, val2
                    self.assertTrue(test)
                else:
                    for val1, val2 in zip(values1, values2):
                        test = (abs(val1 - val2) < 1e-7)
                        if not test:
                            print v.Name, abs(val1 - val2), val1, val2
                        self.assertTrue(test)
        
if __name__ == "__main__":
    unittest.main()