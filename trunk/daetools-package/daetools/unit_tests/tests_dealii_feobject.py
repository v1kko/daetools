#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, numpy, unittest, pprint
daetools_root = os.path.abspath('../../')
if not daetools_root in sys.path:
    sys.path.insert(0, daetools_root)
from daetools.pyDAE import *
from pyDealII import *
import tests_object

"""
Tests for:
"""
       
class case_dealii_fe_object(unittest.TestCase):
    def test_assembly_expressions(self):
        expression = feExpression_2D
        constant   = constant_2D
        phi        = phi_2D
        dphi       = dphi_2D
        JxW        = JxW_2D
        xyz        = xyz_2D
        normal     = normal_2D
        fvalue     = function_value_2D
        grad       = function_gradient_2D

        x = constant(0.2) * JxW(fe_q)
        print constant(0.2) - 1.45 * feNumber.sin(x)
        print phi(fe_i, fe_q)*phi(fe_j, fe_q)*JxW(fe_q) + dphi(fe_i,fe_q)*dphi(fe_j,fe_q)
        print fvalue('D', xyz(fe_q)) + normal(fe_q) * fgrad('v', xyz(fe_q))

        context = getCellContext()
        print Evaluate( constant(0.2), context) 
        print Evaluate( phi(fe_i, fe_q), context) 
        print Evaluate( dphi(fe_i, fe_q), context) 
        print Evaluate( dphi(fe_i, fe_q)*dphi(fe_j, fe_q), context) 
        print Evaluate( xyz(fe_q) * dphi(fe_j, fe_q), context) 
    
        
if __name__ == "__main__":
    unittest.main()