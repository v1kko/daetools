#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import unittest
import tests_object, tests_domain, tests_parameter
import tests_jacobian_expressions, tests_numpy_ufuncs
import tests_dealii_feobject

if __name__ == "__main__":
    alltests = []
    loader = unittest.TestLoader()
    alltests.extend( loader.loadTestsFromModule(tests_object) )
    alltests.extend( loader.loadTestsFromModule(tests_domain) )
    alltests.extend( loader.loadTestsFromModule(tests_parameter) )
    alltests.extend( loader.loadTestsFromModule(tests_jacobian_expressions) )
    alltests.extend( loader.loadTestsFromModule(tests_numpy_ufuncs) )
    alltests.extend( loader.loadTestsFromModule(tests_dealii_feobject) )
    
    suite = unittest.TestSuite(alltests)
    textRunner = unittest.TextTestRunner(verbosity = 2)
    textRunner.run(suite)