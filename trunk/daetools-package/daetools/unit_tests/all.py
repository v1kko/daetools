#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import tests_object, tests_domain

if __name__ == "__main__":
    alltests = []
    loader = unittest.TestLoader()
    alltests.extend( loader.loadTestsFromModule(tests_object) )
    alltests.extend( loader.loadTestsFromModule(tests_domain) )
    
    suite = unittest.TestSuite(alltests)
    textRunner = unittest.TextTestRunner(verbosity = 2)
    textRunner.run(suite)