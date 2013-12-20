#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
python_major = sys.version_info[0]

import unittest
if python_major == 2:
    import tests_object, tests_domain, tests_parameter
    import tests_jacobian_expressions, tests_numpy_ufuncs
    import tests_dealii_feobject
elif python_major == 3:
    from . import tests_object, tests_domain, tests_parameter
    from . import tests_jacobian_expressions, tests_numpy_ufuncs
    from . import tests_dealii_feobject

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