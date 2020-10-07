#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, numpy, unittest
daetools_root = os.path.abspath('../../')
if not daetools_root in sys.path: 
    sys.path.insert(0, daetools_root)
from daetools.pyDAE import *
import tests_object

"""
Tests for:
 - daeParameter (array, distributed domain, operators () and [])
"""

class case_daeParameter(unittest.TestCase):
    def _createParameter(self):
        name      = 'Parameter'
        descr     = 'Parameter description'
        canonName = 'Model.Parameter'
        unit      = (pyUnits.m**2)*(pyUnits.s**-2) / (pyUnits.kg**1)
        parent    = daeModel('Model', None, '')
        param     = daeParameter(name, unit, parent, descr)

        return param, name, canonName, descr, parent, unit

    def _createDomain(self, parent, suffix):
        name      = 'Domain' + suffix
        descr     = 'Domain description' + suffix
        canonName = 'Model.Domain' + suffix
        unit      = (pyUnits.m**2)*(pyUnits.s**-2) / (pyUnits.kg**1)
        domain    = daeDomain(name, parent, unit, descr)

        return domain

    def test_daeObject(self):
        param, name, canonName, descr, parent, unit = self._createParameter()
        tests_object.test_daeObjectInterface(self, param, name, canonName, descr, parent)

    def test_NonDistributed(self):
        param, name, canonName, descr, parent, unit = self._createParameter()

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 1.0)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0)

        # Not initialized yet!
        with self.assertRaises(RuntimeError):
            param.GetValue()
        with self.assertRaises(RuntimeError):
            x = param()

        # Wrong argument type (should be float)!
        with self.assertRaises(TypeError):
            param.SetValue(None)
        with self.assertRaises(TypeError):
            param.SetValue([])

        # SetValue function
        value = 2.0
        param.SetValue(value)
        self.assertEqual(param.GetValue(), value)

        # SetValues function
        value = 1.0
        param.SetValues(value)
        self.assertEqual(param.GetValue(), value)

        # Numpy array
        ndValues = value #numpy.array([value])

        self.assertEqual(param.Domains,        [])
        self.assertEqual(param.Units,          unit)
        self.assertEqual(param.NumberOfPoints, 1)
        self.assertTrue(numpy.array_equal(param.npyValues, ndValues))

        # Default is False
        self.assertEqual(param.ReportingOn, False)
        # Test (re)seting
        param.ReportingOn = False
        self.assertEqual(param.ReportingOn, False)
        param.ReportingOn = True
        self.assertEqual(param.ReportingOn, True)

        return
        index = 3
        value = points[index]

        ad   = d[index] # adouble
        node = ad.Node  # adDomainIndexNode

        self.assertEqual(ad.Value,      0.0)
        self.assertEqual(ad.Derivative, 0.0)

        self.assertEqual(node.Index,     index)
        self.assertEqual(node.Domain.ID, d.ID)
        self.assertEqual(node.Value,     value)

    def test_Distributed1(self):
        param, name, canonName, descr, parent, unit = self._createParameter()
        d1 = self._createDomain(parent, '')

        # Initialize domain
        d1.CreateArray(3)

        # Wrong argument type (should be daeDomain)!
        with self.assertRaises(TypeError):
            param.DistributeOnDomain(None)
        with self.assertRaises(TypeError):
            param.DistributeOnDomain([])

        # Distribute on domain
        param.DistributeOnDomain(d1)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0)
        #with self.assertRaises(RuntimeError):
        #    x = param(0)
        with self.assertRaises(RuntimeError):
            x = param()

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 1.0)
        #with self.assertRaises(RuntimeError):
        #    param.SetValue(0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(1.0)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0)
        #with self.assertRaises(RuntimeError):
        #    param.GetValue(0)
        with self.assertRaises(RuntimeError):
            param.GetValue()

        # Not initialized yet!
        with self.assertRaises(RuntimeError):
            param.GetValue(0)
        with self.assertRaises(RuntimeError):
            x = param(0)

        # Wrong argument type (should be float)!
        with self.assertRaises(TypeError):
            param.SetValue(0, None)
        with self.assertRaises(TypeError):
            param.SetValue(0, [])

        # SetValue function
        value = 2.0
        for i1 in range(d1.NumberOfPoints):
            param.SetValue(i1, value)

        for i1 in range(d1.NumberOfPoints):
            self.assertEqual(param.GetValue(i1), value)

        # SetValues function
        value = 1.0
        param.SetValues(value)
        for i1 in range(d1.NumberOfPoints):
            self.assertEqual(param.GetValue(i1), value)

        # Numpy array
        noPoints = d1.NumberOfPoints
        values   = [1, 2, 3]
        ndValues = numpy.array(values)
        for i1 in range(d1.NumberOfPoints):
            param.SetValue(i1, values[i1])

        self.assertEqual(len(param.Domains),        1)
        self.assertEqual(param.Units,               unit)
        self.assertEqual(param.NumberOfPoints,      noPoints)
        self.assertTrue(numpy.array_equal(param.npyValues, ndValues))

    def test_Distributed2(self):
        param, name, canonName, descr, parent, unit = self._createParameter()
        d1 = self._createDomain(parent, '1')
        d2 = self._createDomain(parent, '2')

        # Initialize domain
        d1.CreateArray(3)
        d2.CreateArray(4)

        # Wrong argument type (should be daeDomain)!
        with self.assertRaises(TypeError):
            param.DistributeOnDomain(None)
        with self.assertRaises(TypeError):
            param.DistributeOnDomain([])

        # Distribute on domains
        param.DistributeOnDomain(d1)
        param.DistributeOnDomain(d2)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0)
        #with self.assertRaises(RuntimeError):
        #    x = param(0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0)
        with self.assertRaises(RuntimeError):
            x = param()

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 1.0)
        #with self.assertRaises(RuntimeError):
        #    param.SetValue(0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(1.0)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0)
        #with self.assertRaises(RuntimeError):
        #    param.GetValue(0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0)
        with self.assertRaises(RuntimeError):
            param.GetValue()

        # Not initialized yet!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0)

        # Wrong argument type (should be float)!
        with self.assertRaises(TypeError):
            param.SetValue(0, 0, None)
        with self.assertRaises(TypeError):
            param.SetValue(0, 0, [])

        # SetValue function
        value = 2.0
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                param.SetValue(i1, i2, value)

        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                self.assertEqual(param.GetValue(i1, i2), value)

        # SetValues function
        value = 1.0
        param.SetValues(value)
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                self.assertEqual(param.GetValue(i1, i2), value)

        # Numpy array
        noPoints = d1.NumberOfPoints * d2.NumberOfPoints
        ndValues = numpy.zeros((d1.NumberOfPoints, d2.NumberOfPoints))
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                ndValues[i1][i2] = i1 * 1.E1 + i2

        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                param.SetValue(i1, i2, float(ndValues[i1][i2]))

        #print ndValues
        #print param.npyValues

        self.assertEqual(len(param.Domains),        2)
        self.assertEqual(param.Units,               unit)
        self.assertEqual(param.NumberOfPoints,      noPoints)
        self.assertTrue(numpy.array_equal(param.npyValues, ndValues))

    def test_Distributed3(self):
        param, name, canonName, descr, parent, unit = self._createParameter()
        d1 = self._createDomain(parent, '1')
        d2 = self._createDomain(parent, '2')
        d3 = self._createDomain(parent, '3')

        # Initialize domain
        d1.CreateArray(3)
        d2.CreateArray(4)
        d3.CreateArray(5)

        # Wrong argument type (should be daeDomain)!
        with self.assertRaises(TypeError):
            param.DistributeOnDomain(None)
        with self.assertRaises(TypeError):
            param.DistributeOnDomain([])

        # Distribute on domains
        param.DistributeOnDomain(d1)
        param.DistributeOnDomain(d2)
        param.DistributeOnDomain(d3)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0)
        #with self.assertRaises(RuntimeError):
        #    x = param(0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0)
        with self.assertRaises(RuntimeError):
            x = param()

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 1.0)
        #with self.assertRaises(RuntimeError):
        #    param.SetValue(0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(1.0)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0)
        #with self.assertRaises(RuntimeError):
        #    param.GetValue(0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0)
        with self.assertRaises(RuntimeError):
            param.GetValue()

        # Not initialized yet!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0)

        # Wrong argument type (should be float)!
        with self.assertRaises(TypeError):
            param.SetValue(0, 0, 0, None)
        with self.assertRaises(TypeError):
            param.SetValue(0, 0, 0, [])

        # SetValue function
        value = 2.0
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    param.SetValue(i1, i2, i3, value)

        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    self.assertEqual(param.GetValue(i1, i2, i3), value)

        # SetValues function
        value = 1.0
        param.SetValues(value)
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    self.assertEqual(param.GetValue(i1, i2, i3), value)

        # Numpy array
        noPoints = d1.NumberOfPoints * d2.NumberOfPoints * d3.NumberOfPoints
        ndValues = numpy.zeros((d1.NumberOfPoints, d2.NumberOfPoints, d3.NumberOfPoints))
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    ndValues[i1][i2][i3] = i1 * 1E2 + i2 * 1E1 + i3

        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    param.SetValue(i1, i2, i3, float(ndValues[i1][i2][i3]))

        #print ndValues
        #print param.npyValues

        self.assertEqual(len(param.Domains),        3)
        self.assertEqual(param.Units,               unit)
        self.assertEqual(param.NumberOfPoints,      noPoints)
        self.assertTrue(numpy.array_equal(param.npyValues, ndValues))


    def test_Distributed4(self):
        param, name, canonName, descr, parent, unit = self._createParameter()
        d1 = self._createDomain(parent, '1')
        d2 = self._createDomain(parent, '2')
        d3 = self._createDomain(parent, '3')
        d4 = self._createDomain(parent, '4')

        # Initialize domain
        d1.CreateArray(3)
        d2.CreateArray(4)
        d3.CreateArray(5)
        d4.CreateArray(5)

        # Wrong argument type (should be daeDomain)!
        with self.assertRaises(TypeError):
            param.DistributeOnDomain(None)
        with self.assertRaises(TypeError):
            param.DistributeOnDomain([])

        # Distribute on domains
        param.DistributeOnDomain(d1)
        param.DistributeOnDomain(d2)
        param.DistributeOnDomain(d3)
        param.DistributeOnDomain(d4)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0, 0)
        #with self.assertRaises(RuntimeError):
        #    x = param(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0)
        with self.assertRaises(RuntimeError):
            x = param()

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 0, 0, 1.0)
        #with self.assertRaises(RuntimeError):
        #    param.SetValue(0, 0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(0, 1.0)
        with self.assertRaises(RuntimeError):
            param.SetValue(1.0)

        # Wrong number of arguments!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0, 0)
        #with self.assertRaises(RuntimeError):
        #    param.GetValue(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0)
        with self.assertRaises(RuntimeError):
            param.GetValue(0)
        with self.assertRaises(RuntimeError):
            param.GetValue()

        # Not initialized yet!
        with self.assertRaises(RuntimeError):
            param.GetValue(0, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            x = param(0, 0, 0, 0)

        # Wrong argument type (should be float)!
        with self.assertRaises(TypeError):
            param.SetValue(0, 0, 0, 0, None)
        with self.assertRaises(TypeError):
            param.SetValue(0, 0, 0, 0, [])

        # SetValue function
        value = 2.0
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    for i4 in range(d4.NumberOfPoints):
                        param.SetValue(i1, i2, i3, i4, value)

        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    for i4 in range(d4.NumberOfPoints):
                        self.assertEqual(param.GetValue(i1, i2, i3, i4), value)

        # SetValues function
        value = 1.0
        param.SetValues(value)
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    for i4 in range(d4.NumberOfPoints):
                        self.assertEqual(param.GetValue(i1, i2, i3, i4), value)

        # Numpy array
        noPoints = d1.NumberOfPoints * d2.NumberOfPoints * d3.NumberOfPoints * d4.NumberOfPoints
        ndValues = numpy.zeros((d1.NumberOfPoints, d2.NumberOfPoints, d3.NumberOfPoints, d4.NumberOfPoints))
        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    for i4 in range(d4.NumberOfPoints):
                        ndValues[i1][i2][i3][i4] = i1 * 1E3 + i2 * 1E2 + i3 * 1E1 + i4

        for i1 in range(d1.NumberOfPoints):
            for i2 in range(d2.NumberOfPoints):
                for i3 in range(d3.NumberOfPoints):
                    for i4 in range(d4.NumberOfPoints):
                        param.SetValue(i1, i2, i3, i4, float(ndValues[i1][i2][i3][i4]))

        #print ndValues
        #print param.npyValues

        self.assertEqual(len(param.Domains),        4)
        self.assertEqual(param.Units,               unit)
        self.assertEqual(param.NumberOfPoints,      noPoints)
        self.assertTrue(numpy.array_equal(param.npyValues, ndValues))

if __name__ == "__main__":
    unittest.main()