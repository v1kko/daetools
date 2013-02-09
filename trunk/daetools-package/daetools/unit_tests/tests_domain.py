#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, numpy, unittest
sys.path.insert(0, os.path.abspath('../../'))
from daetools.pyDAE import *
import tests_object

"""
Tests for:
 - daeDomain (array, distributed domain, operators () and [])
 - adDomainIndexNode
 - daeIndexRange
"""

class case_daeDomain(unittest.TestCase):
    def setUp(self):
        self._printExceptions = True

    def _createDomain(self):
        name      = 'Domain'
        descr     = 'Domain description'
        canonName = 'Model.Domain'
        unit      = (pyUnits.m**2)*(pyUnits.s**-2) / (pyUnits.kg**1)
        parent    = daeModel('Model', None, '')
        domain    = daeDomain(name, parent, unit, descr)

        return domain, name, canonName, descr, parent, unit

    def test_daeObject(self):
        d, name, canonName, descr, parent, unit = self._createDomain()
        tests_object.test_daeObjectInterface(self, d, name, canonName, descr, parent)

    def test_Distributed(self):
        d, name, canonName, descr, parent, unit = self._createDomain()
        discrMethod = eCFDM
        order       = 2
        noIntervals = 10
        noPoints    = noIntervals + 1
        points      = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ndPoints    = numpy.array(points)
        lb          = 0.0
        ub          = 10.0

        # Not initialized yet!
        with self.assertRaises(RuntimeError) as context:
            x = d[0]
        
        with self.assertRaises(RuntimeError) as context:
            x = d(0)
            
        # Discr. method must be eCFDM!
        with self.assertRaises(RuntimeError) as context:
            d.CreateDistributed(eFFDM, order, noIntervals, lb,  ub)

        with self.assertRaises(RuntimeError) as context:
            d.CreateDistributed(eBFDM, order, noIntervals, lb,  ub)

        # Discr. order must be 2!
        with self.assertRaises(RuntimeError) as context:
            d.CreateDistributed(discrMethod, 3, noIntervals, lb,  ub)

        # Lower bound must be less than the upper bound!
        with self.assertRaises(RuntimeError) as context:
            d.CreateDistributed(discrMethod, order, noIntervals, 1.0,  1.0)

        # The number of intervals must be greater than or equal 2!
        with self.assertRaises(RuntimeError) as context:
            d.CreateDistributed(discrMethod, order, 1, lb,  ub)

        # Initialize domain
        d.CreateDistributed(discrMethod, order, noIntervals, lb,  ub)

        # Out of bounds index!
        with self.assertRaises(RuntimeError) as context:
            x = d[100]

        # Setting the point value directly (not through Points attribute) must be illegal!
        with self.assertRaises(TypeError) as context:
            d[0] = 0.0

        self.assertEqual(d.Type,                 eDistributed)
        self.assertEqual(d.DiscretizationMethod, discrMethod)
        self.assertEqual(d.DiscretizationOrder,  order)
        self.assertEqual(d.NumberOfIntervals,    noIntervals)
        self.assertEqual(d.NumberOfPoints,       noPoints)
        self.assertEqual(d.LowerBound,           lb)
        self.assertEqual(d.UpperBound,           ub)
        self.assertEqual(d.Units,                unit)
        self.assertEqual(d.Points,               points)
        self.assertTrue(numpy.array_equal(d.npyPoints, ndPoints))

        index = 3
        value = points[index]

        ad   = d[index] # adouble
        node = ad.Node  # adDomainIndexNode

        self.assertEqual(ad.Value,      0.0)
        self.assertEqual(ad.Derivative, 0.0)

        self.assertEqual(node.Index,     index)
        self.assertEqual(node.Domain.ID, d.ID)
        self.assertEqual(node.Value,     value)

        # Change domain points and check
        new_value = 1000.0
        points[index] = new_value

        d.Points = points

        # Domain's points must match the updated list of points
        self.assertEqual(d.Points, points)

        # Node's value points to the list of domain's points
        # Thus, the value must have been updated to the new_value
        self.assertEqual(node.Value, new_value)

        ir_list = [0, 5, 9]
        ir_slice = slice(1, 10, 2)
        r = range(1, 10, 2)
        
        ir1 = daeIndexRange(d) 
        ir2 = daeIndexRange(d, ir_list)
        ir3 = daeIndexRange(d, ir_slice.start, ir_slice.stop, ir_slice.step)

        self.assertEqual(ir1.Domain.ID,  d.ID)
        self.assertEqual(ir1.NoPoints,   d.NumberOfPoints)
        self.assertEqual(ir1.Type,       eAllPointsInDomain)
        self.assertEqual(ir1.StartIndex, 0)
        self.assertEqual(ir1.EndIndex,   0)
        self.assertEqual(ir1.Step,       0)

        self.assertEqual(ir2.Domain.ID,  d.ID)
        self.assertEqual(ir2.NoPoints,   len(ir_list))
        self.assertEqual(ir2.Type,       eCustomRange)
        self.assertEqual(ir2.StartIndex, 0)
        self.assertEqual(ir2.EndIndex,   0)
        self.assertEqual(ir2.Step,       0)

        self.assertEqual(ir3.Domain.ID,  d.ID)
        self.assertEqual(ir3.NoPoints,   len(r))
        self.assertEqual(ir3.Type,       eRangeOfIndexes)
        self.assertEqual(ir3.StartIndex, ir_slice.start)
        self.assertEqual(ir3.EndIndex,   ir_slice.stop)
        self.assertEqual(ir3.Step,       ir_slice.step)
        
    def test_Array(self):
        d, name, canonName, descr, parent, unit = self._createDomain()
        discrMethod = eCFDM
        order       = 2
        noIntervals = 10
        noPoints    = noIntervals
        points      = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ndPoints    = numpy.array(points)
        lb          = 1.0
        ub          = 10.0

        # Not initialized yet!
        with self.assertRaises(RuntimeError) as context:
            x = d[0]

        with self.assertRaises(RuntimeError) as context:
            x = d(0)

        # The number of intervals must not be 0!
        with self.assertRaises(RuntimeError) as context:
            d.CreateArray(0)

        # Initialize domain
        d.CreateArray(noIntervals)

        # Out of bounds index!
        with self.assertRaises(RuntimeError) as context:
            x = d[100]

        # Setting the point value directly (not through Points attribute) must be illegal!
        with self.assertRaises(TypeError) as context:
            d[0] = 0.0

        self.assertEqual(d.Type,                 eArray)
        self.assertEqual(d.DiscretizationMethod, eDMUnknown)
        self.assertEqual(d.DiscretizationOrder,  0)
        self.assertEqual(d.NumberOfIntervals,    noIntervals)
        self.assertEqual(d.NumberOfPoints,       noPoints)
        self.assertEqual(d.LowerBound,           lb)
        self.assertEqual(d.UpperBound,           ub)
        self.assertEqual(d.Units,                unit)
        self.assertEqual(d.Points,               points)
        self.assertTrue(numpy.array_equal(d.npyPoints, ndPoints))

        index = 3
        value = points[index]

        ad   = d[index] # adouble
        node = ad.Node  # adDomainIndexNode

        self.assertEqual(ad.Value,      0.0)
        self.assertEqual(ad.Derivative, 0.0)

        self.assertEqual(node.Index,     index)
        self.assertEqual(node.Domain.ID, d.ID)
        self.assertEqual(node.Value,     value)

        # Change domain points and check
        new_value = 1000.0
        points[index] = new_value

        # This is forbidden for arrays!
        with self.assertRaises(RuntimeError) as context:
            d.Points = points

        # Node's value points to the list of domain's points
        # Here the previous call did not change the list of points
        # Thus, the value is NOT updated
        self.assertNotEqual(node.Value, new_value)

        ir_list = [0, 5, 9]
        ir_slice = slice(1, 10, 2)
        r = range(1, 10, 2)

        ir1 = daeIndexRange(d)
        ir2 = daeIndexRange(d, ir_list)
        ir3 = daeIndexRange(d, ir_slice.start, ir_slice.stop, ir_slice.step)

        self.assertEqual(ir1.Domain.ID,  d.ID)
        self.assertEqual(ir1.NoPoints,   d.NumberOfPoints)
        self.assertEqual(ir1.Type,       eAllPointsInDomain)
        self.assertEqual(ir1.StartIndex, 0)
        self.assertEqual(ir1.EndIndex,   0)
        self.assertEqual(ir1.Step,       0)

        self.assertEqual(ir2.Domain.ID,  d.ID)
        self.assertEqual(ir2.NoPoints,   len(ir_list))
        self.assertEqual(ir2.Type,       eCustomRange)
        self.assertEqual(ir2.StartIndex, 0)
        self.assertEqual(ir2.EndIndex,   0)
        self.assertEqual(ir2.Step,       0)

        self.assertEqual(ir3.Domain.ID,  d.ID)
        self.assertEqual(ir3.NoPoints,   len(r))
        self.assertEqual(ir3.Type,       eRangeOfIndexes)
        self.assertEqual(ir3.StartIndex, ir_slice.start)
        self.assertEqual(ir3.EndIndex,   ir_slice.stop)
        self.assertEqual(ir3.Step,       ir_slice.step)

if __name__ == "__main__":
    unittest.main()