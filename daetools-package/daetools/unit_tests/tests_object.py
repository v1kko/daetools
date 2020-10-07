#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, unittest
daetools_root = os.path.abspath('../../')
if not daetools_root in sys.path:
    sys.path.insert(0, daetools_root)
from daetools.pyDAE import *

def test_daeObjectInterface(ut, obj, expName, expCanonicalName, expDescription, expParentModel):
    ut.assertEqual(obj.Name,          expName)
    ut.assertEqual(obj.Description,   expDescription)
    ut.assertEqual(obj.CanonicalName, expCanonicalName)
    ut.assertEqual(obj.Model,         expParentModel)

class case_daeObject(unittest.TestCase):
    def test_RelativeNames(self):
        # We'll use daeDomain as a daeObject since we cannot instantiate daeObject directly
        model1 = daeModel('Model1')
        model2 = daeModel('Model2', model1)
        obj    = daeDomain('Domain', model2, pyUnits.unit())

        self.assertEqual(obj.GetNameRelativeToParentModel(), 'Domain')
        self.assertEqual(obj.GetStrippedNameRelativeToParentModel(), 'Domain')

        self.assertEqual(daeGetRelativeName(model2, obj), 'Domain')
        self.assertEqual(daeGetRelativeName(model1, obj), 'Model2.Domain')
        self.assertEqual(daeGetStrippedRelativeName(model1, obj), 'Model2.Domain')

    def test_ObjectNameValidity(self):
        # Valid names
        self.assertTrue(daeIsValidObjectName('a'))
        self.assertTrue(daeIsValidObjectName('a_1'))
        self.assertTrue(daeIsValidObjectName('&alpha;'))
        self.assertTrue(daeIsValidObjectName('&alpha;&beta;'))
        self.assertTrue(daeIsValidObjectName('b_&alpha;&beta;'))
        self.assertTrue(daeIsValidObjectName('b_&alpha;_&beta;'))
        self.assertTrue(daeIsValidObjectName('b_&alpha;_&beta;_'))
        self.assertTrue(daeIsValidObjectName('a_1(1,2)'))
        self.assertTrue(daeIsValidObjectName('a_1(a,b,3)'))

        # Invalid names
        self.assertFalse(daeIsValidObjectName(''))
        self.assertFalse(daeIsValidObjectName(' '))
        self.assertFalse(daeIsValidObjectName('_'))
        self.assertFalse(daeIsValidObjectName('1'))
        self.assertFalse(daeIsValidObjectName(';'))
        self.assertFalse(daeIsValidObjectName(' f'))
        self.assertFalse(daeIsValidObjectName('f '))
        self.assertFalse(daeIsValidObjectName('^ '))
        self.assertFalse(daeIsValidObjectName('/'))
        self.assertFalse(daeIsValidObjectName('_a'))
        self.assertFalse(daeIsValidObjectName('g_ '))
        self.assertFalse(daeIsValidObjectName('&;'))
        self.assertFalse(daeIsValidObjectName(';&'))
        self.assertFalse(daeIsValidObjectName(';gg&'))
        self.assertFalse(daeIsValidObjectName('a;&'))
        self.assertFalse(daeIsValidObjectName('a;&b'))
        self.assertFalse(daeIsValidObjectName('a;name&'))
        self.assertFalse(daeIsValidObjectName('&a;;&'))
        self.assertFalse(daeIsValidObjectName('a;;&'))
        self.assertFalse(daeIsValidObjectName('a&&;'))
        self.assertFalse(daeIsValidObjectName('a_1[a]'))
        self.assertFalse(daeIsValidObjectName('a_1(a, b, 3)'))

if __name__ == "__main__":
    unittest.main()