#!/usr/bin/env python

"""********************************************************************************
                         daeGetParserDictionary.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2011
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

import sys
from daetools.pyDAE import *

def addIdentifiers(model, parent, dictIdentifiers):
    for o in model.Parameters:
        relName = daeGetRelativeName(parent, o)
        print 'CanonicalName: {0}, RelName: {1}'.format(o.CanonicalName, relName)
        dictIdentifiers[relName] = o()

    for o in model.Variables:
        relName = daeGetRelativeName(parent, o)
        print 'CanonicalName: {0}, RelName: {1}'.format(o.CanonicalName, relName)
        dictIdentifiers[relName] = o()

    for port in model.Ports:
        for o in port.Parameters:
            relName = daeGetRelativeName(parent, o)
            print 'CanonicalName: {0}, RelName: {1}'.format(o.CanonicalName, relName)
            dictIdentifiers[relName] = o()
        for o in port.Variables:
            relName = daeGetRelativeName(parent, o)
            print 'CanonicalName: {0}, RelName: {1}'.format(o.CanonicalName, relName)
            dictIdentifiers[relName] = o()

    for m in model.Models:
        dictIdentifiers = addIdentifiers(m, model, dictIdentifiers)
        
    return dictIdentifiers

def getParserDictionaries(model):
    """
    Dictionary should contain the following type of items:
     - string : adouble (for parameters and variables)
     - string : function-of-one-argument (these has to be implemented: sin, cos, tan, exp, ln, log, sqrt)
    """
    dictIdentifiers = {}
    dictFunctions   = {}

    dictIdentifiers['time'] = model.time()

    # DAE Tools mathematical functions:
    dictFunctions['sin']  = Sin
    dictFunctions['cos']  = Cos
    dictFunctions['tan']  = Tan
    dictFunctions['log']  = Log10
    dictFunctions['ln']   = Log
    dictFunctions['sqrt'] = Sqrt
    dictFunctions['exp']  = Exp

    dictIdentifiers = addIdentifiers(model, model, dictIdentifiers)
    
    return dictIdentifiers, dictFunctions
