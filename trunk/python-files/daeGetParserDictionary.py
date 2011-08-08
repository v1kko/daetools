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

def getParserDictionary(model):
    """
    Dictionary should contain the following type of items:
     - string : adouble (for parameters and variables)
     - string : function-of-one-argument (these has to be implemented: sin, cos, tan, exp, ln, log, sqrt)
    """
    dictNameValue = {}

    # adouble values for parameters
    for p in model.Parameters:
        dictNameValue[p.Name] = p()

    # adouble values for variables
    for v in model.Variables:
        dictNameValue[v.Name] = v()

    # DAE Tools mathematical functions:
    dictNameValue['sin']  = Sin
    dictNameValue['cos']  = Cos
    dictNameValue['tan']  = Tan
    dictNameValue['log']  = Log10
    dictNameValue['ln']   = Log
    dictNameValue['sqrt'] = Sqrt
    dictNameValue['exp']  = Exp

    return dictNameValue
