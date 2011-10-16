#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, inspect
from daetools.pyDAE import *
import daetools.pyDAE.pyUnits
from daetools.pyDAE.pyUnits import m, mm, km, kg, N, kJ, MPa, cSt
from time import localtime, strftime

def printQuantityInfo(name, q):
    print('{0} = {1}; SI value: {2} * {3} = {4}'.format(name, q, q.value, q.units.baseUnit, q.valueInSIUnits))
    
if __name__ == "__main__":
    u = unit({'kg':2, 'm':-3})
    print u

    u = mm
    print u.baseUnit
    print u*u
    print kg ** 2.3
    print kg * m**2 / 15 

    # Define some 'quantity' objects (they have 'value' and 'units')
    y   = 15   * mm
    x1  = 1.0  * m
    x2  = 0.2  * m
    x3  = 15   * N
    x4  = 1.25 * kJ
    
    printQuantityInfo('y', y)
    printQuantityInfo('x1', x1)
    printQuantityInfo('x2', x2)
    printQuantityInfo('x3', x3)
    printQuantityInfo('x4', x4)

    print('x1({0}) == x2({1}) ({2})'.format(x1, x2, x1 == x2))
    print('x1({0}) != x2({1}) ({2})'.format(x1, x2, x1 != x2))
    print('x1({0}) > x2({1}) ({2})' .format(x1, x2, x1 >  x2))
    print('x1({0}) >= x2({1}) ({2})'.format(x1, x2, x1 >= x2))
    print('x1({0}) < x2({1}) ({2})' .format(x1, x2, x1 <  x2))
    print('x1({0}) <= x2({1}) ({2})'.format(x1, x2, x1 <= x2))
    
    # quantity in [m]
    z = 1 * m
    print(z)
    z.value = 12.4 * mm # set a new value given in [mm]
    print(z)
    z.value = 0.32 * km # set a new value given in [km]
    print(z)
    z.value = 1 # set a new value in units in the quantity object, here in [m]
    print(z)
    
    print 1*MPa / (2*(cSt**2))
    
    try:
        result = (1*kg) < (2.3*m)
    except Exception as e:
        print str(e)
