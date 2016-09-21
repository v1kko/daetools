"""********************************************************************************
                             deal_ii.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2016
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import pyCore
import pyDataReporting
import pyDealII
from pyDealII import *

def setFEInitialConditions(FEmodel, FEsystem, dofName, ic):
    """
    Arguments:
        FEmodel is an instance of daeFiniteElementModel
        FEsystem is an instance of daeFiniteElementObject (i.e. dealiiFiniteElementSystem_xD)
        dofName is a string with the dof name
        ic can be float value or a callable that returns float value for the given index in variable
    """
    variable = None
    indexMap = {}
    for var in FEmodel.Variables:
        if var.Name == dofName:
            variable = var
            for i in range(var.NumberOfPoints):
                index = var.OverallIndex+i
                indexMap[index] = i

            break

    if not variable:
        raise RuntimeError('DOF %s not found' % dofName)

    #print(indexMap)

    Mij = FEsystem.Msystem()
    for row in range(Mij.n):
        # Iterate over columns and set initial conditions.
        # If an item in the mass matrix is zero skip it.
        for column in FEsystem.RowIndices(row):
            if Mij(row, column).Node or Mij(row, column).Value != 0:
                oi_column = variable.OverallIndex+column
                if oi_column in indexMap:
                    internal_index = indexMap[oi_column]
                    if callable(ic):
                        ic_val = float(ic(internal_index))
                    else:
                        ic_val = float(ic)
                    variable.SetInitialCondition(internal_index, ic_val)
