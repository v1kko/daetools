"""********************************************************************************
                             deal_ii.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
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
      - FEmodel is an instance of daeFiniteElementModel
      - FEsystem is an instance of daeFiniteElementObject (i.e. dealiiFiniteElementSystem_xD)
      - dofName is a string with the dof name
      - ic can be a float value or a callable that returns a float value for the given arguments:
        index_in_the_domain and overall_index

    Nota bene:
        There can be other variables in the model.
        Therefore, the overall index of the first variable does not start at 0.
        For instance, there is one variable apart from the FE system and the system has N=100 DOFs:
          - FE system starts from Nstart=1 and internally in the range [0,N): Mij(N,N)
          - The overall index is Nstart+column where the column is a position of a non-zero item in Mij.

    """
    variable = None
    dict_oi_varIndex = {}
    fe_start_index = 0
    if len(FEmodel.Variables) > 0:
        var = FEmodel.Variables[0]
        fe_start_index = var.OverallIndex
        #print('fe_start_index = %d' % fe_start_index)

    for var in FEmodel.Variables:
        if var.Name == dofName:
            variable = var
            for i in range(var.NumberOfPoints):
                oi_index = var.OverallIndex+i
                dict_oi_varIndex[oi_index] = i
            break
    #print(dict_oi_varIndex)
    
    if not variable:
        raise RuntimeError('DOF %s not found' % dofName)

    # ic can be a callable or a single float value.
    # If it is a float value wrap it into a lambda function.
    if callable(ic):
        fun_ic = ic
    else:
        fun_ic = lambda varIndex, overallIndex: ic

    ic_already_set = {}
    Mij = FEsystem.Msystem()
    for row in range(Mij.n):
        # Iterate over columns and set initial conditions.
        # If an item in the mass matrix is zero skip it.
        for column in FEsystem.RowIndices(row):
            if Mij(row, column).Node or Mij(row, column).Value != 0:
                oi_column = fe_start_index + column
                #print(oi_column)
                # Proceed if the column belongs to the selected variable
                if oi_column in dict_oi_varIndex:
                    # If the IC has already been set skip the column
                    if oi_column in ic_already_set:
                        continue
                    # Internal index is the internal variable index
                    internal_index = dict_oi_varIndex[oi_column]
                    ic_val = float(fun_ic(internal_index, oi_column))
                    variable.SetInitialCondition(internal_index, ic_val)
                    ic_already_set[oi_column] = ic_val # Not used, can be any value
