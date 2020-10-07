**************
Module pyCore
**************
..  
    Copyright (C) Dragan Nikolic
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

.. py:module:: pyCore

Key modelling concepts
======================
 
Classes
--------
.. autosummary::
    daeVariableType
    daeDomain
    daeParameter
    daeVariable
    daeModel
    daeEquation
    daeEquationExecutionInfo
    daeSTN
    daeIF
    daeState
    daePort
    daeEventPort
    daePortConnection
    daeScalarExternalFunction
    daeVectorExternalFunction
    daeDomainIndex
    daeIndexRange
    daeArrayRange
    daeDEDI
    daeAction
    daeOnConditionActions
    daeOnEventActions
    daeOptimizationVariable
    daeObjectiveFunction
    daeOptimizationConstraint
    daeMeasuredVariable

.. autoclass:: pyCore.daeVariableType
    :members:
    :undoc-members:

    .. automethod:: __init__
    
.. autoclass:: pyCore.daeObject
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeDomain
    :members:
    :undoc-members:

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: __call__

.. autoclass:: pyCore.daeParameter
    :members:
    :undoc-members:
    :exclude-members: GetValue, SetValue, GetQuantity, SetValues, __call__, array

    .. automethod:: __init__

    .. method:: GetValue((daeParameter)self, [(int)index1[, ...[, (int)index8]]]) -> float

        Gets the value of the parameter at the specified domain indexes. How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the parameter is distributed on.
    
    .. method:: GetQuantity((daeParameter)self, [(int)index1[, ...[, (int)index8]]]) -> quantity

        Gets the value of the parameter at the specified domain indexes as the ``quantity`` object (with value and units).
        How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the parameter is distributed on.

    .. method:: SetValue((daeParameter)self, [(int)index1[, ...[, (int)index8]]], (float)value) -> None

        Sets the value of the parameter at the specified domain indexes. How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the parameter is distributed on.

    .. method:: SetValue((daeParameter)self, [(int)index1[, ...[, (int)index8]]], (quantity)value) -> None

        Sets the value of the parameter at the specified domain indexes. How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the parameter is distributed on.

    .. method:: SetValues((daeParameter)self, (float)values) -> None

        Sets all values using a single float value.

    .. method:: SetValues((daeParameter)self, (quantity)values) -> None

        Sets all values using a single quantity object.

    .. method:: SetValues((daeParameter)self, (ndarray(dtype=float|quantity))values) -> None

        Sets all values using a numpy array of simple floats or quantity objects.

    .. method:: array((daeParameter)self, [(object)index1[, ...[, (object)index8]]]) -> adouble_array

        Gets the array of parameter's values at the specified domain indexes (used to build equation residuals only). How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the parameter is distributed on. Argument types can be
        one of the following:

        * :py:class:`~pyCore.daeIndexRange` object
        * plain integer (to select a single index from a domain)
        * python ``list`` (to select a list of indexes from a domain)
        * python ``slice`` (to select a range of indexes from a domain: start_index, end_index, step)
        * character ``'*'`` (to select all points from a domain)
        * empty python list ``[]`` (to select all points from a domain)

    .. method:: __call__((daeParameter)self, [(int)index1[, ...[, (int)index8]]]) -> adouble

        Gets the value of the parameter at the specified domain indexes (used to build equation residuals only). How many arguments ``index1``, ..., ``index8`` are used
        depends on the number of domains that the parameter is distributed on.

.. autoclass:: pyCore.daeVariable
    :members:
    :undoc-members:
    :exclude-members: GetValue, SetValue, GetQuantity, SetValues, AssignValue, ReAssignValue, SetInitialGuess, SetInitialCondition, ReSetInitialCondition,
                      AssignValues, ReAssignValues, SetInitialConditions, ReSetInitialConditions, SetInitialGuesses, SetAbsoluteTolerances,
                      __call__, d, d2, dt, array, d_array, d2_array, dt_array

    .. automethod:: __init__

    .. method:: GetValue((daeVariable)self, [(int)index1[, ...[, (int)index8]]]) -> float

        Gets the value of the variable at the specified domain indexes. How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the variable is distributed on.

    .. method:: GetQuantity((daeVariable)self, [(int)index1[, ...[, (int)index8]]]) -> quantity

        Gets the value of the variable at the specified domain indexes as the ``quantity`` object (with value and units).
        How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the variable is distributed on.

    .. method:: SetValue((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (float)value) -> None

        Sets the value of the variable at the specified domain indexes. How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the variable is distributed on.

    .. method:: SetValue((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (quantity)value) -> None

        Sets the value of the variable at the specified domain indexes. How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the variable is distributed on.

    .. method:: SetValues((daeVariable)self, (ndarray(dtype=float|quantity))values) -> None

        Sets all values using a numpy array of simple floats or quantity objects.

    .. method:: AssignValue((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (float)value) -> None

    .. method:: AssignValue((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (quantity)value) -> None

    .. method:: AssignValues((daeVariable)self, (float)values) -> None

    .. method:: AssignValues((daeVariable)self, (quantity)values) -> None

    .. method:: AssignValues((daeVariable)self, (ndarray(dtype=float|quantity))values) -> None

    .. method:: ReAssignValue((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (float)value) -> None

    .. method:: ReAssignValue((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (quantity)value) -> None

    .. method:: ReAssignValues((daeVariable)self, (float)values) -> None

    .. method:: ReAssignValues((daeVariable)self, (quantity)values) -> None

    .. method:: ReAssignValues((daeVariable)self, (ndarray(dtype=float|quantity))values) -> None

    .. method:: SetInitialCondition((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (float)initialCondition) -> None

    .. method:: SetInitialCondition((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (quantity)initialCondition) -> None

    .. method:: SetInitialConditions((daeVariable)self, (float)initialConditions) -> None

    .. method:: SetInitialConditions((daeVariable)self, (quantity)initialConditions) -> None

    .. method:: SetInitialConditions((daeVariable)self, (ndarray(dtype=float|quantity))initialConditions) -> None

    .. method:: ReSetInitialCondition((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (float)initialCondition) -> None

    .. method:: ReSetInitialCondition((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (quantity)initialCondition) -> None

    .. method:: ReSetInitialConditions((daeVariable)self, (float)initialConditions) -> None

    .. method:: ReSetInitialConditions((daeVariable)self, (quantity)initialConditions) -> None

    .. method:: ReSetInitialConditions((daeVariable)self, (ndarray(dtype=float|quantity))initialConditions) -> None

    .. method:: SetInitialGuess((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (float)initialGuess) -> None

    .. method:: SetInitialGuess((daeVariable)self, [(int)index1[, ...[, (int)index8]]], (quantity)initialGuess) -> None

    .. method:: SetInitialGuesses((daeVariable)self, (float)initialGuesses) -> None

    .. method:: SetInitialGuesses((daeVariable)self, (quantity)initialGuesses) -> None

    .. method:: SetInitialGuesses((daeVariable)self, (ndarray(dtype=float|quantity))initialGuesses) -> None

    .. method:: SetAbsoluteTolerances((daeVariable)self, (float)tolerances) -> None

    .. method:: array((daeVariable)self, [(object)index1[, ...[, (object)index8]]]) -> adouble_array

        Gets the array of variable's values at the specified domain indexes (used to build equation residuals only). How many arguments ``index1, ..., index8`` are used
        depends on the number of domains that the variable is distributed on. Argument types are the same
        as those described in :py:meth:`pyCore.daeParameter.array`

.. autoclass:: pyCore.daeModel
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeEquation
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeEquationExecutionInfo
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeSTN
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeIF
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeState
    :members:
    :undoc-members:

.. autoclass:: pyCore.daePort
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeEventPort
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daePortConnection
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeEventPortConnection
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeScalarExternalFunction
    :members:
    :undoc-members:

    .. automethod:: __init__
    .. automethod:: __call__

.. autoclass:: pyCore.daeVectorExternalFunction
    :members:
    :undoc-members:

    .. automethod:: __init__
    .. automethod:: __call__

.. autoclass:: pyCore.daeDomainIndex
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeIndexRange
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeArrayRange
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeDEDI
    :members:
    :undoc-members:

    .. automethod:: __init__
    .. automethod:: __call__

.. autoclass:: pyCore.daeAction
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeOnEventActions
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeOnConditionActions
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeOptimizationVariable
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeObjectiveFunction
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeOptimizationConstraint
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeMeasuredVariable
    :members:
    :undoc-members:

    .. automethod:: __init__

Functions
---------
.. autosummary::
    :nosignatures:

    dt
    d
    d2
    dt_array
    d_array
    d2_array
    Time
    Constant
    Array
    Sum
    Product
    Integral
    Average

.. autofunction:: pyCore.dt
.. autofunction:: pyCore.d
.. autofunction:: pyCore.d2
.. autofunction:: pyCore.dt_array
.. autofunction:: pyCore.d_array
.. autofunction:: pyCore.d2_array
.. autofunction:: pyCore.Time
.. autofunction:: pyCore.Constant
.. autofunction:: pyCore.Array
.. autofunction:: pyCore.Sum
.. autofunction:: pyCore.Product
.. autofunction:: pyCore.Integral
.. autofunction:: pyCore.Average

Auto-differentiation and equation evaluation tree support
=========================================================

Classes
-------

.. autosummary::
    adouble
    adouble_array
    daeCondition

.. autoclass:: pyCore.adouble
    :members:
    
    .. automethod:: __init__

.. autoclass:: pyCore.adouble_array
    :members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeCondition
    :members:

    Since it is not allowed to overload Python’s operators ``and``, ``or`` and ``not`` they cannot be used
    to define logical comparison operations; therefore, the bitwise operators ``&``, ``|`` and ``~`` are
    overloaded and should be used instead.

    .. method:: __or__((daeCondition)self, (daeCondition)right) -> daeCondition

    Bitwise operator or (``\``) in **DAE Tools** is used as a logical comparison operator ``or``.

    .. method:: __and__((daeCondition)self, (daeCondition)right) -> daeCondition

    Bitwise operator and (``&``) in **DAE Tools** is used as a logical comparison operator ``and``.

    .. method:: __inv__((daeCondition)self, (daeCondition)right) -> daeCondition

    Bitwise inversion operator (``~``) in **DAE Tools** is used as a logical comparison operator ``not``.


Mathematical functions
----------------------
.. autosummary::
    :nosignatures:

    Exp
    Log
    Log10
    Sqrt
    Pow
    Sin
    Cos
    Tan
    ASin
    ACos
    ATan
    Sinh
    Cosh
    Tanh
    ASinh
    ACosh
    ATanh
    ATan2
    Erf
    Ceil
    Floor
    Abs
    Min
    Max

.. autofunction:: pyCore.Exp
.. autofunction:: pyCore.Log
.. autofunction:: pyCore.Log10
.. autofunction:: pyCore.Sqrt
.. autofunction:: pyCore.Pow
.. autofunction:: pyCore.Sin
.. autofunction:: pyCore.Cos
.. autofunction:: pyCore.Tan
.. autofunction:: pyCore.ASin
.. autofunction:: pyCore.ACos
.. autofunction:: pyCore.ATan
.. autofunction:: pyCore.Sinh
.. autofunction:: pyCore.Cosh
.. autofunction:: pyCore.Tanh
.. autofunction:: pyCore.ASinh
.. autofunction:: pyCore.ACosh
.. autofunction:: pyCore.ATanh
.. autofunction:: pyCore.ATan2
.. autofunction:: pyCore.Erf
.. autofunction:: pyCore.Ceil
.. autofunction:: pyCore.Floor
.. autofunction:: pyCore.Abs
.. autofunction:: pyCore.Min
.. autofunction:: pyCore.Max


Auxiliary classes
=================
.. autosummary::
    :nosignatures:

    daeVariableWrapper
    daeConfig
    daeNodeGraph

.. autoclass:: pyCore.daeVariableWrapper
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeConfig
    :members:
    :undoc-members:

    .. automethod:: __contains__
    .. automethod:: __getitem__
    .. automethod:: __setitem__


.. autoclass:: daetools.pyDAE.eval_node_graph.daeNodeGraph
    :members:
    :undoc-members:
    
    .. automethod:: __init__
    .. automethod:: SaveGraph
    
Auxiliary functions
===================
.. autosummary::
    :nosignatures:
    
    daeVersion
    daeVersionMajor
    daeVersionMinor
    daeVersionBuild
    daeGetConfig
    daeSetConfigFile

.. autofunction:: pyCore.daeVersion
.. autofunction:: pyCore.daeVersionMajor
.. autofunction:: pyCore.daeVersionMinor
.. autofunction:: pyCore.daeVersionBuild
.. autofunction:: pyCore.daeGetConfig
.. autofunction:: pyCore.daeSetConfigFile

Enumerations
============
.. autosummary::
    daeeDomainType
    daeeParameterType
    daeePortType
    daeeDiscretizationMethod
    daeeDomainBounds
    daeeInitialConditionMode
    daeeDomainIndexType
    daeeRangeType
    daeIndexRangeType
    daeeOptimizationVariableType
    daeeModelLanguage
    daeeConstraintType
    daeeUnaryFunctions
    daeeBinaryFunctions
    daeeSpecialUnaryFunctions
    daeeLogicalUnaryOperator
    daeeLogicalBinaryOperator
    daeeConditionType
    daeeActionType
    daeeEquationType
    daeeModelType

.. autoclass:: pyCore.daeeDomainType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeParameterType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeePortType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeDiscretizationMethod
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeDomainBounds
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeInitialConditionMode
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeDomainIndexType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeRangeType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeIndexRangeType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeOptimizationVariableType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeModelLanguage
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeConstraintType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeUnaryFunctions
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeBinaryFunctions
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeSpecialUnaryFunctions
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeLogicalUnaryOperator
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeLogicalBinaryOperator
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeConditionType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeActionType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeEquationType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyCore.daeeModelType
    :members:
    :undoc-members:
    :exclude-members: names, values

Global constants
================
.. autosummary::
    :nosignatures:
    
    cnAlgebraic
    cnDifferential
    cnAssigned

.. autodata:: pyCore.cnAlgebraic

.. autodata:: pyCore.cnDifferential

.. autodata:: pyCore.cnAssigned

