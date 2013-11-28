***********************
Module Finite Elements
***********************
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

Overview
==========


Base Classes
============
.. py:module:: pyCore
.. autosummary::
    :nosignatures:

    daeFiniteElementVariableInfo
    daeFiniteElementObjectInfo
    daeFiniteElementObject
    daeFiniteElementModel
    daeFiniteElementEquation

.. autoclass:: pyCore.daeFiniteElementVariableInfo
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeFiniteElementObjectInfo
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeFiniteElementObject
    :no-members:
    :no-undoc-members:

    .. method:: GetObjectInfo((daeFiniteElementObject)self) -> daeFiniteElementObjectInfo
    .. method:: NeedsReAssembling((dealiiFiniteElementSystem_1D)self) -> bool
    .. method:: AssembleSystem((dealiiFiniteElementSystem_1D)self) -> None
    .. method:: ReAssembleSystem((dealiiFiniteElementSystem_1D)self) -> None
    .. method:: CreateDataReporter((dealiiFiniteElementSystem_1D)self) -> dealIIDataReporter
    .. method:: RowIndices((dealiiFiniteElementSystem_1D)self, (int)row) -> list
    .. method:: Asystem((daeFiniteElementObject)self) -> daeMatrix
    .. method:: Msystem((daeFiniteElementObject)self) -> daeMatrix
    .. method:: Fload((daeFiniteElementObject)self) -> daeArray


.. autoclass:: pyCore.daeFiniteElementModel
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeFiniteElementEquation
    :members:
    :undoc-members:

.. py:module:: pyDataReporting
.. autoclass:: pyDataReporting.daeDataReporter_t
    :no-undoc-members:
    :no-members:
    :noindex:
   
deal.II
=======
.. py:module:: solvers.deal_II.pyDealII
.. py:currentmodule:: pyDealII

.. autosummary::
    :nosignatures:

    dealiiFiniteElementEquation_1D
    dealiiFiniteElementEquation_2D
    dealiiFiniteElementEquation_3D
    dealiiFiniteElementSystem_1D
    dealiiFiniteElementSystem_2D
    dealiiFiniteElementSystem_3D
    dealIIDataReporter

DataReporting Classes
---------------------
.. autoclass:: pyDealII.dealIIDataReporter
    :no-undoc-members:
    :no-members:

Finite Element Classes
----------------------
.. py:module:: solvers.deal_II.pyDealII
.. py:currentmodule:: pyDealII

.. autoclass:: pyDealII.dealiiFiniteElementEquation_1D
    :no-members:
    :no-undoc-members:

    .. automethod:: __init__
    .. autoattribute:: VariableDescription
    .. autoattribute:: VariableName
    .. autoattribute:: Multiplicity
    .. autoattribute:: Alocal
    .. autoattribute:: Flocal
    .. autoattribute:: FunctionsDirichletBC
    .. autoattribute:: FunctionsNeumannBC
    .. autoattribute:: Mlocal

    .. automethod:: ConvectionDiffusionEquation

                      
.. autoclass:: pyDealII.dealiiFiniteElementEquation_2D
    :no-members:
    :no-undoc-members:
                      
.. autoclass:: pyDealII.dealiiFiniteElementEquation_3D
    :no-members:
    :no-undoc-members:
    
.. autoclass:: pyDealII.dealiiFiniteElementSystem_1D
    :no-members:
    :no-undoc-members:

    .. method:: NeedsReAssembling((dealiiFiniteElementSystem_1D)self) -> bool
    .. method:: AssembleSystem((dealiiFiniteElementSystem_1D)self) -> None
    .. method:: ReAssembleSystem((dealiiFiniteElementSystem_1D)self) -> None
    .. method:: CreateDataReporter((dealiiFiniteElementSystem_1D)self) -> dealIIDataReporter
    .. method:: RowIndices((dealiiFiniteElementSystem_1D)self, (int)row) -> list

.. autoclass:: pyDealII.dealiiFiniteElementSystem_2D
    :no-members:
    :no-undoc-members:

    .. method:: NeedsReAssembling((dealiiFiniteElementSystem_2D)self) -> bool
    .. method:: AssembleSystem((dealiiFiniteElementSystem_2D)self) -> None
    .. method:: ReAssembleSystem((dealiiFiniteElementSystem_2D)self) -> None
    .. method:: CreateDataReporter((dealiiFiniteElementSystem_2D)self) -> dealIIDataReporter
    .. method:: RowIndices((dealiiFiniteElementSystem_2D)self, (int)row) -> list

.. autoclass:: pyDealII.dealiiFiniteElementSystem_3D
    :no-members:
    :no-undoc-members:

    .. method:: NeedsReAssembling((dealiiFiniteElementSystem_3D)self) -> bool
    .. method:: AssembleSystem((dealiiFiniteElementSystem_3D)self) -> None
    .. method:: ReAssembleSystem((dealiiFiniteElementSystem_3D)self) -> None
    .. method:: CreateDataReporter((dealiiFiniteElementSystem_3D)self) -> dealIIDataReporter
    .. method:: RowIndices((dealiiFiniteElementSystem_3D)self, (int)row) -> list

Auxiliary Classes
-----------------
.. autosummary::
    :nosignatures:

    Tensor_1_1D
    Tensor_1_2D
    Tensor_1_3D
    Point_1D
    Point_2D
    Point_3D
    Tensor_2_1D
    Tensor_2_2D
    Tensor_2_3D
    Function_1D
    Function_2D
    Function_3D
    ConstantFunction_1D
    ConstantFunction_2D
    ConstantFunction_3D
    ZeroFunction_1D
    ZeroFunction_2D
    ZeroFunction_3D
    Vector
    FullMatrix
    SparseMatrix
    Quadrature_0D
    Quadrature_1D
    Quadrature_2D
    Quadrature_3D
    QGauss_0D
    QGauss_1D
    QGauss_2D
    QGauss_3D
    QGaussLobatto_0D
    QGaussLobatto_1D
    QGaussLobatto_2D
    QGaussLobatto_3D
    feCellContext_1D
    feCellContext_2D
    feCellContext_3D
    feExpression_1D
    feExpression_2D
    feExpression_3D
    
.. autoclass:: pyDealII.Tensor_1_1D

.. autoclass:: pyDealII.Tensor_1_2D

.. autoclass:: pyDealII.Tensor_1_3D

.. autoclass:: pyDealII.Point_1D

.. autoclass:: pyDealII.Point_2D

.. autoclass:: pyDealII.Point_3D

.. autoclass:: pyDealII.Tensor_2_1D

.. autoclass:: pyDealII.Tensor_2_2D

.. autoclass:: pyDealII.Tensor_2_3D

.. autoclass:: pyDealII.Function_1D

.. autoclass:: pyDealII.Function_2D

.. autoclass:: pyDealII.Function_3D

.. autoclass:: pyDealII.ConstantFunction_1D

.. autoclass:: pyDealII.ConstantFunction_2D

.. autoclass:: pyDealII.ConstantFunction_3D

.. autoclass:: pyDealII.ZeroFunction_1D

.. autoclass:: pyDealII.ZeroFunction_2D

.. autoclass:: pyDealII.ZeroFunction_3D

.. autoclass:: pyDealII.Vector

.. autoclass:: pyDealII.FullMatrix

.. autoclass:: pyDealII.SparseMatrix

.. autoclass:: pyDealII.Quadrature_0D

.. autoclass:: pyDealII.Quadrature_1D

.. autoclass:: pyDealII.Quadrature_2D

.. autoclass:: pyDealII.Quadrature_3D

.. autoclass:: pyDealII.QGauss_0D

.. autoclass:: pyDealII.QGauss_1D

.. autoclass:: pyDealII.QGauss_2D

.. autoclass:: pyDealII.QGauss_3D

.. autoclass:: pyDealII.QGaussLobatto_0D

.. autoclass:: pyDealII.QGaussLobatto_1D

.. autoclass:: pyDealII.QGaussLobatto_2D

.. autoclass:: pyDealII.QGaussLobatto_3D

.. autoclass:: pyDealII.feCellContext_1D

.. autoclass:: pyDealII.feCellContext_2D

.. autoclass:: pyDealII.feCellContext_3D

.. autoclass:: pyDealII.feExpression_1D

.. autoclass:: pyDealII.feExpression_2D

.. autoclass:: pyDealII.feExpression_3D

Auxiliary Functions
-------------------
.. autosummary::
    :nosignatures:

    constant_1D
    constant_2D
    constant_3D
    phi_1D
    phi_2D
    phi_3D
    dphi_1D
    dphi_2D
    dphi_3D
    d2phi_1D
    d2phi_2D
    d2phi_3D
    JxW_1D
    JxW_2D
    JxW_3D
    xyz_1D
    xyz_2D
    xyz_3D
    normal_1D
    normal_2D
    normal_3D
    function_value_1D
    function_value_2D
    function_value_3D
    function_gradient_1D
    function_gradient_2D
    function_gradient_3D

.. autofunction:: constant_1D
.. autofunction:: constant_2D
.. autofunction:: constant_3D
.. autofunction:: phi_1D
.. autofunction:: phi_2D
.. autofunction:: phi_3D
.. autofunction:: dphi_1D
.. autofunction:: dphi_2D
.. autofunction:: dphi_3D
.. autofunction:: d2phi_1D
.. autofunction:: d2phi_2D
.. autofunction:: d2phi_3D
.. autofunction:: JxW_1D
.. autofunction:: JxW_2D
.. autofunction:: JxW_3D
.. autofunction:: xyz_1D
.. autofunction:: xyz_2D
.. autofunction:: xyz_3D
.. autofunction:: normal_1D
.. autofunction:: normal_2D
.. autofunction:: normal_3D
.. autofunction:: function_value_1D
.. autofunction:: function_value_2D
.. autofunction:: function_value_3D
.. autofunction:: function_gradient_1D
.. autofunction:: function_gradient_2D
.. autofunction:: function_gradient_3D

    
Enumerations and Constants
--------------------------
.. autosummary::
    :nosignatures:

    fe_i
    fe_j
    fe_q
    dealiiFluxType

.. autoclass:: pyDealII.dealiiFluxType
    :members:
    :undoc-members:
    :exclude-members: names, values

.. data:: pyDealII.fe_i

.. data:: pyDealII.fe_j

.. data:: pyDealII.fe_q
