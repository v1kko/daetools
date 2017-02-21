***************
Finite Elements
***************
..
    Copyright (C) Dragan Nikolic, 2016
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.


Base Classes
============
.. py:module:: pyCore
.. autosummary::
    :nosignatures:

    daeFiniteElementModel
    daeFiniteElementEquation
    daeFiniteElementVariableInfo
    daeFiniteElementObjectInfo
    daeFiniteElementObject_t

.. autoclass:: pyCore.daeFiniteElementModel
    :no-members:
    :no-undoc-members:

    .. automethod:: __init__

.. autoclass:: pyCore.daeFiniteElementModel
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeFiniteElementEquation
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeFiniteElementVariableInfo
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeFiniteElementObjectInfo
    :members:
    :undoc-members:

.. autoclass:: pyCore.daeFiniteElementObject_t
    :members:
    :undoc-members:


deal.II Main FE Classes
=======================
.. py:module:: solvers.deal_II.pyDealII
.. py:currentmodule:: pyDealII

These classes are instantiated by the users and used to specify:

- Information about Degrees of Freedom (DOFs) including the Finite Element space for each DOF (dealiiFiniteElementDOF_nD class)
- Weak form expressions for cells and faces (including boundaries) (dealiiFiniteElementWeakForm_nD class)
- Information about the mesh, quadrature rules, DOFs and weak forms (dealiiFiniteElementSystem_1D class)
- Data reporter that stores the results in .vtk format in the specified directory (dealIIDataReporter class)

Since it is not possible to use deal.II template classes in Python, separate classes are provided for three spatial dimensions.


.. autosummary::
    :nosignatures:

    dealiiFiniteElementDOF_1D
    dealiiFiniteElementDOF_2D
    dealiiFiniteElementDOF_3D
    dealiiFiniteElementWeakForm_1D
    dealiiFiniteElementWeakForm_2D
    dealiiFiniteElementWeakForm_3D
    dealiiFiniteElementSystem_1D
    dealiiFiniteElementSystem_2D
    dealiiFiniteElementSystem_3D
    dealIIDataReporter

.. autoclass:: pyDealII.dealiiFiniteElementDOF_1D
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementDOF_2D
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementDOF_3D
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementWeakForm_1D
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementWeakForm_2D
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementWeakForm_3D
    :members:
    :undoc-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementSystem_1D
    :no-undoc-members:
    :no-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementSystem_2D
    :no-undoc-members:
    :no-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealiiFiniteElementSystem_3D
    :no-undoc-members:
    :no-members:

    .. automethod:: __init__

.. autoclass:: pyDealII.dealIIDataReporter
    :no-undoc-members:
    :no-members:

deal.II Finite Elements
=======================
.. autosummary::
    :nosignatures:

    FiniteElement_1D
    FiniteElement_2D
    FiniteElement_3D
    FE_Q_1D
    FE_Q_2D
    FE_Q_3D
    FE_Bernstein_1D
    FE_Bernstein_2D
    FE_Bernstein_3D
    FE_RaviartThomas_1D
    FE_RaviartThomas_2D
    FE_RaviartThomas_3D
    FE_DGRaviartThomas_1D
    FE_DGRaviartThomas_2D
    FE_DGRaviartThomas_3D
    FE_Nedelec_1D
    FE_Nedelec_2D
    FE_Nedelec_3D
    FE_DGNedelec_1D
    FE_DGNedelec_2D
    FE_DGNedelec_3D
    FE_BDM_1D
    FE_BDM_2D
    FE_BDM_3D
    FE_DGBDM_1D
    FE_DGBDM_2D
    FE_DGBDM_3D
    FE_ABF_1D
    FE_ABF_2D
    FE_ABF_3D
    FE_DGQ_1D
    FE_DGQ_2D
    FE_DGQ_3D
    FE_DGP_1D
    FE_DGP_2D
    FE_DGP_3D

.. autoclass:: FiniteElement_1D

.. autoclass:: FiniteElement_2D

.. autoclass:: FiniteElement_3D

.. autoclass:: FE_Q_1D

    .. automethod:: __init__

.. autoclass:: FE_Q_2D

    .. automethod:: __init__

.. autoclass:: FE_Q_3D

    .. automethod:: __init__

.. autoclass:: FE_Bernstein_1D

    .. automethod:: __init__

.. autoclass:: FE_Bernstein_2D

    .. automethod:: __init__

.. autoclass:: FE_Bernstein_3D

    .. automethod:: __init__

.. autoclass:: FE_RaviartThomas_1D

    .. automethod:: __init__

.. autoclass:: FE_RaviartThomas_2D

    .. automethod:: __init__

.. autoclass:: FE_RaviartThomas_3D

    .. automethod:: __init__

.. autoclass:: FE_DGRaviartThomas_1D

    .. automethod:: __init__

.. autoclass:: FE_DGRaviartThomas_2D

    .. automethod:: __init__

.. autoclass:: FE_DGRaviartThomas_3D

    .. automethod:: __init__

.. autoclass:: FE_Nedelec_1D

    .. automethod:: __init__

.. autoclass:: FE_Nedelec_2D

    .. automethod:: __init__

.. autoclass:: FE_Nedelec_3D

    .. automethod:: __init__

.. autoclass:: FE_DGNedelec_1D

    .. automethod:: __init__

.. autoclass:: FE_DGNedelec_2D

    .. automethod:: __init__

.. autoclass:: FE_DGNedelec_3D

    .. automethod:: __init__

.. autoclass:: FE_BDM_1D

    .. automethod:: __init__

.. autoclass:: FE_BDM_2D

    .. automethod:: __init__

.. autoclass:: FE_BDM_3D

    .. automethod:: __init__

.. autoclass:: FE_DGBDM_1D

    .. automethod:: __init__

.. autoclass:: FE_DGBDM_2D

    .. automethod:: __init__

.. autoclass:: FE_DGBDM_3D

    .. automethod:: __init__

.. autoclass:: FE_ABF_1D

    .. automethod:: __init__

.. autoclass:: FE_ABF_2D

    .. automethod:: __init__

.. autoclass:: FE_ABF_3D

    .. automethod:: __init__

.. autoclass:: FE_DGQ_1D

    .. automethod:: __init__

.. autoclass:: FE_DGQ_2D

    .. automethod:: __init__

.. autoclass:: FE_DGQ_3D

    .. automethod:: __init__

.. autoclass:: FE_DGP_1D

    .. automethod:: __init__

.. autoclass:: FE_DGP_2D

    .. automethod:: __init__

.. autoclass:: FE_DGP_3D

    .. automethod:: __init__


deal.II Quadrature Rules
========================
.. autosummary::
    :nosignatures:

    Quadrature_0D
    Quadrature_1D
    Quadrature_2D
    Quadrature_3D
    QGauss_1D
    QGauss_2D
    QGauss_3D
    QGaussLobatto_1D
    QGaussLobatto_2D
    QGaussLobatto_3D
    QMidpoint_1D
    QMidpoint_2D
    QMidpoint_3D
    QSimpson_1D
    QSimpson_2D
    QSimpson_3D
    QTrapez_1D
    QTrapez_2D
    QTrapez_3D
    QMilne_1D
    QMilne_2D
    QMilne_3D
    QWeddle_1D
    QWeddle_2D
    QWeddle_3D
    QGaussLog_1D
    QGaussLogR_1D
    QGaussOneOverR_2D
    QGaussChebyshev_1D
    QGaussChebyshev_2D
    QGaussChebyshev_3D
    QGaussLobattoChebyshev_1D
    QGaussLobattoChebyshev_2D
    QGaussLobattoChebyshev_3D

.. autoclass:: pyDealII.Quadrature_0D

.. autoclass:: pyDealII.Quadrature_1D

.. autoclass:: pyDealII.Quadrature_2D

.. autoclass:: pyDealII.Quadrature_3D

.. autoclass:: pyDealII.QGauss_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGauss_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGauss_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLobatto_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLobatto_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLobatto_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QMidpoint_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QMidpoint_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QMidpoint_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QSimpson_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QSimpson_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QSimpson_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QTrapez_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QTrapez_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QTrapez_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QMilne_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QMilne_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QMilne_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QWeddle_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QWeddle_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QWeddle_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLog_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLogR_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussOneOverR_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussChebyshev_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussChebyshev_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussChebyshev_3D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLobattoChebyshev_1D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLobattoChebyshev_2D

    .. automethod:: __init__

.. autoclass:: pyDealII.QGaussLobattoChebyshev_3D

    .. automethod:: __init__


Functions for Specification of Weak Forms
=========================================
.. autosummary::
    :nosignatures:

    feExpression_1D
    feExpression_2D
    feExpression_3D
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
    function_adouble_value_1D
    function_adouble_value_1D
    function_adouble_value_1D
    function_adouble_gradient_1D
    function_adouble_gradient_1D
    function_adouble_gradient_1D
    tensor1_function_value_1D
    tensor1_function_value_1D
    tensor1_function_value_1D
    tensor2_function_value_1D
    tensor2_function_value_1D
    tensor2_function_value_1D
    tensor1_function_gradient_1D
    tensor1_function_gradient_1D
    tensor1_function_gradient_1D
    tensor2_function_gradient_1D
    tensor2_function_gradient_1D
    tensor2_function_gradient_1D
    phi_vector_1D
    phi_vector_2D
    phi_vector_3D
    dphi_vector_1D
    dphi_vector_2D
    dphi_vector_3D
    d2phi_vector_1D
    d2phi_vector_2D
    d2phi_vector_3D
    div_phi_1D
    div_phi_2D
    div_phi_3D
    dof_1D
    dof_2D
    dof_3D
    dof_approximation_1D
    dof_approximation_2D
    dof_approximation_3D
    dof_gradient_approximation_1D
    dof_gradient_approximation_2D
    dof_gradient_approximation_3D
    dof_hessian_approximation_1D
    dof_hessian_approximation_2D
    dof_hessian_approximation_3D
    vector_dof_approximation_1D
    vector_dof_approximation_2D
    vector_dof_approximation_3D
    vector_dof_gradient_approximation_1D
    vector_dof_gradient_approximation_2D
    vector_dof_gradient_approximation_3D
    adouble_1D
    adouble_2D
    adouble_3D
    tensor1_1D
    tensor1_2D
    tensor1_3D
    tensor2_1D
    tensor2_2D
    tensor2_3D
    tensor3_1D
    tensor3_2D
    tensor3_3D
    adouble_tensor1_1D
    adouble_tensor1_2D
    adouble_tensor1_3D
    adouble_tensor2_1D
    adouble_tensor2_2D
    adouble_tensor2_3D
    adouble_tensor3_1D
    adouble_tensor3_2D
    adouble_tensor3_3D

.. autoclass:: pyDealII.feExpression_1D

.. autoclass:: pyDealII.feExpression_2D

.. autoclass:: pyDealII.feExpression_3D

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

.. autofunction:: function_adouble_value_1D

.. autofunction:: function_adouble_value_1D

.. autofunction:: function_adouble_value_1D

.. autofunction:: function_adouble_gradient_1D

.. autofunction:: function_adouble_gradient_1D

.. autofunction:: function_adouble_gradient_1D

.. autofunction:: tensor1_function_value_1D

.. autofunction:: tensor1_function_value_1D

.. autofunction:: tensor1_function_value_1D

.. autofunction:: tensor2_function_value_1D

.. autofunction:: tensor2_function_value_1D

.. autofunction:: tensor2_function_value_1D

.. autofunction:: tensor1_function_gradient_1D

.. autofunction:: tensor1_function_gradient_1D

.. autofunction:: tensor1_function_gradient_1D

.. autofunction:: tensor2_function_gradient_1D

.. autofunction:: tensor2_function_gradient_1D

.. autofunction:: tensor2_function_gradient_1D

.. autofunction:: phi_vector_1D

.. autofunction:: phi_vector_2D

.. autofunction:: phi_vector_3D

.. autofunction:: dphi_vector_1D

.. autofunction:: dphi_vector_2D

.. autofunction:: dphi_vector_3D

.. autofunction:: d2phi_vector_1D

.. autofunction:: d2phi_vector_2D

.. autofunction:: d2phi_vector_3D

.. autofunction:: div_phi_1D

.. autofunction:: div_phi_2D

.. autofunction:: div_phi_3D

.. autofunction:: dof_1D

.. autofunction:: dof_2D

.. autofunction:: dof_3D

.. autofunction:: dof_approximation_1D

.. autofunction:: dof_approximation_2D

.. autofunction:: dof_approximation_3D

.. autofunction:: dof_gradient_approximation_1D

.. autofunction:: dof_gradient_approximation_2D

.. autofunction:: dof_gradient_approximation_3D

.. autofunction:: dof_hessian_approximation_1D

.. autofunction:: dof_hessian_approximation_2D

.. autofunction:: dof_hessian_approximation_3D

.. autofunction:: vector_dof_approximation_1D

.. autofunction:: vector_dof_approximation_2D

.. autofunction:: vector_dof_approximation_3D

.. autofunction:: vector_dof_gradient_approximation_1D

.. autofunction:: vector_dof_gradient_approximation_2D

.. autofunction:: vector_dof_gradient_approximation_3D

.. autofunction:: adouble_1D

.. autofunction:: adouble_2D

.. autofunction:: adouble_3D

.. autofunction:: tensor1_1D

.. autofunction:: tensor1_2D

.. autofunction:: tensor1_3D

.. autofunction:: tensor2_1D

.. autofunction:: tensor2_2D

.. autofunction:: tensor2_3D

.. autofunction:: tensor3_1D

.. autofunction:: tensor3_2D

.. autofunction:: tensor3_3D

.. autofunction:: adouble_tensor1_1D

.. autofunction:: adouble_tensor1_2D

.. autofunction:: adouble_tensor1_3D

.. autofunction:: adouble_tensor2_1D

.. autofunction:: adouble_tensor2_2D

.. autofunction:: adouble_tensor2_3D

.. autofunction:: adouble_tensor3_1D

.. autofunction:: adouble_tensor3_2D

.. autofunction:: adouble_tensor3_3D


deal.II Function<dim> classes
=============================
.. autosummary::
    :nosignatures:

    Function_1D
    Function_2D
    Function_3D
    adoubleFunction_1D
    adoubleFunction_2D
    adoubleFunction_3D
    ConstantFunction_1D
    ConstantFunction_2D
    ConstantFunction_3D
    adoubleConstantFunction_1D
    adoubleConstantFunction_2D
    adoubleConstantFunction_3D
    TensorFunction_1_1D
    TensorFunction_1_2D
    TensorFunction_1_3D
    TensorFunction_2_1D
    TensorFunction_2_2D
    TensorFunction_2_3D
    adoubleTensorFunction_1_1D
    adoubleTensorFunction_1_2D
    adoubleTensorFunction_1_3D
    adoubleTensorFunction_2_1D
    adoubleTensorFunction_2_2D
    adoubleTensorFunction_2_3D

.. autoclass:: pyDealII.Function_1D

.. autoclass:: pyDealII.Function_2D

.. autoclass:: pyDealII.Function_3D

.. autoclass:: pyDealII.adoubleFunction_1D

.. autoclass:: pyDealII.adoubleFunction_2D

.. autoclass:: pyDealII.adoubleFunction_3D

.. autoclass:: pyDealII.ConstantFunction_1D

.. autoclass:: pyDealII.ConstantFunction_2D

.. autoclass:: pyDealII.ConstantFunction_3D

.. autoclass:: pyDealII.adoubleConstantFunction_1D

.. autoclass:: pyDealII.adoubleConstantFunction_2D

.. autoclass:: pyDealII.adoubleConstantFunction_3D

.. autoclass:: pyDealII.TensorFunction_1_1D

.. autoclass:: pyDealII.TensorFunction_1_2D

.. autoclass:: pyDealII.TensorFunction_1_3D

.. autoclass:: pyDealII.TensorFunction_2_1D

.. autoclass:: pyDealII.TensorFunction_2_2D

.. autoclass:: pyDealII.TensorFunction_2_3D

.. autoclass:: pyDealII.adoubleTensorFunction_1_1D

.. autoclass:: pyDealII.adoubleTensorFunction_1_2D

.. autoclass:: pyDealII.adoubleTensorFunction_1_3D

.. autoclass:: pyDealII.adoubleTensorFunction_2_1D

.. autoclass:: pyDealII.adoubleTensorFunction_2_2D

.. autoclass:: pyDealII.adoubleTensorFunction_2_3D


Enumerations and Constants
==========================
.. autosummary::
    :nosignatures:

    fe_i
    fe_j
    fe_q

.. data:: pyDealII.fe_i

.. data:: pyDealII.fe_j

.. data:: pyDealII.fe_q



Auxiliary Classes
=================
.. autosummary::
    :nosignatures:

    Tensor_1_1D
    Tensor_1_2D
    Tensor_1_3D
    Tensor_2_1D
    Tensor_2_2D
    Tensor_2_3D
    adoubleTensor_1_1D
    adoubleTensor_1_2D
    adoubleTensor_1_3D
    adoubleTensor_2_1D
    adoubleTensor_2_2D
    adoubleTensor_2_3D
    Point_1D
    Point_2D
    Point_3D
    Vector
    FullMatrix
    SparseMatrix

..
    Not really needed in python
    feRuntimeNumber_1D
    feRuntimeNumber_2D
    feRuntimeNumber_3D

.. autoclass:: pyDealII.Tensor_1_1D

.. autoclass:: pyDealII.Tensor_1_2D

.. autoclass:: pyDealII.Tensor_1_3D

.. autoclass:: pyDealII.Tensor_2_1D

.. autoclass:: pyDealII.Tensor_2_2D

.. autoclass:: pyDealII.Tensor_2_3D

.. autoclass:: pyDealII.adoubleTensor_1_1D

.. autoclass:: pyDealII.adoubleTensor_1_2D

.. autoclass:: pyDealII.adoubleTensor_1_3D

.. autoclass:: pyDealII.adoubleTensor_2_1D

.. autoclass:: pyDealII.adoubleTensor_2_2D

.. autoclass:: pyDealII.adoubleTensor_2_3D

.. autoclass:: pyDealII.Point_1D

.. autoclass:: pyDealII.Point_2D

.. autoclass:: pyDealII.Point_3D

.. autoclass:: pyDealII.Vector

.. autoclass:: pyDealII.FullMatrix

.. autoclass:: pyDealII.SparseMatrix

..
    Not really needed in python
    .. autoclass:: pyDealII.feRuntimeNumber_1D
    .. autoclass:: pyDealII.feRuntimeNumber_2D
    .. autoclass:: pyDealII.feRuntimeNumber_3D
