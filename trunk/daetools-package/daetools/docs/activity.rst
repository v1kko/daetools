******************
Module pyActivity
******************
..  
    Copyright (C) Dragan Nikolic
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

Overview
==========
.. uml:: pyActivity

Classes
=======
.. py:module:: pyActivity
.. py:currentmodule:: pyActivity

.. autosummary::
    daeSimulation
    daeOptimization
    daeActivity

.. autoclass:: pyActivity.daeSimulation
    :members:
    :undoc-members:
    :exclude-members: Initialize, SolveInitial, DAESolver, Log, DataReporter, RelativeTolerance, AbsoluteTolerances, LoadInitializationValues, StoreInitializationValues,
                      TotalNumberOfVariables, NumberOfEquations, m, model, Model,
                      CleanUpSetupData, Finalize,
                      SetUpParametersAndDomains, SetUpVariables,
                      SetUpOptimization, CreateInequalityConstraint, CreateEqualityConstraint, SetContinuousOptimizationVariable, SetIntegerOptimizationVariable, SetBinaryOptimizationVariable, OptimizationVariables, Constraints, ObjectiveFunction,
                      SetUpParameterEstimation, SetMeasuredVariable, SetInputVariable, SetModelParameter, InputVariables, MeasuredVariables, ModelParameters, 
                      SetUpSensitivityAnalysis,
                      Run, ReRun, Pause, Resume, Integrate, IntegrateForTimeInterval, IntegrateUntilTime, Reinitialize, Reset,
                      CurrentTime, TimeHorizon, ReportingInterval, NextReportingTime, ReportingTimes,
                      RegisterData, ReportData,
                      NumberOfObjectiveFunctions, ObjectiveFunctions, 
                      ActivityAction, IndexMappings, InitialConditionMode, SimulationMode, VariableTypes

                
    .. rubric:: Initialization methods

    .. automethod:: __init__
    .. automethod:: Initialize
    .. automethod:: SolveInitial
    .. autoattribute:: m
    .. autoattribute:: model
    .. autoattribute:: Model
    .. autoattribute:: DAESolver
    .. autoattribute:: Log
    .. autoattribute:: DataReporter
    .. autoattribute:: AbsoluteTolerances
    .. autoattribute:: RelativeTolerance
    .. autoattribute:: TotalNumberOfVariables
    .. autoattribute:: NumberOfEquations

    .. rubric:: Loading/storing the initialization data

    .. automethod:: LoadInitializationValues
    .. automethod:: StoreInitializationValues
    
    .. rubric:: Clean up methods

    .. method:: CleanUpSetupData((daeSimulation)self) -> None
    .. automethod:: Finalize

    
    .. rubric:: Simulation setup methods

    .. method:: SetUpParametersAndDomains((daeSimulation)self) -> None

    .. method:: SetUpVariables((daeSimulation)self) -> None


    .. rubric:: Optimization setup methods

    .. method:: SetUpOptimization((daeSimulation)self) -> None
    .. automethod:: CreateInequalityConstraint
    .. automethod:: CreateEqualityConstraint
    .. automethod:: SetContinuousOptimizationVariable
    .. automethod:: SetIntegerOptimizationVariable
    .. automethod:: SetBinaryOptimizationVariable
    .. autoattribute:: OptimizationVariables
    .. autoattribute:: Constraints
    .. autoattribute:: NumberOfObjectiveFunctions
    .. autoattribute:: ObjectiveFunction


    .. rubric:: Parameter estimation setup methods

    .. method:: SetUpParameterEstimation((daeSimulation)self) -> None
    .. automethod:: SetMeasuredVariable
    .. automethod:: SetInputVariable
    .. automethod:: SetModelParameter
    .. autoattribute:: InputVariables
    .. autoattribute:: MeasuredVariables
    .. autoattribute:: ModelParameters


    .. rubric:: Parameter estimation setup methods

    .. method:: SetUpSensitivityAnalysis((daeSimulation)self) -> None

   
    .. rubric:: Operating procedures methods

    .. method:: Run((daeSimulation)self) -> None
    .. automethod:: ReRun
    .. automethod:: Pause
    .. automethod:: Resume
    .. autoattribute:: ActivityAction

    .. automethod:: Integrate
    .. automethod:: IntegrateForTimeInterval
    .. automethod:: IntegrateUntilTime
    .. automethod:: Reinitialize
    .. automethod:: Reset

    .. autoattribute:: CurrentTime
    .. autoattribute:: TimeHorizon
    .. autoattribute:: ReportingInterval
    .. autoattribute:: NextReportingTime
    .. autoattribute:: ReportingTimes

    .. rubric:: Data reporting methods

    .. automethod:: ReportData
    .. autoattribute:: ReportTimeDerivatives
    .. autoattribute:: ReportSensitivities

    .. rubric:: Various information

    .. autoattribute:: IndexMappings
    .. autoattribute:: InitialConditionMode
    .. autoattribute:: SimulationMode
    .. autoattribute:: VariableTypes

.. autoclass:: pyActivity.daeOptimization
    :members:
    :undoc-members:
    :exclude-members: Initialize, Run, Finalize

    .. automethod:: __init__
    .. automethod:: Initialize
    .. automethod:: Run
    .. automethod:: Finalize

.. autoclass:: pyDAE.daeActivity
    :members:
    :undoc-members:

Enumerations
=============
.. autosummary::
    :nosignatures:

    daeeStopCriterion
    daeeActivityAction
    daeeSimulationMode

.. autoclass:: pyActivity.daeeStopCriterion
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyActivity.daeeActivityAction
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pyActivity.daeeSimulationMode
    :members:
    :undoc-members:
    :exclude-members: names, values

