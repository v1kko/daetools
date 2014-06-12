*******************
Third party solvers
*******************
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

Linear solvers
==============
.. py:currentmodule:: pyCore

.. class:: daeIDALASolver_t

   .. attribute:: Name

   .. method:: SaveAsXPM((daeIDALASolver_t)self, (str)xpmFilename) -> int

SuperLU
-------
.. py:module:: solvers.superlu.pySuperLU

.. rubric:: Instantiation function
.. autofunction:: pySuperLU.daeCreateSuperLUSolver

.. rubric:: Classes
.. autoclass:: pySuperLU.daeSuperLU_Solver
    :members:
    :undoc-members:
    :exclude-members: SaveAsXPM, Name

.. autoclass:: pySuperLU.superlu_options_t
    :members:
    :undoc-members:

.. rubric:: Enumerations
.. autoclass:: pySuperLU.IterRefine_t
    :members:
    :undoc-members:
    :exclude-members: names, values

.. autoclass:: pySuperLU.rowperm_t
    :members:
    :undoc-members:
    :exclude-members: names, values
    
.. autoclass:: pySuperLU.yes_no_t
    :members:
    :undoc-members:
    :exclude-members: names, values
    
.. autoclass:: pySuperLU.colperm_t
    :members:
    :undoc-members:
    :exclude-members: names, values
    

SuperLU_MT
----------
.. py:module:: solvers.superlu_mt.pySuperLU_MT

.. rubric:: Instantiation function
.. autofunction:: pySuperLU_MT.daeCreateSuperLUSolver

.. rubric:: Classes
.. autoclass:: pySuperLU_MT.daeSuperLU_MT_Solver
    :members:
    :undoc-members:
    :exclude-members: SaveAsXPM, Name

.. autoclass:: pySuperLU_MT.superlumt_options_t
    :members:
    :undoc-members:

.. rubric:: Enumerations
.. autoclass:: pySuperLU_MT.yes_no_t
    :members:
    :undoc-members:
    :exclude-members: names, values
    
.. autoclass:: pySuperLU_MT.colperm_t
    :members:
    :undoc-members:
    :exclude-members: names, values
    

Trilinos
--------
.. py:module:: solvers.trilinos.pyTrilinos

.. rubric:: Instantiation function
.. autofunction:: pyTrilinos.daeTrilinosSupportedSolvers
.. autofunction:: pyTrilinos.daeCreateTrilinosSolver

.. rubric:: Classes
.. autoclass:: pyTrilinos.daeTrilinosSolver
    :members:
    :undoc-members:
    :exclude-members: SaveAsXPM, Name
    
.. autoclass:: pyTrilinos.TeuchosParameterList
    :members:
    :undoc-members:


Pardiso
-------
.. py:module:: solvers.pardiso.pyPardiso

.. rubric:: Instantiation function
.. autofunction:: pyPardiso.daeCreatePardisoSolver

.. rubric:: Classes
.. autoclass:: pyPardiso.daePardisoSolver
    :members:
    :undoc-members:
    :exclude-members: SaveAsXPM, Name


IntelPardiso
------------
.. py:module:: solvers.intel_pardiso.pyIntelPardiso

.. rubric:: Instantiation function
.. autofunction:: pyIntelPardiso.daeCreateIntelPardisoSolver

.. rubric:: Classes
.. autoclass:: pyIntelPardiso.daeIntelPardisoSolver
    :members:
    :undoc-members:
    :exclude-members: SaveAsXPM, Name

    
Optimization solvers
====================
.. py:currentmodule:: pyCore

.. class:: daeIDALASolver_t

   .. attribute:: Name

   .. method:: Initialize((daeIDALASolver_t)self, (daeSimulation_t)simulation, (daeDAESolver_t)daeSolver, (daeDataReporter_t)dataReporter, (daeLog_t)log) -> None

   .. method:: Solve((daeIDALASolver_t)self) -> None


.. py:module:: solvers.ipopt.pyIPOPT
.. autoclass:: pyIPOPT.daeIPOPT
    :members:
    :undoc-members:
    :exclude-members: Name, Initialize, Solve

.. py:module:: solvers.bonmin.pyBONMIN
.. autoclass:: pyBONMIN.daeBONMIN
    :members:
    :undoc-members:
    :exclude-members: Name, Initialize, Solve

.. py:module:: solvers.nlopt.pyNLOPT
.. autoclass:: pyNLOPT.daeNLOPT
    :members:
    :undoc-members:
    :exclude-members: Name, Initialize, Solve

Parameter estimation solvers
============================

.. py:currentmodule:: daetools.solvers.minpack
.. autoclass:: daetools.solvers.minpack.daeMinpackLeastSq
    :members:
    :undoc-members:
