*******************
Third party solvers
*******************

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

.. py:currentmodule:: daetools.solvers.daeMinpackLeastSq
.. autoclass:: daetools.solvers.daeMinpackLeastSq.daeMinpackLeastSq
    :members:
    :undoc-members:

    
.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
