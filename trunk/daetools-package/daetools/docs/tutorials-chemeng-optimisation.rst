*******************************************
Chemical Engineering Optimisation Examples
*******************************************
..
    Copyright (C) Dragan Nikolic, 2016
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.


==========================   =================================================================
:ref:`tutorial_che_opt_1`    |tceo_1|
--------------------------   -----------------------------------------------------------------
:ref:`tutorial_che_opt_2`    |tceo_2|
--------------------------   -----------------------------------------------------------------
:ref:`tutorial_che_opt_3`    |tceo_3|
--------------------------   -----------------------------------------------------------------
:ref:`tutorial_che_opt_4`    |tceo_4|
--------------------------   -----------------------------------------------------------------
:ref:`tutorial_che_opt_5`    |tceo_5|
--------------------------   -----------------------------------------------------------------
:ref:`tutorial_che_opt_6`    |tceo_6|
==========================   =================================================================

The implementations of the `COPS <http://www.mcs.anl.gov/~more/cops>`_ tests differ from the
original ones in following:

- The Direct Sequential Approach has been applied while the original tests use
  the Direct Simultaneous Approach
- The analytical sensitivity Hessian matrix is not available. The limited memory
  Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm from IPOPT is used.

As a consequence, the results slightly differ from the published results.
In addition, the solver options should be tuned to achieve faster convergence.


.. |tceo_1| replace:: Optimisation of the CSTR with energy balance and Van de Vusse reactions
                      (not fully implemented yet).

.. |tceo_2| replace:: COPS test 5 (parameter estimation): Determination of the reaction coefficients
                      in the thermal isometrization of α-pinene.

.. |tceo_3| replace:: COPS test 6 (parameter estimation): Determine stage specific growth and mortality rates
                      for species at each stage as a function of time.

.. |tceo_4| replace:: COPS test 12 (parameter estimation): Determination of the reaction coefficients
                      for the catalytic cracking of gas oil and other byproducts.

.. |tceo_5| replace:: COPS test 13 (parameter estimation): Determination of the reaction coefficients
                      for the conversion of methanol into various hydrocarbons.

.. |tceo_6| replace:: COPS test 14 (optimal control): Catalyst mixing in a tubular plug flow reactor.


.. _tutorial_che_opt_1:

Chem. Eng. Optimisation Example 1
=================================
.. rubric:: Description

.. automodule:: daetools.examples.tutorial_che_opt_1
   :no-members:
   :no-undoc-members:

.. rubric:: Files 

=====================   =================================================================
Model report            `tutorial_che_opt_1.xml <../examples/tutorial_che_opt_1.xml>`_
Runtime model report    `tutorial_che_opt_1-rt.xml <../examples/tutorial_che_opt_1-rt.xml>`_
Source code             `tutorial_che_opt_1.py <../examples/tutorial_che_opt_1.html>`_
=====================   =================================================================


.. _tutorial_che_opt_2:

Chem. Eng. Optimisation Example 2
=================================
.. rubric:: Description

.. automodule:: daetools.examples.tutorial_che_opt_2
   :no-members:
   :no-undoc-members:

.. rubric:: Files

=====================   =================================================================
Model report            `tutorial_che_opt_2.xml <../examples/tutorial_che_opt_2.xml>`_
Runtime model report    `tutorial_che_opt_2-rt.xml <../examples/tutorial_che_opt_2-rt.xml>`_
Source code             `tutorial_che_opt_2.py <../examples/tutorial_che_opt_2.html>`_
=====================   =================================================================


.. _tutorial_che_opt_3:

Chem. Eng. Optimisation Example 3
=================================
.. rubric:: Description

.. automodule:: daetools.examples.tutorial_che_opt_3
   :no-members:
   :no-undoc-members:

.. rubric:: Files

=====================   =================================================================
Model report            `tutorial_che_opt_3.xml <../examples/tutorial_che_opt_3.xml>`_
Runtime model report    `tutorial_che_opt_3-rt.xml <../examples/tutorial_che_opt_3-rt.xml>`_
Source code             `tutorial_che_opt_3.py <../examples/tutorial_che_opt_3.html>`_
=====================   =================================================================


.. _tutorial_che_opt_4:

Chem. Eng. Optimisation Example 4
=================================
.. rubric:: Description

.. automodule:: daetools.examples.tutorial_che_opt_4
   :no-members:
   :no-undoc-members:

.. rubric:: Files

=====================   =================================================================
Model report            `tutorial_che_opt_4.xml <../examples/tutorial_che_opt_4.xml>`_
Runtime model report    `tutorial_che_opt_4-rt.xml <../examples/tutorial_che_opt_4-rt.xml>`_
Source code             `tutorial_che_opt_4.py <../examples/tutorial_che_opt_4.html>`_
=====================   =================================================================


.. _tutorial_che_opt_5:

Chem. Eng. Optimisation Example 5
=================================
.. rubric:: Description

.. automodule:: daetools.examples.tutorial_che_opt_5
   :no-members:
   :no-undoc-members:

.. rubric:: Files

=====================   =================================================================
Model report            `tutorial_che_opt_5.xml <../examples/tutorial_che_opt_5.xml>`_
Runtime model report    `tutorial_che_opt_5-rt.xml <../examples/tutorial_che_opt_5-rt.xml>`_
Source code             `tutorial_che_opt_5.py <../examples/tutorial_che_opt_5.html>`_
=====================   =================================================================


.. _tutorial_che_opt_6:

Chem. Eng. Optimisation Example 6
=================================
.. rubric:: Description

.. automodule:: daetools.examples.tutorial_che_opt_6
   :no-members:
   :no-undoc-members:

.. rubric:: Files

=====================   =================================================================
Model report            `tutorial_che_opt_6.xml <../examples/tutorial_che_opt_6.xml>`_
Runtime model report    `tutorial_che_opt_6-rt.xml <../examples/tutorial_che_opt_6-rt.xml>`_
Source code             `tutorial_che_opt_6.py <../examples/tutorial_che_opt_6.html>`_
=====================   =================================================================





