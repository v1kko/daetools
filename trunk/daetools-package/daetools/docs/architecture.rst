.. _architecture:

************
Architecture
************
..
    Copyright (C) Dragan Nikolic, 2014
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

**DAE Tools** consists of several interdependent components:

* Model
* Simulation
* Optimization
* DAE solver
* LA solver
* NLP solver
* Log
* Data reporter
* Data receiver

The components are located in the following modules:

* pyCore module (Model and Log components)
* pyActivity module (Simulation and Optimization components)
* pyIDAS module (DAE solver component)
* pyDataReporting module (Data reporter and Data receiver components)
* pyUnits module
* Large number of third party linear equation solver modules (LA solver component):
  pySuperLU, pySuperLU_MT, pyTrilinos, pyPardiso, pyIntelPardiso
* Large number of third party NLP/MINLP solver modules (NLP solver component):
  pyIPOPT, pyBONMIN, pyNLOPT

An overview of **DAE Tools** components and their interdepedency is presented in the
:ref:`Figure-Architecture`.

.. _Figure-Architecture:
.. figure:: _static/daetools-architecture.png
   :width: 400 pt
   :figwidth: 450 pt
   :align: center

   **DAE Tools** architecture
