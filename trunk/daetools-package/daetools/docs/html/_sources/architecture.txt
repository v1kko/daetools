.. _architecture:

************
Architecture
************
..
    Copyright (C) Dragan Nikolic, 2013
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
  pySuperLU, pySuperLU_MT, pyTrilinos
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

pyCore module
-------------
pyCore module defines the key modelling concepts such as:

* Model
 A model of the process is a simplified abstraction of real world process/phenomena describing its most important/driving elements and their interactions. In **DAE Tools** models are created by defining their parameters, distribution domains, variables, equations, and ports.

* Distribution domain
 Domain is a general term used to define an array of different objects (parameters, variables, equations but models and ports as well).

* Parameter
 Parameter can be defined as a time invariant quantity that will not change during a simulation.

* Variable
 Variable can be defined as a time variant quantity, also called a *state variable*.

* Equation
 Equation can be defined as an expression used to calculate a variable value, which can be created
 by performing basic mathematical operations (+, -, *, /) and functions (such as sin, cos, tan, sqrt, log, ln, exp, pow, abs etc)
 on parameter and variable values (and time and partial derivatives as well).

* State transition network
 State transition networks are used to model a special type of equations:
 *discontinuous equation*s. Discontinuous equations are equations that take different forms subject to certain conditions.
 They are composed of a finite number of *states*.

* State
 States can be defined as a set of actions (in our case a set of equations) under current operating conditions.
 In addition, every state contains a set of state transitions which describe conditions
 when the state changes occur.

* OnEvent and OnCondition event handlers

* Port
 Ports are objects used to connect two model instances and exchange continuous information.
 Like models, they may contain domains, parameters and variables.

* EventPort
 Event ports are objects used to connect two model instances and exchange discrete information (events/messages).

* Log
 Log is defined as an object used to send messages from the various parts of **DAE Tools** framework
 (messages from solvers or simulation).

pyActivity module
-----------------

* Simulation
 Simulation of a process can be considered as the model run for certain input conditions.
 To define a simulation, several tasks are necessary such as: specifying information about domains and parameters,
 fixing the degrees of freedom by assigning values to certain variables, setting the initial conditions and many other
 (setting the initial guesses, absolute tolerances, etc).

* Optimization
 Process optimization can be considered as a process adjustment so as to minimize or maximize a specified goal while
 satisfying imposed set of constraints. The most common goals are minimizing cost, maximizing throughput, and/or
 efficiency. In general there are three types of parameters that can be adjusted to affect optimal performance:

 * Equipment optimization
 * Operating procedures
 * Control optimization

pyDataReporting module
----------------------

* Data Reporter
 Data reporter is defined as an object used to report the results of a simulation/optimization.
 They can either keep the results internally (and export them into a file, for instance) or send them via
 TCP/IP protocol to the **DAE Tools** plotter.

* Data Receiver
 Data receiver can be defined as on object which duty is to receive the results from a data reporter.
 These data can be later plotted or processed in some other ways.

pyIDAS module
-------------
Contains an implementation of the `Sundials IDAS <https://computation.llnl.gov/casc/sundials/main.html>`_
DAE solver.

pyUnits module
------------
Defines two key concepts:

* Unit (SI)
 7 fundamental dimensions (length, mass, time, electrical current, temperature, luminous intensity, amount of substance)
 * Multiplier
 * Offset

* Quantity
 * Value
 * Unit

NLP/MINLP modules
-----------------
Contain implementations of various NLP/MINLP solvers:

* `IPOPT <https://projects.coin-or.org/IPOPT>`_ in the pyIPOPTmodule
* `NLOPT <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_ in the pyNLOPT module
* `BONMIN <https://projects.coin-or.org/Bonmin>`_ in the pyBONMIN module

LA solver modules
-----------------
Contain implementations of various third party linear equation solvers:

* `Trilinos Amesos <http://trilinos.sandia.gov/packages/amesos>`_ in the pyTrilinos module
* `Trilinos AztecOO <http://trilinos.sandia.gov/packages/aztecoo>`_ in the pyTrilinos module
* `SuperLU <http://crd.lbl.gov/~xiaoye/SuperLU/index.html>`_ in the pySuperLU module
* `SuperLU_MT <http://crd.lbl.gov/~xiaoye/SuperLU/index.html>`_ in the pySuperLU_MT module
* `Intel MKL <http://software.intel.com/en-us/intel-mkl>`_ in the pyTrilinos module


.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
