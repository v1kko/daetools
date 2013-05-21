*****************
pyDAE User Guide
*****************
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

The main concepts
=================

**DAE Tools** contains 5 modules:

* Core
* Activity
* DataReporting
* IDAS
* Units

Core module
-----------

Core module defines the main modelling concepts:

* **Model**
 A model of the process is a simplified abstraction of real world process/phenomena describing its most important/driving elements and their interactions. In **DAE Tools** models are created by defining their parameters, distribution domains, variables, equations, and ports.

 * **Distribution domain**
  Domain is a general term used to define an array of different objects (parameters, variables, equations but models and ports as well).

 * **Parameter**
  Parameter can be defined as a time invariant quantity that will not change during a simulation.

 * **Variable**
  Variable can be defined as a time variant quantity, also called a *state variable*.

 * **Equation**
  Equation can be defined as an expression used to calculate a variable value, which can be created
  by performing basic mathematical operations (+, -, *, /) and functions (such as sin, cos, tan, sqrt, log, ln, exp, pow, abs etc)
  on parameter and variable values (and time and partial derivatives as well).

 * **State transition network**
  State transition networks are used to model a special type of equations:
  *discontinuous equation*s. Discontinuous equations are equations that take different forms subject to certain conditions.
  They are composed of a finite number of *states*.

 * **State**
  States can be defined as a set of actions (in our case a set of equations) under current operating conditions.
  In addition, every state contains a set of state transitions which describe conditions
  when the state changes occur.

 * **State Transition**
  State transition can be defined as a transition from the current to some other state, subject to given conditions.

 * **Port**
  Ports are objects used to connect two model instances and exchange continuous information.
  Like models, they may contain domains, parameters and variables.

 * **EventPort**
  Event ports are objects used to connect two model instances and exchange discrete information (events/messages).

* **Simulation**
 Simulation of a process can be considered as the model run for certain input conditions.
 To define a simulation, several tasks are necessary such as: specifying information about domains and parameters,
 fixing the degrees of freedom by assigning values to certain variables, setting the initial conditions and many other
 (setting the initial guesses, absolute tolerances, etc).

* **Optimization**
 Process optimization can be considered as a process adjustment so as to minimize or maximize a specified goal while
 satisfying imposed set of constraints. The most common goals are minimizing cost, maximizing throughput, and/or
 efficiency. In general there are three types of parameters that can be adjusted to affect optimal performance:

 * Equipment optimization
 * Operating procedures
 * Control optimization

* **Solver**
 Solver is a set of mathematical procedures/algorithms necessary to solve a given set of equations.
 There are several types of solvers: Linear Algebraic solvers (**LA**), used to solve linear systems of equations;
 Nonlinear Algebraic solvers (**NLA**), used to solve non-linear systems of equations; Differential Algebraic solvers
 (**DAE**), used to solve mixed systems of differential and algebraic equations; Nonlinear Programming solvers (**NLP**),
 used to solve nonlinear optimization problems; Mixed-integer Nonlinear Programming solvers (**MINLP**), used to solve
 mixed-integer nonlinear optimization problems. In **DAE Tools** it is possible to choose **DAE**
 (currently only `Sundials IDAS <https://computation.llnl.gov/casc/sundials/main.html>`_), **NLP/MINLP**
 (currently `IPOPT/BONMIN <https://projects.coin-or.org/Bonmin>`_
 and `NLOPT <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_),
 and **LA** solvers (built-in Sundials LA solvers; `Trilinos Amesos <http://trilinos.sandia.gov/packages/amesos>`_;
 `Trilinos AztecOO <http://trilinos.sandia.gov/packages/aztecoo>`_;
 `SuperLU/SuperLU_MT <http://crd.lbl.gov/~xiaoye/SuperLU/index.html>`_;
 `Intel MKL <http://software.intel.com/en-us/intel-mkl>`_; `AMD ACML <http://www.amd.com/acml>`_).

* **Data Reporter**
 Data reporter is defined as an object used to report the results of a simulation/optimization.
 They can either keep the results internally (and export them into a file, for instance) or send them via
 TCP/IP protocol to the **DAE Tools** plotter.

* **Data Receiver**
 Data receiver can be defined as on object which duty is to receive the results from a data reporter.
 These data can be later plotted or processed in some other ways.

* **Log**
 Log is defined as an object used to send messages from the various parts of **DAE Tools** framework
 (messages from solvers or simulation).

Activity module
---------------

DataReporting module
--------------------

IDAS module
-----------

Units module
------------




Distribution domains
--------------------
There are two types of domains in **DAE Tools**: simple arrays and distributed domains (commonly used to distribute variables,
parameters and equations in space). The distributed domains can have a uniform (default) or a user specified non-uniform grid.
At the moment, only the following finite difference methods can be used to calculate partial derivatives:

 * Backward finite difference method (BFD)
 * Forward finite difference method (FFD)
 * Center finite difference method (CFD)

In **DAE Tools** just anything can be distributed on domains: parameters, variables, equations even models and ports.
Obviously it does not have a physical meaning to distribute a model on a domain, However that can be useful for modelling
of complex processes where we can create an array of models where each point in a distributed domain have a corresponding
model so that a user does not have to take care of number of points in the domain, etc. In addition, domain points values
can be obtained as a **NumPy** one-dimensional array; this way **DAE Tools** can be easily used in conjuction with other
scientific python libraries `NumPy <http://numpy.scipy.org>`_, `SciPy <http://www.scipy.org>`_, for instance and many
`other <http://www.scipy.org/Projects>`_.

Domains in **pyDAE** can be defined by the following statement:

.. code-block:: python

    myDomain = daeDomain("myDomain", Parent_Model_or_Port, Description)

while in **cDAE**:

.. code-block:: cpp

    daeDomain myDomain("myDomain", &Parent_Model_or_Port, Description);

More information about domains can be found in :doc:`pyDAE_user_guide` and :py:class:`pyCore.daeDomain`.
Also, do not forget to have a look on :doc:`tutorials`.

Parameters
----------

There are two types of parameters in **DAE Tools**: ordinary and distributed. Several functions to get a parameter
value (function call operator :py:meth:`~pyCore.daeParameter.__call__`) and array of values
(:py:meth:`~pyCore.daeParameter.array`) have been defined. In addition, distributed parameters have
:py:attr:`~pyCore.daeParameter.npyValues` property to get the values as a numpy multi-dimensional array.

Parameters in **pyDAE** can be defined by the following statement:

.. code-block:: python

    myParam = daeParameter("myParam", eReal, Parent_Model_or_Port, "Description")

while in **cDAE**:

.. code-block:: cpp

    daeParameter myParam("myParam", eReal, &Parent_Model_or_Port, "Description");

More information about parameters can be found in :doc:`pyDAE_user_guide` and :py:class:`pyCore.daeParameter`.
Also, do not forget to have a look on :doc:`tutorials`.

Variables
---------
There are two types of variables in **DAE Tools**: ordinary and distributed. Functions to get a variable value
(function call operator :py:meth:`~pyCore.daeVariable.__call__`), a time or a partial derivative
(:py:meth:`~pyCore.daeVariable.dt`, :py:meth:`~pyCore.daeVariable.d`, or :py:meth:`~pyCore.daeVariable.d2`) or
functions to obtain an array of values, time or partial derivatives (:py:meth:`~pyCore.daeVariable.array`,
:py:meth:`~pyCore.daeVariable.dt_array`, :py:meth:`~pyCore.daeVariable.d_array`, or :py:meth:`~pyCore.daeVariable.d2_array`)
have been defined. In addition, distributed variables have :py:attr:`~pyCore.daeVariable.npyValues` property to get
the values as a numpy multi-dimensional array.

Variables in **pyDAE** can be defined by the following statement:

.. code-block:: python

    myVar = daeVariable("myVar", variableType, Parent_Model_or_Port, "Description")

while in **cDAE**:

.. code-block:: cpp

    daeVariable myVar("myVar", variableType, &Parent_Model_or_Port, "Description");

More information about variables can be found in :doc:`pyDAE_user_guide` and :py:class:`pyCore.daeVariable`.
Also, do not forget to have a look on :doc:`tutorials`.

Equations
---------
**DAE Tools** introduce two types of equations: ordinary and distributed. What makes distributed
equations special is that an equation expression is valid on every point within the domains that
the equations is distriibuted on. Equations can be distributed on a whole domain, on a part of it
or on some of the points in a domain. Equations in **pyDAE** can be defined by the following statement:

.. code-block:: python

    eq = model.CreateEquation("myEquation", "Description")

while in **cDAE**:

.. code-block:: cpp

    daeEquation* eq = model.CreateEquation("myEquation", "Description");

To define an equation expression (used to calculate its residual and its gradient - which represent a single row in a
Jacobian matrix) **DAE Tools** combine the
`operator overloading <http://en.wikipedia.org/wiki/Automatic_differentiation#Operator_overloading>`_
technique for `automatic differentiation <http://en.wikipedia.org/wiki/Automatic_differentiation>`_
(adopted from `ADOL-C <https://projects.coin-or.org/ADOL-C>`_ library) with the concept of representing equations as
**evaluation trees**. Evaluation trees are made of binary or unary nodes, itself representing four basic mathematical
operations and frequently used mathematical functions, such as ``sin, cos, tan, sqrt, pow, log, ln, exp, min, max, floor, ceil,
abs, sum, product, ...``. These basic mathematical operations and functions are implemented to operate on **a heavily
modified ADOL-C** library class :py:class:`~pyCore.adouble` (which has been extended to contain information about
domains/parameters/variables etc). In adition, a new :py:class:`~pyCore.adouble_array` class has been introduced to apply all
above-mentioned operations on arrays of variables.
What is different here is that :py:class:`~pyCore.adouble`/:py:class:`~pyCore.adouble_array` classes and mathematical
operators/functions work in two modes; they can either **build-up an evaluation tree** or **calculate a value of an expression**.
Once built the evaluation trees can be used to calculate equation residuals or derivatives to fill a Jacobian matrix
necessary for a Newton-type iteration. A typical evaluation tree is presented in :ref:`Figure 4. <Figure-4>`.

.. _Figure-4:
.. figure:: _static/EvaluationTree.png
   :width: 250 pt
   :figwidth: 300 pt
   :align: center

   **Figure 4.** DAE Tools equation evaluation tree

As it has been noted before, domains, parameters, and variables contain functions that return
:py:class:`~pyCore.adouble`/:py:class:`~pyCore.adouble_array` objects, which can be used to calculate
residuals and derivatives. These functions include functions to get a value of
a domain/parameter/variable (function call operator), to get a time or a partial derivative of a variable
(functions :py:meth:`~pyCore.daeVariable.dt`, :py:meth:`~pyCore.daeVariable.d`, or :py:meth:`~pyCore.daeVariable.d2`)
or functions to obtain an array of values, time or partial derivatives (:py:meth:`~pyCore.daeVariable.array`,
:py:meth:`~pyCore.daeVariable.dt_array`, :py:meth:`~pyCore.daeVariable.d_array`, or :py:meth:`~pyCore.daeVariable.d2_array`).
Another useful feature of **DAE Tools** equations is that they can be
exported into MathML or Latex format and easily visualized.

For example, the equation *F* (given in :ref:`Figure 4. <Figure-4>`) can be defined in **pyDAE** by using the following
statements:

.. code-block:: python

    F = model.CreateEquation("F", "F description")
    F.Residal = V14.dt() + V1() / (V14() + 2.5) + Sin(3.14 * V3())

while in **cDAE** by:

.. code-block:: cpp

    daeEquation* F = model.CreateEquation("F", "F description");
    F->SetResidal( V14.dt() + V1() / (V14() + 2.5) + sin(3.14 * V3()) );

More information about equations can be found in :doc:`pyDAE_user_guide` and :py:class:`pyCore.daeEquation`.
Also, do not forget to have a look on :doc:`tutorials`.

State Transition Networks (discontinuous equations)
---------------------------------------------------

Discontinuous equations are equations that take different forms subject to certain conditions. For example,
if we want to model a flow through a pipe we may observe three different flow regimes:

* Laminar: if Reynolds number is less than 2,100
* Transient: if Reynolds number is greater than 2,100 and less than 10,000
* Turbulent: if Reynolds number is greater than 10,000

What we can see is that from any of these three states we can go to any other state. This type of discontinuities
is called a **reversible discontinuity** and can be described by the
:py:meth:`~pyCore.daeModel.IF`, :py:meth:`~pyCore.daeModel.ELSE_IF`, :py:meth:`~pyCore.daeModel.ELSE`
state transient network functions.
In **pyDAE** it is given by the following statement:

.. code-block:: python

    IF(Re() <= 2100)                      # (Laminar flow)
    #... (equations go here)

    ELSE_IF(Re() > 2100 and Re() < 10000) # (Transient flow)
    #... (equations go here)

    ELSE()                                # (Turbulent flow)
    #... (equations go here)

    END_IF()

while in **cDAE** by:

.. code-block:: cpp

    IF(Re() <= 2100);                      // (Laminar flow)
    //... (equations go here)

    ELSE_IF(Re() > 2100 && Re() < 10000);  // (Transient flow)
    //... (equations go here)

    ELSE();                                // (Turbulent flow)
    //... (equations go here)

    END_IF();

**Reversible discontinuities** can be **symmetrical** and **non-symmetrical**. The above example is **symmetrical**.
However, if we have a CPU and we want to model its power dissipation we may have three operating modes with the
following state transitions:

* Normal mode

  * switch to **Power saving mode** if CPU load is below 5%
  * switch to **Fried mode** if the temperature is above 110 degrees

* Power saving mode

  * switch to **Normal mode** if CPU load is above 5%
  * switch to **Fried mode** if the temperature is above 110 degrees

* Fried mode (no escape from here... go to the nearest shop and buy a new one!)

What we can see is that from the **Normal mode** we can either go to the **Power saving mode** or to the **Fried mode**.
The same stands for the **Power saving mode**: we can either go to the **Normal mode** or to the **Fried mode**.
However, once the temperature exceeds 110 degrees the CPU dies (let's say we heavily overclocked it) and there
is no going back. This type of discontinuities is called an **irreversible discontinuity** and can be described by
using  :py:meth:`~pyCore.daeModel.STN`, :py:meth:`~pyCore.daeModel.STATE`, :py:meth:`~pyCore.daeModel.END_STN`
functions while state transitions using :py:meth:`~pyCore.daeModel.ON_CONDITION` function.
In **pyDAE** this type of state transitions is given by the following statement:

.. code-block:: python

    STN("CPU")

    STATE("Normal")
    #... (equations go here)
    ON_CONDITION(CPULoad() < 0.05, switchToState = "PowerSaving")
    ON_CONDITION(T() > 110,        switchToState = "Fried")

    STATE("PowerSaving")
    #... (equations go here)
    ON_CONDITION(CPULoad() >= 0.05, switchToState = "Normal")
    ON_CONDITION(T() > 110,         switchToState = "Fried")

    STATE("Normal")
    #... (equations go here)

    END_STN()

while in **cDAE** by:

.. code-block:: cpp

    STN("CPU");

    STATE("Normal");
    //... (equations go here)
    ON_CONDITION(CPULoad() < 0.05, switchToState = "PowerSaving");
    ON_CONDITION(T() > 110,        switchToState = "Fried");


    STATE("PowerSaving");
    //... (equations go here)
    ON_CONDITION(CPULoad() >= 0.05, switchToState = "Normal");
    ON_CONDITION(T() > 110,         switchToState = "Fried");

    STATE("Normal");
    //... (equations go here)

    END_STN();

More information about state transition networks can be found in :doc:`pyDAE_user_guide` and :py:class:`pyCore.daeSTN`.
Also, do not forget to have a look on :doc:`tutorials`.

Ports
-----

Ports are used to connect two models. Like models, they may contain domains, parameters and variables. For instance,
in **pyDAE** ports can be defined by the following statements:

.. code-block:: python

    class myPort(daePort):
        def __init__(self, Name, Type, Parent = None, Description = ""):
            daePort.__init__(self, Name, Type, Parent, Description)
            #... (here go declarations of domains, parameters and variables)

while in **cDAE** by:

.. code-block:: cpp

    class myPort : public daePort
    {
    public:
    myPort(string strName, daeePortType eType, daeModel* pParent, string strDescription = "")
            : daePort(strName, eType, pParent, strDescription)
        {
            //... (here go additional properties of domains, parameters and variables)
        }

    public:
        //... (here go declarations of domains, parameters and variables)
    };

More information about ports can be found in :doc:`pyDAE_user_guide` and :py:class:`pyCore.daePort`.
Also, do not forget to have a look on :doc:`tutorials`.

Event Ports
-----------

Event ports are also used to connect two models; however, they allow sending of discrete messages (events) between
model instances. Events can be triggered manually or as a result of a state transition in a model. The main difference
between event and ordinary ports is that the former allow a discrete communication between model instances while
latter allow a continuous exchange of information. A single outlet event port can be connected to unlimited number
of inlet event ports. Messages contain a floating point value that can be used by a recipient (these actions are
specified in :py:meth:`~pyCore.daeModel.ON_EVENT` function); that value might be a simple number or an expression
involving model variables/parameters.

More information about event ports can be found in :doc:`pyDAE_user_guide` and :py:class:`pyCore.daeEventPort`.
Also, do not forget to have a look on :doc:`tutorials`.








= Core module =

== Models ==

Models have the following properties:

* '''Name''': string (read-only)<br />Defines a name of an object ("Temperature" for instance)
* '''CanonicalName''': string (read-only)<br />It is a method use to describe a location of the object ("HeatExchanger.Temperature" for instance means that the object Temperature belongs to the parent object HeatExchanger). Object names are separated by dot symbols (".")
* '''Description: '''string
* '''Domains''': daeDomain list
* '''Parameters''': daeParameter list
* '''Variables''': daeVariable list
* '''Equations''': daeEquation list
* '''Ports''': daePort list
* '''ChildModels''': daeModel list
* '''PortArrays''': daePortArray list
* '''ChildModelArrays''': daeModelArray list
* '''InitialConditionMode''': daeeInitialConditionMode

The most important functions are:

* '''ConnectPorts'''
* '''SetReportingOn'''
* '''sum, product, integral, average'''
* '''d, dt'''
* '''CreateEquation'''
* '''IF, ELSE_IF, ELSE, END_IF'''
* '''STN, STATE, SWITCH_TO, END_STN'''

Every user model has to implement two functions: '''__init__''' and '''DeclareEquations'''. '''__init__''' is the constructor and all parameters, distribution domains, variables, ports, and child models must be declared here. '''DeclareEquations''' function is used to declare equations and state transition networks.

Models in '''pyDAE''' can be defined by the following statement:

<syntaxhighlight lang="python">
class myModel(daeModel):
    def __init__(self, Name, Parent = None):
        daeModel.__init__(self, Name, Parent)
        ... (here go declarations of domains, parameters, variables, ports, etc)
    def DeclareEquations(self):
        ... (here go declarations of equations and state transitions)
</syntaxhighlight>

Details of how to declare and use parameters, distribution domains, variables, ports, equations, state transition networks (STN) and child models are given in the following sections.

== Equations ==

'''DAE Tools''' introduce two types of equations: ordinary and distributed. A residual expression of distributed equations is valid on every point in distributed domains that the equations is distriibuted on. The most important equation properties are:

* '''Name''': string (read-only)
* '''CanonicalName''': string (read-only)
* '''Description: '''string
* '''Domains''': daeDomain list (read-only)
* '''Residual''': adouble

=== Declaring equations ===

The following statement is used in '''pyDAE''' to declare an ordinary equation:

<syntaxhighlight lang="python">
eq = model.CreateEquation("MyEquation")
</syntaxhighlight>

while to declare a distributed equation the next statemets are used:

<syntaxhighlight lang="python">
eq = model.CreateEquation("MyEquation")
d = eq.DistributeOnDomain(myDomain, eClosedClosed)
</syntaxhighlight>

Equations can be distributed on a whole domain or on a part of it. Currently there are 7 options:

* Distribute on a closed domain - analogous to: x ∈ '''[ '''x<sub>0</sub>, x<sub>n</sub> ''']'''
* Distribute on a left open domain - analogous to: x ∈ '''( '''x<sub>0</sub>, x<sub>n</sub> ''']'''
* Distribute on a right open domain - analogous to: x ∈ '''[''' x<sub>0</sub>, x<sub>n</sub> ''')'''
* Distribute on a domain open on both sides - analogous to: x ∈ '''(''' x<sub>0</sub>, x<sub>n</sub> ''')'''
* Distribute on the lower bound - only one point: x ∈ { x<sub>0</sub> }<br />This option is useful for declaring boundary conditions.
* Distribute on the upper bound - only one point: x ∈ { x<sub>n</sub> }<br />This option is useful for declaring boundary conditions.
* Custom array of points within a domain

where LB stands for the LowerBound and UB stands for the UpperBound of the domain. An overview of various bounds is given in '''Figures 1a. to 1h.'''. Here we have an equation which is distributed on two domains: '''x''' and '''y''' and we can see various available options. Green squares represent the intervals included in the distributed equation, while white squares represent excluded intervals.

[[Image:EquationBounds CC CC.png|thumb|200px|Figure 1a.<br />x: eClosedClosed; y: eClosedClosed<br />x ∈ [x<sub>0</sub>, x<sub>n</sub>], y ∈ [y<sub>0</sub>, y<sub>n</sub>] ]]
[[Image:EquationBounds OO OO.png|thumb|200px|Figure 1b.<br />x: eOpenOpen; y: eOpenOpen<br />x  ( x<sub>0</sub>, x<sub>n</sub> ), y ∈ ( y<sub>0</sub>, y<sub>n</sub> )]]
[[Image:EquationBounds CC OO.png|thumb|200px|Figure 1c.<br />x: eClosedClosed; y: eOpenOpen<br />x ∈ [x<sub>0</sub>, x<sub>n</sub>], y ∈ ( y<sub>0</sub>, y<sub>n</sub> ) ]]
[[Image:EquationBounds CC OC.png|thumb|200px|Figure 1d.<br />x: eClosedClosed; y: eOpenClosed<br />x ∈ [x<sub>0</sub>, x<sub>n</sub>], y ∈ ( y<sub>0</sub>, y<sub>n</sub> ] ]]
[[Image:EquationBounds LB CO.png|thumb|200px|Figure 1e.<br />x: LB; y: eClosedOpen<br />x = x<sub>0</sub>, y ∈ [ y<sub>0</sub>, y<sub>n</sub> ) ]]
[[Image:EquationBounds LB CC.png|thumb|200px|Figure 1f.<br />x: LB; y: eClosedClosed<br />x = x<sub>0</sub>, y ∈ [y<sub>0</sub>, y<sub>n</sub>] ]]
[[Image:EquationBounds UB CC.png|thumb|200px|Figure 1g.<br />x: UB; y: eClosedClosed<br />x = x<sub>n</sub>, y ∈ [y<sub>0</sub>, y<sub>n</sub>] ]]
[[Image:EquationBounds LB UB.png|thumb|200px|Figure 1h.<br />x: LB; y: UB<br />x = x<sub>0</sub>, y = y<sub>n</sub>]]

=== Defining equations (equation residual expression) ===

The following statement can be used in '''pyDAE''' to create a residual expression of the ordinary equation:

<syntaxhighlight lang="python">
# Notation:
#  - V1, V3, V14 are ordinary variables
eq.Residal = V14.dt() + V1() / (V14() + 2.5) + sin(3.14 * V3())
</syntaxhighlight>

The above code translates into:

<math name="Eqn1" />

To define a residual expression of the distributed equation the next statements can be used:

<syntaxhighlight lang="python">
# Notation:
#  - V1, V3 and V14 are distributed variables on domains X and Y
eq = model.CreateEquation("MyEquation")
x = eq.DistributeOnDomain(X, eClosedClosed)
y = eq.DistributeOnDomain(Y, eOpenOpen)
eq.Residal = V14.dt(x,y) + V1(x,y) / ( V14(x,y) + 2.5) + sin(3.14 * V3(x,y) )
</syntaxhighlight>

The above code translates into:

<math name="Eqn2" />

=== Defining boundary conditions ===

Assume that we have a simple heat conduction through a very thin rectangular plate. At one side (Y = 0) we have a constant temperature (500 K) while at the opposide end we have a constant flux (1E6 W/m<sup>2</sup>). The problem can be defined by the following statements:

<syntaxhighlight lang="python">
# Notation:
#  - T is a variable distributed on domains X and Y
#  - ro, k, and cp are parameters
eq = model.CreateEquation("MyEquation")
x = eq.DistributeOnDomain(X, eClosedClosed)
y = eq.DistributeOnDomain(Y, eOpenOpen)
eq.Residual = ro() * cp() * T.dt(x,y) - k() * ( T.d2(X,x,y) + T.d2(Y,x,y) )
</syntaxhighlight>

We can note that the equation is defined on the domain Y, which is open on both ends. Now we have to specify the boundary conditions (2 additional equations). To do so, the following statements can be used:

<syntaxhighlight lang="python">
# "Left" boundary conditions:
lbc = model.CreateEquation("Left_BC")
x = lbc.DistributeOnDomain(X, eClosedClosed)
y = lbc.DistributeOnDomain(Y, eLowerBound)
lbc.Residal = T(x,y) - 500  # Constant temperature (500 K)
# "Right" boundary conditions:
rbc = model.CreateEquation("Right_BC")
x = rbc.DistributeOnDomain(X, eClosedClosed)
y = rbc.DistributeOnDomain(Y, eUpperBound)
rbc.Residal = - k() * T.d(Y,x,y) - 1E6  # Constant flux (1E6 W/m2)
</syntaxhighlight>

The above statements transform into:

<math name="Eqn3" />

and:

<math name="Eqn4" />

== Distribution Domains ==

A distribution domain is a general term used to define an array of different objects. Two types of domains exist: arrays and distributed domains. Array is a synonym for a simple vector of objects. Distributed domains are most frequently used to model a spatial distribution of parameters, variables and equations, but can be equally used to spatially distribute just any other object (even ports and models). Domains have the following properties:

* '''Name''': string (read-only)
* '''CanonicalName''': string (read-only)
* '''Description: '''string
* '''Type''': daeeDomainType (read-only; array or distributed)
* '''NumberOfIntervals''': unsigned integer (read-only)
* '''NumberOfPoints''': unsigned integer (read-only)
* '''Points''': list of floats
* '''LowerBound''': float (read-only)
* '''UpperBound''': float (read-only)

Distributed domains also have:

* '''DiscretizationMethod''': daeeDiscretizationMethod (read-only)<br />Currently backward finite difference ('''BFDM'''), forward finite difference ('''FFDM''') and center finite difference method ('''CFDM''') are implemented.
* '''DiscretizationOrder''': unsigned integer (read-only)<br />At the moment, only the 2<sup>nd</sup> order is supported.

There is a difference between number of points in domain and number of intervals. Number of intervals is a number of points (if it is array) or a number of finite difference elements (if it is distributed domain). Number of points is actual number of points in the domain. If it is array then they are equal. If it is distributed, and the scheme is one of finite differences for instance, it is equal to number of intervals + 1.

The most important functions are:

* '''CreateArray '''<span style="font-weight: normal">for creating a simple array</span>
* '''CreateDistributed''' for creating a distributed array
* '''operator []''' for getting a value of the point within domain for a given index (used only to construct equation residuals)
* '''Overloaded operator ()''' for creating '''daeIndexRange''' object (used only to construct equation residuals: as an argument of functions array, dt_array, d_array, d2_array)
* '''GetNumPyArray''' for getting the point values as a numpy one-dimensional array

The process of creating domains is two-fold: first you declare a domain in the model and then you define it (by assigning its properties) in the simulation.

=== Declaring a domain ===

The following statement is used to declare a domain:

<syntaxhighlight lang="python">
myDomain = daeDomain("myDomain", Parent, "Description")
</syntaxhighlight>

=== Defining a domain ===

The following statement is used to define a distributed domain:

<syntaxhighlight lang="python">
# Center finite diff, 2nd order, 10 elements, Bounds: 0.0 to 1.0
myDomain.CreateDistributed(eCFDM, 2, 10, 0.0,  1.0)
</syntaxhighlight>

while to define an array:

<syntaxhighlight lang="python">
# Array of 10 elements
myDomain.CreateArray(10)
</syntaxhighlight>

=== Non-uniform grids ===

In certain situations it is not desired to have a uniform distribution of the points within the given interval (LowerBound, UpperBound). In these cases, a non-uniform grid can be specified by the following statement:

<syntaxhighlight lang="python">
# First create a distributed domain
myDomain.CreateDistributed(eCFDM, 2, 10, 0.0,  1.0)
# The original 11 points are: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# If we are have a stiff profile at the beginning of the domain,
# then we can place more points there
myDomain.Points = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.60, 1.00]
</syntaxhighlight>

The comparison of the effects of uniform and non-uniform grids is given in '''Figure 2.''' (a simple heat conduction problem from the Tutorial3 has been served as a basis for comparison). Here we have the following cases:

* Blue line (normal case, 10 intervals): uniform grid - a very rough prediction
* Red line (10 intervals): more points at the beginning of the domain
* Black line (100 intervals): uniform-grid (closer to the analytical solution)

[[Image:NonUniformGrid.png|thumb|200px|Figure 2. Comparison of the effects of uniform and non-uniform grids on the numerical solution]]

We can clearly observe that we get much more precise results by using denser grid at the beginning of the domain.

=== Using domains ===

'''NOTE''': It is important to understand that all functions in this section are used ONLY to construct equation residuals and NOT to access the real (raw) data.

'''I)''' To get a value of the point within the domain at the given index we can use '''operator []'''. For instance if we want variable myVar to be equal to the sixth point (indexing in python and c/c++ starts at 0) in the domain myDomain, we can write:

<syntaxhighlight lang="python">
# Notation:
#  - eq is a daeEquation object
#  - myDomain is daeDomain object
#  - myVar is an daeVariable object
eq.Residual = myVar() - myDomain[5]
</syntaxhighlight>

The above statement translates into:

<math name="Eqn5" />

'''II)''' daeDomain '''operator ()''' returns the daeIndexRange object which is used as an argument of functions '''array''', '''dt_array''', '''d_array''' and '''d2_array''' in '''daeParameter''' and '''daeVariable''' classes to obtain an array of parameter/variable values, or an array of variable time (or partial) derivatives.

More details on parameter/variable arrays will be given in the following sections.

== Parameters ==

Parameters are time invariant quantities that will not change during simulation. Usually a good choice what should be a parameter is a physical constant, number of discretization points in a domain etc. Parameters have the following properties:

* '''Name''': string (read-only)
* '''CanonicalName''': string (read-only)
* '''Description: '''string
* '''Type''': daeeParameterType (read-only; real, integer, boolean)
* '''Domains''': daeDomain list

The most important functions are:

* Overloaded '''operator ()''' for getting the parameter value (used only to construct equation residuals)
* Overloaded function '''array''' for getting an array of values (used only to construct equation residuals as an argument of functions like sum, product etc)
* Overloaded functions '''SetValue''' and '''GetValue''' for access to the parameter's raw data
* '''GetNumPyArray''' for getting the values as a numpy multidimensional array

The process of creating parameters is two-fold: first you declare a parameter in the model and then you define it (by assigning its value) in the simulation.

=== Declaring a parameter ===

Parameters are declared in a model constructor ('''__init__''' function). An ordinary parameter can be declared by the following statement:

<syntaxhighlight lang="python">
myParam = daeParameter("myParam", eReal, Parent, "Description")
</syntaxhighlight>

Parameters can be distributed on domains. A distributed parameter can be declared by the next statement:

<syntaxhighlight lang="python">
myParam = daeParameter("myParam", eReal, Parent, "Description")
myParam.DistributeOnDomain(myDomain)
</syntaxhighlight>

Here, argument Parent can be either '''daeModel''' or '''daePort'''. Currently only eReal type is supported (others are ignored and used identically as the eReal type).

=== Defining a parameter ===

Parameters are defined in a simulation class ('''SetUpParametersAndDomains''' function). To set a value of an ordinary parameter:

<syntaxhighlight lang="python">
myParam.SetValue(1.0)
</syntaxhighlight>

To set a value of distributed parameters (one-dimensional for example):

<syntaxhighlight lang="python">
for i in range(0, myDomain.NumberOfPoints)
    myParam.SetValue(i, 1.0)
</syntaxhighlight>

=== Using parameters ===

'''NOTE:''' It is important to understand that all functions in this section are used ONLY to construct equation residuals and NOT to access the real (raw) data.

'''I)''' To get a value of the ordinary parameter the '''operator ()''' can be used. For instance, if we want variable myVar to be equal to the sum of the value of the parameter myParam and 15, we can write the following statement:

<syntaxhighlight lang="python"># Notation:
#  - eq is a daeEquation object
#  - myParam is an ordinary daeParameter object (not distributed)
#  - myVar is an ordinary daeVariable (not distributed)
eq.Residual = myVar() - myParam() - 15
</syntaxhighlight>

This code translates into:

<math name="Eqn6" />

'''II)''' To get a value of a distributed parameter we can again use '''operator ()'''. For instance, if we want distributed variable myVar to be equal to the sum of the value of the parameter myParam and 15 at each point of the domain myDomain, we need an equation for each point in the myDomain and we can write:

<syntaxhighlight lang="python">
# Notation:
#  - myDomain is daeDomain object
#  - n is the number of points in the myDomain
#  - eq is a daeEquation object distributed on the myDomain
#  - d is daeDEDI object (used to iterate through the domain points)
#  - myParam is daeParameter object distributed on the myDomain
#  - myVar is daeVariable object distributed on the myDomain
d = eq.DistributeOnDomain(myDomain, eClosedClosed)
eq.Residual = myVar(d) - myParam(d) - 15
</syntaxhighlight>

This code translates into n equations:

<math name="Eqn7" />

which is equivalent to writing (in pseudo-code):

<syntaxhighlight lang="python">
for d = 0 to n:
    myVar(d) = myParam(d) + 15
</syntaxhighlight>

which internally transforms into n separate equations.

Obviously, a parameter can be distributed on more than one domain. In that case we can use identical functions which accept two arguments:

<syntaxhighlight lang="python">
# Notation:
#  - myDomain1, myDomain2 are daeDomain objects
#  - n is the number of points in the myDomain1
#  - m is the number of points in the myDomain2
#  - eq is a daeEquation object distributed on the domains myDomain1 and myDomain2
#  - d is daeDEDI object (used to iterate through the domain points)
#  - myParam is daeParameter object distributed on the myDomain1 and myDomain2
#  - myVar is daeVariable object distributed on the myDomaina and myDomain2
d1 = eq.DistributeOnDomain(myDomain1, eClosedClosed)
d2 = eq.DistributeOnDomain(myDomain2, eClosedClosed)
eq.Residual = myVar(d1,d2) - myParam(d1,d2) - 15
</syntaxhighlight>

The above statement translates into:

<math name="Eqn8" />

'''III)''' To get an array of parameter values we can use the function '''array''' which returns the '''adouble_array''' object. Arrays of values can only be used in conjunction with mathematical functions that operate on '''adouble_array''' objects: '''sum''', '''product''', '''sqrt''', '''sin''', '''cos''', '''min''', '''max''', '''log''', '''log10''' etc. For instance, if we want variable myVar to be equal to the sum of values of the parameter myParam for all points in the domain myDomain, we can use the function '''sum''' (defined in '''daeModel''' class) which accepts results of the '''array''' function (defined in '''daeParameter''' class). Arguments for the array function are '''daeIndexRange''' objects obtained by the call to '''daeDomain's operator ()'''. Thus, we can write the following statement:

<syntaxhighlight lang="python">
# Notation:
#  - myDomain is daeDomain object
#  - n is the number of points in the domain myDomain
#  - eq is daeEquation object
#  - myVar is daeVariable object
#  - myParam is daeParameter object distributed on the myDomain
eq.Residual = myVar() - sum( myParam.array( myDomain() ) )
</syntaxhighlight>

This code translates into:

<math name="Eqn10" />

The above example could be also written in the following form:

<syntaxhighlight lang="python">
# points_range is daeDomainRange object
points_range = daeDomainRange(myDomain)
# arr is adouble_array object
arr = myVar2.array(points_range)
# Finally:
eq.Residual = myVar() - sum(arr)
</syntaxhighlight>

On the other hand, if we want variable myVar to be equal to the sum of values of the parameter myParam only for certain points in the myDomain, there are two ways to do it:

<syntaxhighlight lang="python">
# Notation:
#  - myDomain is daeDomain object
#  - n is the number of points in the domain myDomain
#  - eq is a daeEquation object
#  - myVar is an ordinary daeVariable object
#  - myParam is a daeParameter object distributed on the myDomain
# 1) For a given array of points; the points must be in the range [0,n-1]
eq.Residual = myVar() - sum( myParam.array( myDomain( [0, 5, 12] ) ) )
# 2) For a given slice of points in the domain;
#    slices are defined by 3 arguments: start_index, end_index, step
#    in this example: start_index = 1
#                     end_index = 10
#                     step = 2
eq.Residual = myVar() - sum( myParam.array( myDomain(1, 10, 2) ) )
</syntaxhighlight>

The code sample 1) translates into:

<math name="Eqn11" />

The code sample 2) translates into:

<math name="Eqn12" />

'''NOTE: '''One may argue that the function '''array''' calls can be somewhat simpler and directly accept python lists or slices as its arguments. For instance it would be possible to write:

<syntaxhighlight lang="python">
eq.Residual = myVar() - sum( myParam.array( [0, 1, 3] ) )
</syntaxhighlight>

or:

<syntaxhighlight lang="python">
eq.Residual = myVar() - sum( myParam.array( slice(1,10,2) ) )
</syntaxhighlight>

However, that would be more error prone since it does not check whether a valid domain is used for that index and whether specified indexes lay within the domain bounds (which should be done by the user).

== Variable Types ==

Variable types are used to describe variables. The most important properties are:

* '''Name''': string
* '''Units''': string
* '''LowerBound''': float
* '''UpperBound''': float
* '''InitialGuess''': float
* '''AbsoluteTolerance''': float

Declaration of variable types is usually done outside of model definitions (as global variables).

=== Declaring a variable type ===

To declare a variable type:

<syntaxhighlight lang="python">
# Temperature, units: Kelvin, limits: 100 - 1000K, Def.value: 273K, Abs.Tol: 1E-5
typeTemperature = daeVariableType("Temperature", "K", 100, 1000, 273, 1E-5)
</syntaxhighlight>

== Variables ==

Variables are time variant quantities (state variables). The most important properties are:

* '''Name''': string (read-only)
* '''CanonicalName''': string (read-only)
* '''Description: '''string
* '''Type''': daeVariableType object
* '''Domains''': daeDomain list
* '''ReportingOn''': boolean

The most important functions are:

* Overloaded '''operator ()''' for getting the variable value/time derivative/partial derivative (used only to construct equation residuals)
* Overloaded functions '''array''', '''dt_array''', '''d_array''', and '''d2_array''' for getting an array of values/time derivatives/partial derivatives (used only to construct equation residuals as an argument of functions like '''sum''', '''product''' etc)
* Overloaded functions '''AssignValue''' to fix degrees of freedom of the model
* Overloaded functions '''ReAssignValue''' to change a value of a fixed variable
* Overloaded functions '''SetValue''' and '''GetValue''' for access to the variable's raw data
* Overloaded function '''SetInitialGuess''' for setting an initial guess of the variable
* Overloaded function '''SetInitialCondition''' for setting an initial condition of the variable
* Overloaded function '''ReSetInitialCondition''' for re-setting an initial condition of the variable
* Overloaded function '''SetAbsoluteTolerances''' for setting an absolute tolerance of the variable
* '''GetNumPyArray''' for getting the values as a numpy multidimensional array

The process of creating variables is two-fold: first you declare a variable in the model and then you define it (by assigning its value) in the simulation.

=== Declaring a variable ===

Variables are declared in a model constructor ('''__init__''' function). To declare an ordinary variable:

<syntaxhighlight lang="python">
myVar = daeVariable("myVar", variableType, Parent, "Description")
</syntaxhighlight>

Variables can be distributed on domains. To declare a distributed variable:

<syntaxhighlight lang="python">
myVar = daeVariable("myVar", variableType, Parent, "Description")
myVar.DistributeOnDomain(myDomain)
</syntaxhighlight>

Here, argument Parent can be either '''daeModel''' or '''daePort'''.

=== Assigning a variable value (setting the degrees of freedom of a model) ===

Degrees of freedom can be fixed in a simulation class in '''SetUpVariables''' function by assigning the value of a variable. Assigning the value of an ordinary variables can be done by the following statement:

<syntaxhighlight lang="python">
myVar.AssignValue(1.0)
</syntaxhighlight>

while the assigning the value of a distributed variable (one-dimensional for example) can be done by the next statement:

<syntaxhighlight lang="python">
for i in range(myDomain.NumberOfPoints)
    myVar.AssignValue(i, 1.0)
</syntaxhighlight>

=== Re-assigning a variable value ===

Sometime during a simulation it is necessary to re-assign the variable value. This can be done by the following statement:

<syntaxhighlight lang="python">
myVar.ReAssignValue(1.0)
... re-assign or re-initialize some other variables too (optional)
simulation.ReInitialize()
</syntaxhighlight>

'''NOTE:''' After re-assigning or after re-initializing variable(s) the function '''ReInitialize'''in the simulation object '''MUST''' be called before continuing with the simulation!

=== Accessing a variable raw data ===

Functions '''GetValue/SetValue''' access the variable raw data and should be used directly with a great care!!!

'''NOTE:''' ONLY USE THIS FUNCTION IF YOU EXACTLY KNOW WHAT ARE YOU DOING AND THE POSSIBLE IMPLICATIONS!!

Setting the value of ordinary variables can be done by the following statement::

<syntaxhighlight lang="python">
myVar.SetValue(1.0)
</syntaxhighlight>

while setting the value of a distributed variable can be done by:

<syntaxhighlight lang="python">
for i in range(myDomain.NumberOfPoints)
    myVar.SetValue(i, 1.0)
</syntaxhighlight>

=== Setting an initial guess ===

Initial guesses can be set in a simulation class in '''SetUpVariables''' function. An initial guess of an ordinary variable can be set by the following statement:

<syntaxhighlight lang="python">
myVar.SetInitialGuess(1.0)
</syntaxhighlight>

while the initial guess of a distributed variable by:

<syntaxhighlight lang="python">
for i in range(myDomain.NumberOfPoints)
    myVar.SetInitialGuess(i, 1.0)
</syntaxhighlight>

Setting an initial guess of a distributed variable to a single value for all points in all domains can be done by the following statement:

<syntaxhighlight lang="python">
myVar.SetInitialGuesses(1.0)
</syntaxhighlight>

=== Setting an initial condition ===

Initial conditions can be set in a simulation class in '''SetUpVariables''' function. In '''DAE Tools''' there are two modes. You can set either set an algebraic value or use the eSteadyState flag. This is controlled by the property '''InitialConditionMode''' in the simulation class (can be eAlgebraicValuesProvided or eSteadyState). '''However, only the algebraic parts can be set at the moment'''. An initial condition of an ordinary variable can be set by the following statement:

<syntaxhighlight lang="python">
myVar.SetInitialCondition(1.0)
</syntaxhighlight>

while the initial guess of a distributed variable by:

<syntaxhighlight lang="python">
for i in range(myDomain.NumberOfPoints)
    myVar.SetInitialCondition(i, 1.0)
</syntaxhighlight>

=== Re-setting an initial condition ===

Sometime during a simulation it is necessary to re-initialize the variable value. This can be done by the following statement:

<syntaxhighlight lang="python">
myVar.ReSetInitialCondition(1.0)
... re-assign or re-initialize some other variables too (optional)
simulation.ReInitialize()
</syntaxhighlight>

'''NOTE:''' After re-assigning or after re-initializing the variable values the function '''ReInitialize''' in the simulation object '''MUST''' be called before continuing with the simulation!

=== Setting an absolute tolerance ===

Absolute tolerances can be set in a simulation class in '''SetUpVariables''' function by the following statement:

<syntaxhighlight lang="python">
myVar.SetAbsoluteTolerances(1E-5)
</syntaxhighlight>

=== Getting a variable value ===

'''NOTE:''' It is important to understand that all functions in this and all following sections are used '''ONLY''' to construct equation residuals and '''NOT''' no to access the real (raw) data.

For the examples how to get a variable value see the sub-sections '''I - III''' in the section [[pydae_user_guide#Using parameters|Using parameters]]. '''Operator ()''' in '''daeVariable''' class behaves in the same way as the '''operator ()''' in '''daeParameter''' class.

=== Getting a variable time derivative ===

'''I)''' To get a time derivative of the ordinary variable the function '''dt''' can be used. For example, if we want a time derivative of the variable myVar to be equal to some constant, let's say 1.0, we can write:

<syntaxhighlight lang="python">
# Notation:
#  - eq is a daeEquation object
#  - myVar is an ordinary daeVariable (not distributed)
eq.Residual = myVar.dt() - 1
</syntaxhighlight>

The above statement translates into:

<math name="Eqn13" />

'''II)''' Getting a time derivative of distributed variables is analogous to getting a parameter value (see the sub-section '''II '''in the section [[pydae_user_guide#Using Parameters]). The function '''dt''' accepts the same arguments and it is called in the same way as the '''operator ()''' in '''daeParameter''' class.

<br />'''III)''' Getting an array of time derivatives of distributed variables is analogous to getting an array of parameter values (see the sub-section '''III '''in the section [[pydae_user_guide#Using Parameters]). The function '''dt_array''' accepts the same arguments and it is called in the same way as the function '''array''' in '''daeParameter''' class.

<br />'''Note:''' Sometime a derivative of an expression is needed. In that case the function '''dt''' from the daeModel class can be used.

<syntaxhighlight lang="python"># Notation:
#  - eq is a daeEquation object
#  - myVar1 is an ordinary daeVariable (not distributed)
#  - myVar2 is an ordinary daeVariable (not distributed)
eq.Residual = model.dt( myVar1() + myVar2() )
</syntaxhighlight>

=== Getting a variable partial derivative ===

It is possible to get a partial derivative only of the distributed variables and only for a domain which is distributed (not an ordinary array).

'''I)''' To get a partial derivative of the variable per some domain, we can use functions '''d''' or '''d2''' (the function d calculates a partial derivative of the first order while the function '''d2''' calculates a partial derivative of the second order). For instance, if we want a first order partial derivative of the variable '''myVar''' to be equal to some constant, let's say 1.0, we can write:

<syntaxhighlight lang="python"># Notation:
#  - myDomain is daeDomain object
#  - n is the number of points in the myDomain
#  - eq is a daeEquation object distributed on the myDomain
#  - d is daeDEDI object (used to iterate through the domain points)
#  - myVar is daeVariable object distributed on the myDomain
d = eq.DistributeOnDomain(myDomain, eOpenOpen)
eq.Residual = myVar.d(myDomain, d) - 1
</syntaxhighlight>

This code translates into:

<math name="Eqn14" />

Please note that the function myEquation is not distributed on the whole myDomain (it does not include the bounds). <br />In the case we want to get a partial derivative of the second order we can use the function '''d2''' which is called in the same fashion as the function '''d''':

<syntaxhighlight lang="python">
d = eq.DistributeOnDomain(myDomain, eOpenOpen)
eq.Residual = myVar.d2(myDomain, d) - 1
</syntaxhighlight>

which translates into:

<math name="Eqn15" />

'''II)''' To get an array of partial derivatives we can use functions '''d_array''' and '''d2_array''' which return the '''adouble_array''' object (the function '''d_array''' returns an array of partial derivatives of the first order while the function '''d2_array''' returns an array of partial derivatives of the second order). Again these arrays can only be used in conjunction with mathematical functions that operate on '''adouble_array''' objects: '''sum''', '''product''', etc. For instance, if we want variable myVar to be equal to the minimal value in the array of partial derivatives of the variable myVar2 for all points in the domain myDomain, we can use the function '''min''' (defined in '''daeModel''' class) which accepts arguments of type '''adouble_array'''. Arguments for the d_array function are '''daeIndexRange '''objects obtained by the call to '''daeDomain''' '''operator ()'''. In this particular example we need a minimum among partial derivatives for the specified points (0, 1, and 3). Thus, we can write:

<syntaxhighlight lang="python">
# Notation:
#  - myDomain is daeDomain object
#  - n is the number of points in the domain myDomain
#  - eq is daeEquation object
#  - myVar is daeVariable object
#  - myVar2 is daeVariable object distributed on myDomain
eq.Residual = myVar() - min( myVar2.d_array(myDomain, myDomain( [0, 1, 3] ) )
</syntaxhighlight>

The above code translates into:

<math name="Eqn16" />

'''Note:''' Sometime a partial derivative of an expression is needed. In that case the function '''d''' from the daeModel class can be used.

<syntaxhighlight lang="python">
# Notation:
#  - myDomain is daeDomain object
#  - eq is a daeEquation object
#  - myVar1 is an ordinary daeVariable (not distributed)
#  - myVar2 is an ordinary daeVariable (not distributed)
eq.Residual = model.d( myVar1() + myVar2(), myDomain )
</syntaxhighlight>

== Ports ==

Ports are used to connect two instances of models. Like models, ports can contain domains, parameters and variables. The most important properties are:

* '''Name''': string (read-only)
* '''CanonicalName''': string (read-only)
* '''Description: '''string
* '''Type''': daeePortType (inlet, outlet, inlet-outlet)
* '''Domains''': daeDomain list
* '''Parameters''': daeParameter list
* '''Variables''': daeVariable list

The most important functions are:

* '''SetReportingOn'''

= Activity module =

= DataReporting module =

= Solver module =


.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:

    