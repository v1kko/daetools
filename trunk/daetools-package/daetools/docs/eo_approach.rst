**************************
Equation-Oriented approach
**************************
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

In general, three approaches to process modelling exist ([#Morton2003]_):

* Sequential Modular (**SeqM**) approach
* Simultaneous Modular (**SimM**) approach
* Equation-Oriented (**EO**) approach

The pros & cons of the first two approaches are extensively studied in the literature. Under the **EO** approach we generate
and gather together all equations and variables which constitute the model representing the process. The equations are solved
simultaneously using a suitable mathematical algorithm (Morton, 2003 [#Morton2003]_). Equation-oriented simulation requires
simultaneous solution of a set of differential algebraic equations (**DAE**) which itself requires a solution of a set of
nonlinear algebraic equations (**NLAE**) and linear algebraic equations (**LAE**). The Newton's method or some variant of it
is almost always used to solve problems described by NLAEs. A brief history of Equation-Oriented solvers and comparison of
**SeqM** and **EO** approaches as well as descriptions of the simultaneous modular and equation-oriented methods can be found
in Morton, 2003 ([#Morton2003]_). Also a good overview of the equation-oriented approach and its application in
`gPROMS <http://www.psenterprise.com/gproms>`_ is given by Barton & Pantelides ([#Pantelides1]_, [#Pantelides2]_, [#Pantelides3]_).

**DAE Tools** use the Equation-Oriented approach to process modelling, and the following types of processes can be modelled:

* Lumped and distributed
* Steady-state and dynamic

Problems can be formulated as linear, non-linear, and (partial) differential algebraic systems (of index 1).
The most common problems are initial value problems of implicit form. Equations can be ordinary or discontinuous,
where discontinuities are automatically handled by the framework. A good overview of discontinuous equations and
a procedure for location of equation discontinuities is given by Park & Barton ([#ParkBarton]_)
and in `Sundials IDA <https://computation.llnl.gov/casc/sundials/documentation/ida_guide/node3.html#SECTION00330000000000000000 documentation>`_
(used in DAE Tools).

The main characteristics of the Equation-oriented (acausal) approach:

* Equations are given in an implicit form (as a residual):

  .. math::

    F(\dot {x}, x, y, p) = 0

  where :math:`x` and :math:`\dot {x}` are state variables and their derivatives,
  :math:`y` are degrees of freedom and :math:`p` are parameters.

* Input-Output causality is not fixed
 The benefits are:
  * Increased model re-use
  * Support for different simulation scenarios (based on a single model) by specifying
    different degrees of freedom. For instance, an equation given in the following form:

    .. math::
      x_1 + x_2 + x_3 = 0

    can be used to determine either ``x1``, ``x2`` or ``x3`` depending on what combination
    of variables is known:

    .. math::
      x_1 = -x_2 - x_3 \newline
      
      \vee \newline

      x_2 = -x_1 - x_3 \newline

      \vee \newline

      x_3 = -x_1 - x_2

      
.. rubric:: Footnotes
      
.. [#Morton2003]  Morton, W., Equation-Oriented Simulation and Optimization. *Proc. Indian Natl. Sci. Acad.* 2003, 317-357.
.. [#Pantelides1] Pantelides, C. C., and P. I. Barton, Equation-oriented dynamic simulation current status and future perspectives, *Computers & Chemical Engineering*, vol. 17, no. Supplement 1, pp. 263 - 285, 1993.
.. [#Pantelides2] Barton, P. I., and C. C. Pantelides, gPROMS - a Combined Discrete/Continuous Modelling Environment for Chemical Processing Systems, *Simulation Series*, vol. 25, no. 3, pp. 25-34, 1993.
.. [#Pantelides3] Barton, P. I., and C. C. Pantelides, Modeling of combined discrete/continuous processes", *AIChE Journal*, vol. 40, pp. 966-979, 1994.
.. [#ParkBarton]  Park, T., and P. I. Barton, State event location in differential-algebraic models", *ACM Transactions on Modeling and Computer Simulation*, vol. 6, no. 2, New York, NY, USA, ACM, pp. 137-165, 1996.



.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
