*****************
DAE Tools Project
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

.. toctree::
   :hidden:
   :maxdepth: 1

   news
   contact
   docs/index.rst


.. include:: docs/introduction.rst
   :start-after: begin-command
   :end-before: end-command

More information about DAE Tools can be found in the :doc:`docs/introduction` section
of the :doc:`docs/index`.

Download
========

The current release is 1.4.0.

Installation files can be found in the SourceForge website
`download section <https://sourceforge.net/projects/daetools/files/1.4.0>`_,
and the source code in the SourceForge
`subversion repository <https://sourceforge.net/p/daetools/code/HEAD/tree>`_.

More information on system requirements, downloading and installing **DAE Tools**
can be found in :doc:`docs/getting_daetools`.

News
====
The new 1.4.0 version is released on 22 December 2014. It contains a large number of important features and bug fixes.

The most important new features:
    
-  Code generators for `Modelica`_, `gPROMS <http://www.psenterprise.com/gproms.html>`_ and `c99`_.
   They can be found in daetools/code\_generators. Almost all features
   available in daetools are supported except event ports, user defined actions,
   external functions and finite element objects whose equations need to be updated during
   a simulation.
-  Support for simulation in other simulators using standard interfaces for Co-Simulation:
   `Functional Mockup Interface <https://www.fmi-standard.org>`_, `Matlab MEX-functions <http://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html>`_
   and `Simulink S-functions <http://www.mathworks.com/help/simulink/sfg/what-is-an-s-function.html>`_.
-  DAE Tools objects such as adouble can be used as NumPy native data type.
   The most of the NuPy and SciPy functions are supported.
-  New data reporters that export the simulation results to various file formats (MS Excel, hdf5, xml, json) and
   to Pandas data sets.
-  Added new math functions: Sinh, Cosh, Tanh, ASinh, ACosh, ATanh, ATan2 and Erf to adouble/adouble_array.
-  Added `Pardiso <http://www.pardiso-project.org>`_ linear solver.
-  Added SimulationExplorer GUI that lists all domains, parameters, initial conditions, degrees of freedom
   and state transition networks.
-  Simulations can export the initialization values to JSON format and initialize using a JSON string.
   daetools.cfg config file is now in JSON format.
-  Domains and parameters can now be propagated through the whole model hierarchy (daeModel.PropagateDomain() and
   daeModel.PropagateParameter()). All domains/parameters with the same name will have identical properties.
-  daeVariable functions SetValues, SetInitialConditions, AssignValues etc. accept NumPy arrays as arguments.
   Now, values and initial conditions can be set using numpy float or quantity arrays.
-  All equation can generate Jacobian expressions by setting daeEquation.BuildJacobianExpressions to True.
   This is useful when an expression is huge and contains a large number of variables. Calculation of a Jacobian
   for such equation would take a very long time. Generation of Jacobian expressions will increase the memory
   requirements but may tremendously decrease the computational time.
-  Numerical simulation of partial differential equations on adaptive unstructured grids using Finite Elements Method.
   `deal.II`_ library is used for low-level tasks such as mesh loading/processing
   and assembly of the system stiffness/mass matrices and the system load vector.
   `deal.II`_ structures are then used to
   generate daetools equations which are solved together with the rest of the model
   equations. All details about the mesh, basis functions, quadrature rules, refinement
   etc. are handled by the `deal.II`_ library. The advantage of this concept is that the
   generated equations (linear, nonlinear or differential - depending on the class of the system)
   can be coupled with other FE-unrelated equations in a daetools model and solved
   together by daetools solvers; system discontinuities can be handled
   as usual in daetools; modelled processes can be optimized, etc.

.. _libMesh: http://libmesh.sourceforge.net/index.php
.. _deal.II: http://dealii.org
.. _NineML: http://software.incf.org/software/nineml
.. _More details: News
.. _Modelica: http://www.modelica.org
.. _FMI: https://www.fmi-standard.org
.. _Simulink: http://www.mathworks.com/products/simulink
.. _c99: https://en.wikipedia.org/wiki/C99
  
Full list of news can be found here: :doc:`news`

Contact
=======
The author and the main developer is dr. Dragan Nikolic |LinkedIn|

Please send your comments and questions to: dnikolic at daetools dot com.

More information about the author can be found in :doc:`contact`.

.. |LinkedIn| image:: http://www.linkedin.com/img/webpromo/btn_liprofile_blue_80x15.png
                :width: 80px
                :height: 15px
                :target: http://rs.linkedin.com/in/dragannikolic
                :alt: View Dragan Nikolić's profile on LinkedIn

Documentation
=============

Detailed information about using **DAE Tools**, presentations, API reference and tutorials
can be found in :doc:`docs/index`.
