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
   superlu_cuda


.. include:: docs/introduction.rst
   :start-after: begin-command
   :end-before: end-command

More information about DAE Tools can be found in the :doc:`docs/introduction` section
of the :doc:`docs/index`.

Download
========

The current and a very outdated version is 1.2.1. A new release 1.3.0 is in the beta 3 state.

Installation files can be found on the SourceForge website 
`download section <https://sourceforge.net/projects/daetools/files>`_,
and the source code in the SourceForge
`subversion repository <https://sourceforge.net/p/daetools/code/HEAD/tree>`_.

More information on system requirements, downloading and installing **DAE Tools**
can be found in :doc:`docs/getting_daetools`.

News
====
The third beta is released. It contains many important bug fixes, memory leaks fixes and new features.

A new version 1.3.0 will bring the following new features and
improvements:

-  Numerical simulation of partial differential equations on adaptive
   unstructured grids using Finite Elements Method.
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

-  Code generators for `Modelica`_ (whole simulation or just
   selected models/ports), `FMI`_ and `c99`_. They are already functional
   (available only in python) and located in the folder
   daetools/code\_generators (with some tests). Almost all features
   available in daetools can be exported to Modelica and c,
   except event ports, user defined actions, external functions and 
   finite element objects whose equations need to be updated during 
   a simulation. The existing model analyzer make code generation rather simple 
   (as long as the very basic modelling concepts such as parameters, variables
   and discontinuous equations are supported in the target language).
-  Support for Functional Mock-up Interface for Model Exchange and
   Co-Simulation (FMI) `FMI`_.

A bug fix in 3D plot when detecting free domains (by Caleb Hattingh).

**DAE Tools** v1.2.1 is released on 14 June 2012. It brings several new
features and improvements (:ref:`v1.2.1 <v1_2_1>`). The most important are:

- Integration speed improvements (more than an order of magnitude, in
  some cases); no need for a memory copy from/to the DAE solver, a
  better integration step control and an option to avoid sparse matrix
  re-creations after a discontinuity
- Added support for units; variables, parameters, domains points must
  have a numerical value in terms of a unit of measurement (quantity)
  and units-consistency is strictly enforced (although it can be
  switched off in the daetools.cfg config file); all constants in
  equations must be dimensional and assigned units
- A basic support for external functions that can handle and evaluate
  functions in external libraries (the goal is to support certain software
  components such as thermodynamic property packages)
- A new type of 2D plots: Animated2D plot
- Equations can have an optional scaling
- Improved data reporting speed and changes in data reporting during an optimization
- New distribution format (python disutils)
- Mac OSX port
- c++ (cDAE) tutorials
- Support for the information about the progress of a simulation/optimization activity
- Other small improvements and minor bugs fixes

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

SuperLU_CUDA
============

This is a new DAE Tools subproject with the aim to provide
a direct sparse linear equation solver which works with NVidia CUDA GPUs.
More information: :doc:`superlu_cuda`.

