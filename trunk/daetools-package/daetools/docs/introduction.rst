************
Introduction
************
..
    Copyright (C) Dragan Nikolic
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

About DAE Tools
===============

.. begin-command

**DAE Tools** is a cross-platform equation-based object-oriented modelling, simulation
and optimisation software. It is not a modelling language nor a collection of numerical
libraries but rather a higher level structure – an architectural design of interdependent
software components providing an API for:
   
* Model development/specification
* Activities on developed models, such as simulation, optimisation, and parameter estimation
* Processing of the results, such as plotting and exporting to various file formats
* Report generation
* Code generation, co-simulation and model exchange

The following class of problems can be solved by **DAE Tools**:

* Initial value problems of implicit form, described by a system of linear, non-linear, and (partial-)differential
  algebraic equations
* Index-1 DAE systems
* With lumped or distributed parameters: Finite Difference or Finite Elements Methods (still experimental)
* Steady-state or dynamic
* Continuous with some elements of event-driven systems (discontinuous equations, state transition networks
  and discrete events)

Type of activities that can be performed on models developed in **DAE Tools**:

* Simulation (steady-state or dynamic, with simple or complex schedules)

* Optimisation (NLP and MINLP problems)

* Parameter estimation

* Generation of model reports (in XML + MathML format with XSL transformations for XHTML code generation)

* Code generation for other modelling or general-purpose programming languages

  * `Modelica <http://www.modelica.org>`_
  * `gPROMS <http://www.psenterprise.com/gproms.html>`_
  * `Standard ISO C (c99) <http://www.open-std.org/jtc1/sc22/wg14/www/standards>`_
  * C++/MPI

* Simulation in other simulators using standard co-simulation interfaces

  * `Functional Mockup Interface (FMI) for Co-Simulation <https://www.fmi-standard.org>`_
  * `Matlab MEX-functions <http://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html>`_
  * `Simulink user-defined S-functions <http://www.mathworks.com/help/simulink/sfg/what-is-an-s-function.html>`_

* Export of the simulation results to various file formats:

  * `Matlab MAT (.mat) <http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf>`_
  * `Microsoft Excel (.xls) <http://office.microsoft.com/en-gb/excel>`_
  * `JSON (.json) <http://www.json.org>`_
  * `XML (.xml) <http://www.w3.org/XML>`_
  * `Hierarchical Data Format (.hdf5) <http://www.hdfgroup.org/HDF5>`_
  * `Pandas (Python Data Analysis) data sets <http://pandas.pydata.org>`_
  * `VTK (.vtk) <http://www.vtk.org>`_

**DAE Tools** run on all major operating systems (Windows, GNU Linux and Mac OS X)
and architectures (x86, x86_64, arm).

It is free to use, since it is `free software <http://www.gnu.org/>`_ and released
under the `GNU General Public Licence <http://www.gnu.org/licenses/licenses.html#GPL>`_.

**DAE Tools** is initially developed to model and simulate processes in chemical process industry
(mass, heat and momentum transfers, chemical reactions, separation processes, thermodynamics).
However, **DAE Tools** can be used to develop high-accuracy models of (in general) many different
kind of processes/phenomena, simulate/optimise them, visualise and analyse the results.

The following approaches/paradigms are adopted in **DAE Tools**:

* A hybrid approach between general-purpose programming languages (such as c++ and Python) and
  domain-specific modelling languages (such as `Modelica <http://www.modelica.org>`_,
  `gPROMS <http://www.psenterprise.com/gproms>`_, `Ascend <http://ascend4.org>`_ etc.)
  (more information: :ref:`hybrid_approach`).
  
* An object-oriented approach to process modelling (more information: :ref:`object_oriented_approach`).

* An Equation-Oriented (acausal) approach where all model variables and equations are generated and
  gathered together and solved simultaneously using a suitable mathematical algorithm
  (more information: :ref:`equation_oriented_approach`).
  
* Separation of the model definition from the activities that can be carried out on that model.
  The structure of the model (parameters, variables, equations, state transition networks etc.)
  is given in the model class while the runtime information in the simulation class. This way,
  based on a single model definition, one or more different simulation/optimisation scenarios
  can be defined.

* Core libraries are written in standard c++, however `Python <http://www.python.org>`_ is used as
  the main modelling language (more information: :ref:`python_programming_language`).

.. end-command

All core libraries are written in standard c++. It is highly portable - it runs on all
major operating systems (GNU/Linux, MacOS, Windows) and all platforms with a decent c++ compiler,
Boost and standard c/c++ libraries (by now it is tested on 32/64 bit x86 and ARM architectures
making it suitable for use in embedded systems). Models can be developed in Python
(**pyDAE** module) or c++ (**cDAE** module), compiled into an independent
executable and deployed without a need for any run time libraries.

**DAE Tools** support a large number of solvers. Currently `Sundials IDAS <https://computation.llnl.gov/casc/sundials/main.html>`_
solver is used to solve DAE systems and calculate sensitivities, while `BONMIN <https://projects.coin-or.org/Bonmin>`_,
`IPOPT <https://projects.coin-or.org/IPOPT>`_, and `NLOPT <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_
solvers are used to solve NLP/MINLP problems.
**DAE Tools** support direct dense and sparse matrix linear solvers (sequential and multi-threaded versions)
at the moment. In addition to the built-in Sundials linear solvers, several third party libraries are interfaced:
`SuperLU/SuperLU_MT <http://crd.lbl.gov/~xiaoye/SuperLU/index.html>`_,
`Pardiso <http://www.pardiso-project.org>`_,
`Intel Pardiso <http://software.intel.com/en-us/intel-mkl>`_,
`Trilinos Amesos <http://trilinos.sandia.gov/packages/amesos/>`_ (KLU, Umfpack, SuperLU, Lapack),
and `Trilinos AztecOO <http://trilinos.sandia.gov/packages/aztecoo>`_ (with built-in, Ifpack or ML preconditioners)
which can take advantage of multi-core/cpu computers.
Linear solvers that exploit general-purpose graphics processing
units (`GPGPU <http://en.wikipedia.org/wiki/GPGPU>`_, such as `NVidia CUDA <http://www.nvidia.com/object/cuda_home_new.html>`_)
are also available (`CUSP <http://code.google.com/p/cusp-library>`_) but in an early development stage.

Licence
=======

**DAE Tools** is `free software <http://www.gnu.org/>`_ and you can redistribute it and/or modify it under the terms of
the `GNU General Public Licence <http://www.gnu.org/licenses/licenses.html#GPL>`_ version 3 as published by
the Free Software Foundation (`GNU philosophy <http://www.gnu.org/philosophy/free-sw.html>`_).

How to cite
===========

If you use DAE Tools in your work then please cite the following article:
  Nikolić DD. (2016) *DAE Tools: equation-based object-oriented modelling, simulation and optimisation software*.
  **PeerJ Computer Science** 2:e54 `<https://doi.org/10.7717/peerj-cs.54>`_.

BibTeX: `daetools-peerj.bib <http://www.daetools.com/docs/presentations/daetools-peerj-cs-54.bib>`_.

History
=======

**"Necessity, who is the mother of invention"**
    *Plato, Greek author & philosopher (427 BC - 347 BC), The Republic*

**"Every good work of software starts by scratching a developer's personal itch"**
    *Eric S. Raymond, hacker, The Cathedral and the Bazaar, 1997*

The latter cannot be more true [#EricRaymond]_.
The early ideas of starting a project like this go back into 2007. At that time I have been working on my
PhD thesis using one of commercially available process modelling software. It was everything nice and well
until I discovered some annoying bugs and lack of certain highly appreciated features. The developers of that
proprietary program (as it is a case with all proprietary computer programs) had their own agenda fixing only
what they wanted to fix and introducing new features that they anticipated. Although I was able to improve
the code and introduce certain features which will help (not only) me - I was helpless. The source code was
not available and nobody will ever consider giving it to me to create patches with bugs fixes/new features.
Not even if I swear on the holy (c++) bible!!

Very soon the contours of a new process modelling software slowly began to form. It took me a while until
I made a definite plan and initial features, and I had to abandon a couple of initial versions...

**"Plan to throw one away; you will, anyhow"**
    *Eric S. Raymond, hacker, The Cathedral and the Bazaar, 1997*

Damn you Eric Raymond, interfering with my business again! :-)
The new project was officially born early next year - 2008.

.. [#EricRaymond] However, I do not agree with Eric Raymond and the Open Source Iniative views - they miss the point IMO, but let us leave it beside at the moment.

Acknowledgements
================

DAE Tools use the following third party free software libraries (GNU GPL, GNU LGPL, CPL, EPL, BSD or some other type of free/permissive/copy-left licences):

* Sundials IDAS: `<https://computation.llnl.gov/casc/sundials/main.html>`_
* Boost: `<http://www.boost.org>`_
* ADOL-C: `<https://projects.coin-or.org/ADOL-C>`_
* Qt and pyQt4: `<http://qt.nokia.com>`_, `<http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_
* Numpy: `<http://numpy.scipy.org http://numpy.scipy.org>`_
* Scipy: `<http://www.scipy.org>`_
* Blas/Lapack/CLapack: `<http://www.netlib.org>`_
* Minpack: `<http://www.netlib.org/minpack>`_
* Atlas: `<http://math-atlas.sourceforge.net>`_
* Trilinos Amesos: `<http://trilinos.sandia.gov/packages/amesos>`_
* Trilinos AztecOO: `<http://trilinos.sandia.gov/packages/aztecoo>`_
* SuperLU/SuperLU_MT: `<http://crd.lbl.gov/~xiaoye/SuperLU/index.html>`_
* Umfpack: `<http://www.cise.ufl.edu/research/sparse/umfpack>`_
* MUMPS:  `<http://graal.ens-lyon.fr/MUMPS>`_
* IPOPT: `<https://projects.coin-or.org/Ipopt>`_
* Bonmin: `<https://projects.coin-or.org/Bonmin>`_
* NLOPT: `<http://ab-initio.mit.edu/wiki/index.php/NLopt>`_
* CUSP: `<http://code.google.com/p/cusp-library>`_

**DAE Tools** can optionally use the following proprietary software libraries:

* Pardiso linear solver (pyPardiso module): `<http://www.pardiso-project.org>`_
* Intel Pardiso linear solver (pyIntelPardiso module): `<http://software.intel.com/en-us/articles/intel-mkl>`_

Please see the corresponding websites for more details about the licences.
