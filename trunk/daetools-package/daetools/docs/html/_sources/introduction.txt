************
Introduction
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

==================
What is DAE Tools?
==================

**DAE Tools** is a collection of software tools for modelling, simulation and optimization
of real-world processes. Process modelling and simulation can be defined as theoretical
concepts and computational methods that describe, represent in a mathematical form and simulate
the functioning of real-world processes. **DAE Tools** is initially developed to model and
simulate processes in chemical process industry (mass, heat and momentum transfers, chemical
reactions, separation processes, thermodynamics). However, **DAE Tools** can help you develop
high-accuracy models of (in general) many different kind of processes/phenomena, simulate/optimize
them, visualize and analyse the results. Its features should be sufficient to enable mathematical
description of chemical, physical or socio/economic phenomena. The most common are initial value
problems of implicit form, which can be formulated as systems of linear, non-linear, and (partial)
differential algebraic equations.

=====================
Programming paradigms
=====================

In general, there are two types of approaches that can be applied to process modelling:
Domain Specific Language approach and a general-purpose programming language approach (such as
c/c++, Java or Python). A Domain Specific Language (DSL) is a special-purpose programming or
specification language dedicated to a particular problem domain and so designed that it directly
supports the key concepts necessary to describe the underlying problems. A domain-specific
language is created specifically to solve problems in a particular domain and is usually not
intended to be able to solve problems outside it (although that may be technically possible in
some cases). In contrast, general-purpose languages are created to solve problems in a wide
variety of application domains.

Domain-specific languages are languages with very specific goals in design and implementation and
commonly lack low-level functions for filesystem access, interprocess control, and other functions
that characterize full-featured programming languages, scripting or otherwise.

A good example of general purpose (multi-domain) domain specific language is `Modelica <http://www.modelica.org>`_
while single-domain (chemical processing industry related) DSLs are `gPROMS <http://www.psenterprise.com/gproms>`_,
`Ascend <http://ascend4.org>`_, `SpeedUp <http://www.aspentech.com>`_ etc.

DAE Tools approach is a sort of the hybrid approach: it applies general-purpose programming languages
such as c++ and Python, but offers a class-hierarchy/API that resembles a syntax of a DSL as much as
possible, an access to the low-level functions, large number of standard and third party libraries and
uses state of the art free/open-source software components to accomplish particular tasks (calculating
derivatives and sensitivities, solving systems of differential and algebraic systems of equations and
optimization problems, processing and plotting results etc).

.. list-table::
    :widths: 80 80
    :header-rows: 1

    * - **DSL Approach**
      - **DAE Tools Approach**
    * - Domain-specific languages allow solutions to be expressed in the idiom and at the level of abstraction
        of the problem domain (direct support for all modelling concepts by the language syntax)
      - Modelling concepts cannot be expressed directly in the programming language and have to be emulated in
        the API or in some other way
    * - Clean, concise, ellegant and natural way of building model descriptions: the code can be self documenting
      - The support for modelling concepts is much more verbose and less elegant; however, DAE Tools can generate
        XML+MathML based model reports that can be either rendered in XHTML format using XSLT transformations
        (representing the code documentation) or used as an XML-based model exchange language.
    * - Domain-specific languages could enhance quality, productivity, reliability, maintainability and portability
      -
    * - DSLs could be and often are simulator independent making a model exchange easier
      - Programming language dependent; however, a large number of scientific software libraries exposes its
        functionality to Python via Python wrappers
    * - Cost of designing, implementing, and maintaining a domain-specific language as well as the tools required
        to develop with it (IDE): a compiler/lexical parser/interpreter must be developed with all burden that comes
        with it (such as error handling, grammar ambiguities, hidden bugs etc)
      - A compiler/lexical parser/interpreter is an integral part of the programming language (c++, Python) with a
        robust error handling, universal grammar and massively tested
    * - Cost of learning a new language vs. its limited applicability: users are required to master a new language
        (yet another language grammar)
      - No learning of a new language required (everything can get done in a favourite programming language)
    * - Increased difficulty of integrating the DSL with other components: calling external functions/libraries and
        interaction with other software is limited by the existence of wrappers around a simulator engine
        (for instance some scripting languages like Python or javascript)
      - Calling external functions/libraries is a natural and straightforward Interaction with other software is
        natural and straightforward
    * - Models usually cannot be created in the runtime/on the fly (or at least not easily) and cannot be modified
        in the runtime
      - Models can be created in the runtime/on the fly and easily modified in the runtime
    * - Setting up a simulation (ie. the values of parameters values, initial conditions, initially active states)
        is embedded in the language and it is typically difficult to do it on the fly or to obtain the values from
        some other software (for example to chain several software calls where outputs of previous calls represent
        inputs to the subsequent ones)
      - Setting up a simulation is done programmaticaly and the initial values can be obtained from some other software
        in a natural way (chaining several software calls is easy since a large number of libraries make Python wrappers
        available)
    * - Simulation operating procedures are not flexible; manipulation of model parameters, variables, equations,
        simulation results etc is limited to only those operations provided by the language
      - Operating procedures are completely flexible (within the limits of a programming language itself) and a
        manipulation of model parameters, variables, equations, simulation results etc can be done in any way which
        a user cosiders suitable for his/her problem
    * - Only the type of results provided by the language/simulator is available; custom processing is usually not
        possible or if a simulator does provide a way to build extensions it is limited to the functionality made
        available to them
      - The results processing can be done in any way which a user considers suitable(again within the limits of a
        programming language itself)


=================
The main features
=================

**DAE Tools** is a cross-platform equation-oriented process modelling and optimization system. All core libraries
are written in standard ANSI/ISO c++ . It is highly portable - it can run on every platform with a decent c++ compiler,
Boost and standard c/c++ libraries (by now it is tested on 32/64 bit x86 and ARM architectures making it suitable for
use in embedded systems). **DAE Tools** core libraries are small and fast, and each module can be easily extended.
Models can be developed in Python (**pyDAE** module) or c++ (**cDAE** module), compiled into an independent
executable and deployed without a need for any run time libraries.

Various types of processes (lumped or distributed, steady-state or dynamic) can be modelled and optimized. They may
range from very simple to those which require complex operating procedures. Equations can be ordinary or discontinuous,
where discontinuities are automatically handled by the framework. Model reports  containing all information about
a model can be exported in XML MathML format automatically creating a high quality documentation. The simulation
results can be visualized, plotted and/or exported into various formats.

Currently `Sundials IDAS <https://computation.llnl.gov/casc/sundials/main.html>`_ solver is used to solve DAE systems
and calculate sensitivities, while `BONMIN <https://projects.coin-or.org/Bonmin>`_,
`IPOPT <https://projects.coin-or.org/IPOPT>`_,
and `NLOPT <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_ solvers are used to solve NLP/MINLP problems.
**DAE Tools** support direct dense and sparse matrix linear solvers (sequential and multi-threaded versions)
at the moment. In addition to the built-in Sundials linear solvers, several third party libraries are interfaced:
`SuperLU/SuperLU_MT <http://crd.lbl.gov/~xiaoye/SuperLU/index.html>`_,
`Intel Pardiso <http://software.intel.com/en-us/intel-mkl>`_, `AMD ACML <http://www.amd.com/acml>`_,
`Trilinos Amesos <http://trilinos.sandia.gov/packages/amesos/>`_ (KLU, Umfpack, SuperLU, Lapack),
and `Trilinos AztecOO <http://trilinos.sandia.gov/packages/aztecoo>`_ (with built-in, Ifpack or ML preconditioners)
which can take advantage of multi-core/cpu computers. Linear solvers that exploit general-purpose graphics processing
units (`GPGPU <http://en.wikipedia.org/wiki/GPGPU>`_, such as `NVidia CUDA <http://www.nvidia.com/object/cuda_home_new.html>`_)
are also available ([[SuperLU_CUDA]], `CUSP <http://code.google.com/p/cusp-library>`_) but in an early development stage.

DAE Tools models can be exported into some other modelling languages. At the moment, models can be exported into
pyDAE (python) and cDAE (c++) but other languages will be supported in the future (such as OpenModelica, EMSO ...).

===============
System overview
===============

=======
Licence
=======

**DAE Tools** is `free software <http://www.gnu.org/>`_ and you can redistribute it and/or modify it under the terms of
the `GNU General Public Licence <http://www.gnu.org/licenses/licenses.html#GPL>`_ version 3 as published by
the Free Software Foundation (`GNU philosophy <http://www.gnu.org/philosophy/free-sw.html>`_).

=======
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

================
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

DAE Tools can optionally use the following proprietary software libraries:

* AMD ACML linear solver (pyAmdACML module): `<http://www.amd.com/acml>`_
* Intel MKL linear solvers (pyIntelMKL and pyIntelPardiso modules): `<http://software.intel.com/en-us/articles/intel-mkl>`_

Please see the corresponding websites for more details about the licences.

===========
How to cite
===========

If you use **DAE Tools** in your work then please cite it in the following way:
D. Nikolic, DAE Tools process modelling software, 2010. http://www.daetools.com


.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
