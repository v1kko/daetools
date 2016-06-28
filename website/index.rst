*****************
DAE Tools Project
*****************
..
    Copyright (C) Dragan Nikolic, 2016
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

..
    .. include:: docs/introduction.rst
        :start-after: begin-command
        :end-before: end-command

**DAE Tools** is a cross-platform equation-based object-oriented modelling, simulation and optimisation software.
It is initially developed for simulation of processes in chemical process industry
(mass, heat and momentum transfers, chemical reactions, separation processes, thermodynamics).
Today, **DAE Tools** is multi-domain.
**DAE Tools**  is released under the `GNU General Public Licence <http://www.gnu.org/licenses/licenses.html#GPL>`_
and runs on all major operating systems (Windows, GNU Linux and Mac OS) and architectures (x86, x86_64, arm).

Broadly speaking, it is not a modelling language nor an integrated software suite of data structures and routines
for scientific applications, but rather a higher level structure – an architectural design of interdependent
software components providing an API for:

* Model specification
* Activities on developed models (simulation, optimisation, parameter estimation)
* Processing of the results, such as plotting and exporting to various file formats
* Report generation
* Code generation, co-simulation and model exchange

Class of problems that can be solved by **DAE Tools**:

* Initial value problems of implicit form, described by a system of linear, non-linear, and (partial-)differential
  algebraic equations
* Index-1 DAE systems
* With lumped or distributed parameters: Finite Difference or Finite Elements Methods (still experimental)
* Steady-state or dynamic
* Continuous with some elements of event-driven systems (discontinuous equations, state transition networks
  and discrete events)

**DAE Tools** apply a hybrid approach between modelling and general purpose programming languages, combining
the strengths of both approaches into a single one. The most important features of the hybrid approach are:

1. Support for the runtime model generation
2. Support for the runtime simulation set-up
3. Support for complex runtime operating procedures
4. Interoperability with the third-party software
5. Suitability for embedding and use as a web application or software as a service
6. Code-generation, model exchange and co-simulation capabilities

More information about DAE Tools can be found in the :doc:`docs/introduction` section
of the :doc:`docs/index` and the following publications:

- Nikolić DD. (2016) *DAE Tools: equation-based object-oriented modelling, simulation and optimisation software*.
  **PeerJ Computer Science** 2:e54. `doi:10.7717/peerj-cs.54 <https://doi.org/10.7717/peerj-cs.54>`_
  (preprint available on: `ResearchGate <https://www.researchgate.net/publication/285596842_DAE_Tools_Equation-Based_Object-Oriented_Modelling_Simulation_and_Optimisation_Software>`_).
- `Introduction to DAE Tools <http://www.slideshare.net/DraganNikoli5/dae-tools-introduction>`_
  (on `ResearchGate <https://www.researchgate.net/publication/303260138_DAE_Tools_-_Introduction>`_)


Download
========

**The current release is 1.5.0**.

Installation files can be found in the SourceForge website
`download section <https://sourceforge.net/projects/daetools/files/1.5.0>`_,
and the source code in the SourceForge
`subversion repository <https://sourceforge.net/p/daetools/code/HEAD/tree>`_.

More information on system requirements, downloading and installing **DAE Tools**
can be found in :doc:`docs/getting_daetools`.

News
====
[**June 29 2016**] The new 1.5.0 version is released. The most important new features:
    
-  The new c++/MPI code generator. It can generate the c++ source code that contains the exported simulation,
   data partitioning and interprocess communication using MPI. At the moment it is in the prrof of the concept stage.
-  Updated other code generators. FMI code generator tested using the available tests.
-  New types of plots in the DAE Plotter: animated 2D plot (including the video export), user-defined plots
   (through user-specified python source code) and plotting of user specified data.
-  Fixed bugs in calculation of initial conditions in daeSimulation.SolveInitial() and daeSimulation.Reinitialize() functions.
-  Added global dt, d, d2, dt_array, d_array and d2_array functions that calculate time/partial derivatives.
-  A number of small fixes and updates

Full list of news can be found here: :doc:`news`

[**April 8 2016**] The first article on DAE Tools has been published in *PeerJ Computer Science*:
     Nikolić DD. (2016) *DAE Tools: equation-based object-oriented modelling, simulation and optimisation software*.
     **PeerJ Computer Science** 2:e54. `doi:10.7717/peerj-cs.54 <https://doi.org/10.7717/peerj-cs.54>`_.

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
