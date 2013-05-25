*****
News
*****
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

.. begin-command

.. end-command

Bug fixes, 11.10.2012.
----------------------

-  3D plot bug fix when detecting free domains (by Caleb Huttingh)

.. _v1_2_1:
    
v1.2.1, 14.06.2012.
-------------------

List of changes/new features:

-  Integration speed improvements (more than an order of magnitude, in
   some cases); no need for a memory copy from/to the DAE solver, a
   better integration step control and an option to avoid sparse matrix
   re-creations after a discontinuity
-  A new option added to the daetools.cfg config. file:
   resetLAMatrixAfterDiscontinuity; it applies only to sparse matrix LA
   solvers; if true LA solvers will recreate sparse matrix each time a
   discontinuity is detected (since the sparsity pattern might be
   changed); if false the DAE solver will create a single sparse matrix
   that includes a sparsity pattern from all states so that there is no
   need to recreate matrix each time a discontinuity is located; this
   obviously introduces higher memory requirements but brings
   significant integration speed improvements
-  SuperLU LA solver can choose between two modes of reusing the
   factorization information from the previous steps: SamePattern and
   SamePattern\_SameRowPerm (for more info see the superlu
   documentation); a new option added to the daetools.cfg config. file:
   factorizationMethod which can have one of the values above
-  SuperLU LA solver can be instructed to create all the memory it needs
   at the beginning of simulation; this can be controlled in the
   daetools.cfg file by setting the useUserSuppliedWorkSpace option to
   true and adjusting the workspaceSizeMultiplier and
   workspaceMemoryIncrement options (for more info see the superlu
   ocumentation)
-  Added support for units; variables, parameters, domains must have a
   numerical value in terms of a unit of measurement (quantity) and
   units-consistency is strictly enforced (although it can be switched
   off in the daetools.cfg config file); added three new classes:
   base\_unit, unit and quantity and a new module: pyUnits
-  A new option added to the daetools.cfg config. file:
   checkUnitsConsistency; if true the system will perform
   units-consistency tests for equations and logical expressions during
   the initialization phase
-  Functions (Re)SetInitialCondition, SetInitialGuess, (Re)AssignValue
   in daeVariable and SetValue in daeParameter accept both floating
   point values and quantities; in the former case it is assumed that
   the value is in the units of the parameter/variable while in the
   later the numerical values is first converted to the
   parameter/variable units
-  C++ tutorials and the python modules reorganized; now c++ tutorials
   are in the folder cxx-tutorials while the completely new folder tree
   has been created for python modules: all files are in the
   daetools-package folder
-  Added platform specific folders for python extension modules
-  Added support for python dist-utils (the file setup.py in the
   daetools-package folder)
-  New functions in daeVariable: (Re)SetInitialConditions,
   SetInitialGuesses, (Re)AssignValues that set init. conditions, init.
   guesses or assign values of all points in a distributed variable
-  All constants in equations must be dimensional and assigned units;
   two new functions (Constant and Array) are added that create single
   or an array of dimensional quantities
-  The functions Time and Constant moved from the daeModel class to the
   global namespace
-  A basic support for external functions (daeScalarExternalFunction and
   daeVectorExternalFunction) that can handle and evaluate functions
   existing in external libraries; in the future versions of daetool
   certain software components such as thermodynamic property packages
   will be supported
-  A new type of 2D plots: Animated2D plot
-  Trilinos family of LA solvers have also cDAE version
-  Added a new function to daeSimulation: CleanUpData; in case of very
   large systems a lot of memory can be freed and made available to the
   system after the initialization; this is still an experimental option
-  Array\_xxx functions in daeVariable/daeParameter accept python lists
   and slices
-  Equations can have an optional scaling; added two new functions:
   GetScaling/SetScaling (the property Scaling in pyDAE)
-  Improved data reporting speed
-  Parameters values can also be reported
-  Changes in data reporting during an optimization (now all iterations
   are reported independently)
-  A new data reporter class: daeNoOpDataReporter; it just collects the
   reported values and does not do any processing (useful for building
   custom data reporters)
-  OnEvent function can also accept events from outlet ports
-  Enabled the option for the Lapack LA solver in Sundials IDAS
-  Mac OSX port
-  c++ (cDAE) tutorials
-  Added several new functions to the daeLog\_t and a progress bar to
   the daeSimulator; the new functions are GetProgress/SetProgress
   (property Progress), GetEnabled/SetEnabled (property Enabled),
   GetPrintProgress/SetPrintProgress (property PrintProgress),
   GetPercentageDone/SetPercentageDone (property PercentageDone) and
   GetETA (read-only property ETA)
-  daeStdOutLog and daePythonStdOutLog print the progress information to
   the console
-  Fixed bug in all versions of LA solvers in cDAE (a responsibility to
   destroy objects and to free memory was done automatically by a DAE
   solver: now it is users responsibility)
-  SuperLU and SuperLU\_MT now statically linked
-  Removed dependence on the system version of the boost libraries; all
   platforms now use the same version of the custom built boost libs
   (1.49.0)
-  Fixed bug in python wrappers ("pure virtual function called") that
   was related to the sequence of datareporter and simulation objects
   instantiation
-  Updated stylesheets and xsl transformation files for model reports
-  Other small improvements and minor bugs fixes

v1.1.2, 29.09.2011
------------------

List of new features:

-  *daeObjectiveFunction*, *daeOptimizationVariable*, and
   *daeOptimizationConstraint* classes have two new attributes (*Value*
   and *Gradients*). *daeSimulation::Initialize* function accepts an
   additional argument *bCalculateGradients* (default is false) which
   instructs simulation object to calculate gradients of the objective
   function and optimization variables specified in
   *daeSimulation::SetUpSensitivityAnalysis* overloaded function. These
   changes allow much easier coupling of daetools with some external
   software (as given in optimization tutorials 4 and 5).
-  New type of ports: *event ports* (*daeEventPort* class). Event ports
   allow sending of messages (events) between two units (models). Events
   can be triggered manually or as a result of a state transition in a
   model. The main difference between event and ordinary ports is that
   the former allow a discrete communication between units while latter
   allow a continuous exchange of information. A single outlet event
   port can be connected to unlimited number of inlet event ports.
   Messages contain a floating point value that can be used by a
   recipient; that value might be a simple number or an expression
   involving model variables/parameters.
-  A new function *ON\_EVENT* in the daeModel class that specifies how
   the incoming events on a specific event port are handled (that is the
   actions to be undertaken when the event is received; class:
   *daeAction*). *ON\_EVENT* handlers can be specified in models and in
   states so that the actions executed when the event is trigerred can
   differ subject to the current active state. Four different types of
   actions can be specified:

   -  Change the active state in the specified state transition network
   -  Trigger an event on the specified outlet event port
   -  Reassign or reinitialize a value of the specified variable
   -  Execute the user-defined action (users should derive a new class
      from daeAction and overload the function *Execute*)

-  A new way of handling state transitions: the function *ON\_CONDITION*
   in daeModel that specifies actions to be undertaken when the logical
   condition is satisfied. The same types of actions as in the function
   *ON\_EVENT* are supported. The old function SWITCH\_TO is still
   supported but the new one should be used for it is much flexible.
-  Non-linear least square minimization with daeMinpackLeastSq (scipy
   wrapper of Levenberg-Marquardt algorithm from
   `Minpack <http://www.netlib.org/minpack>`__)
-  Examples of *DAE Tools* and *Scipy* interoperabilty
   (*scipy.optimize.fmin*, *scipy.optimize.leastsq*)
-  Fixed sensitivity calculation in steady-state models. There was no
   bug in the previous versions, but if the objective function or
   constraint did not explicitly depend on some of the optimization
   variables the calculated sensitivity for these variables was zero.
-  Developed shell scripts to compile third party libraries (Sundials
   IDAS, SuperLU/SuperLU\_MT, Trilinos, Bonmin, and NLopt), DAE Tools
   core libraries and boost.python extension modules
   (*compile\_libraries\_linux.sh*, *compile\_linux.sh*).
-  The new function *time* in *daeModel* class; it returns adouble
   object with the current time elapsed in the simulation that can be
   used in define equations' residuals.
-  The new property 'ReportingTimes' in daeSimulation class that returns
   time points when data should be reported.
-  Fixed bug in daePlotter when there was a variable and a port with the
   same name within the model. Now a port and a variable can have the
   same name.
-  Some of the tutorials are available in c++ (cDAE) too.
-  Because of the way how the standard c++ library handles the
   ''std::vector' internal memory storage the memory requiremens could
   possibly grow rather high for large models. That is fixed now and
   vectors will not demand more memory than required for elements
   storage; that is achieved by explicitly allocating memory for all
   elements and comes with some penalties (small speed loss during the
   creation of the system, approximately 1%; however, the system
   creation time is very low and there is no overall performance
   degradation).
-  Some API polishing

v1.1.1, 17.06.2011
------------------

List of new features:

-  The main focus was to find and adapt a free multithreaded sparse
   direct solver for use with DAE Tools and it turned out that the best
   candidate is
   `SuperLU\_MT <http://crd.lbl.gov/~xiaoye/SuperLU/index.html>`__. As
   of DAE Tools v1.1.1 SuperLU (singlethreaded) and SuperLU\_MT
   (multithreaded) are recommended linear equation solvers. All the
   other (Trilinos group of solvers, Intel Pardiso, ...) will remain
   there but with less support.
-  A set of Krylov iterative solvers has been added. Trilinos AztecOO
   solver with IFPACK, ML or built-in preconditioners is available.
   However, iterative solvers are not fully working yet and these
   solvers are still in an early/experimental phase.
-  As the GPGPUs become more and more attractive an effort is made to
   try to offload computation of the most demanding tasks to GPU. The
   starting point is obviously a linear equation solver and two options
   are offered:

   -  `CUSP <http://code.google.com/p/cusp-library/>`__
   -  SuperLU_CUDA (OpenMP version of SuperLU\_MT modifed to work on
      CUDA GPU devices). The solver is still in the early development
      phase and the brief description is given in SuperLU_CUDA. Few
      issues still remain unsolved and a help from CUDA experienced
      developers is welcomed!

-  The new NLP solver has been added (NLOPT from the `Massachusetts
   Institute of Technology <http://web.mit.edu>`__). More information
   about NLOPT and available solvers can be found on `NLOPT wiki
   pages <http://ab-initio.mit.edu/wiki/index.php/NLopt>`__.

-  To separate NLP from MINLP problems the IPOPT is now a standalone
   solver.

-  All linear solvers are located in daetools/solvers directory.

-  Now all linear solvers support exporting sparse/dense matrices in
   .xpm image and matrix market file formats.

-  Models and ports now can be exported into some other modelling
   language. At the moment, models can be exported into pyDAE (python)
   and cDAE (c++) but other languages will be supported in the future
   (such as OpenModelica, EMSO, perhaps some proprietary etc...).

-  New data reporter (daeMatlabMATDataReporter) has been added that
   allows user to export the result into the Matlab MAT file format.

-  Operators + and - for daeDistributedEquationDomainInfo (daeDEDI)
   class which enable getting values/derivatives in distributed
   equations that are not equal to the index of the current iterator
   (see distillation column example for usage).

-  daeParameter/daeVariable constructors accept a list of domains
   (analogous to calling DistributeOnDomain for each domain).

-  Now all constraints are specified in the following way:

   -  Inequality constraints: g(i) <= 0
   -  Equality constraints: h(i) = 0

-  DAE Tools source code has been checked by Valgrind and no memory
   leaks has been detected.

-  Development of some useful models has been started. The models are
   located in model\_library directory.

-  A set of standard variable types has been developed. Variable types
   are located in daeVariableTypes.py file.

-  Several minor bug fixes.


.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
