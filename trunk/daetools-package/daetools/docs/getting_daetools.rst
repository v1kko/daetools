*****************
Getting DAE Tools
*****************
..
    Copyright (C) Dragan Nikolic
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.


System requirements
===================

Supported platforms:
    
* GNU/Linux (i686, x86_64)
* Windows (32 bit and 32 bit version of DAE Tools on 64 bit)
* MacOS (x86, x86_64)

The software works on both python 2 and 3. The binaries are provided for 2.7, 3.4 and 3.5 (except on Windows).

Mandatory packages:

* Python (2.7, 3.4\ :sup:`+`): `<http://www.python.org>`_
* Numpy (1.8\ :sup:`+`): `<http://www.numpy.org>`_
* Scipy (0.14\ :sup:`+`): `<http://www.scipy.org>`_
* Matplotlib (1.4\ :sup:`+`): `<http://matplotlib.sourceforge.net>`_
* pyQt4: `<http://www.riverbankcomputing.co.uk/software/pyqt>`_
* mayavi2
* python-lxml

Optional packages:

* python-xlwt
* python-h5py
* python-pandas

Optional packages (proprietary):

* Pardiso linear solver: `<http://www.pardiso-project.org>`_
* Intel Pardiso linear solver: `<https://software.intel.com/en-us/intel-mkl>`_

For more information on how to install packages please refer to the documentation for the specific library.
By default all versions (GNU/Linux, Windows and MacOS) come with the Sundials dense LU linear solver,
SuperLU, SuperLU_MT, Trilinos Amesos (with built-in support for KLU, SuperLU and Lapack linear solvers),
Trilinos AztecOO (with built-in support for Ifpack and ML preconditioners), NLOPT and IPOPT/BONMIN
(with MUMPS linear solver and PORD ordering).

Additional linear solvers (such as Pardiso and IntelPardiso) must be downloaded
separately since they are subject to different licensing conditions (not free software).

Getting the packages
====================

The installation files can be downloaded from the `downloads <http://daetools.com/downloads.html>`_ section
or from the `SourceForge <https://sourceforge.net/projects/daetools/files>`_ website.

.. note:: From the version 1.2.1 **DAE Tools** use distutils to distribute python packages and extensions.

.. note:: From the version 1.6.0 Windows installer is not provided anymore.

The naming convention for the installation files: ``daetools-major.minor.platform-architecture.tar.gz``
where ``major.minor.build`` represents the version (``1.5.0`` for instance),
``platform`` can be ``gnu_linux``, ``win32`` and ``macosx``, and
``architecture`` can be ``i686``, ``x86_64`` or ``universal``.

An example: ``daetools-1.5.0-gnu_linux-x86_64.tar.gz`` is the version 1.5.0 for 64 bit GNU/Linux.

For other platforms, architectures and python versions not listed in `System requirements`_
daetools must be compiled from the source.
The source code can be downloaded either from the subversion tree or from the download section
(``daetools-1.5.0-source.tar.gz`` for instance).

Installation
============

GNU/Linux
---------

Install Python and Python packages
++++++++++++++++++++++++++++++++++
Use the system's Python
///////////////////////

* Debian GNU/Linux and derivatives (Ubuntu, Linux Mint)

  .. code-block:: bash

     sudo apt-get install python-numpy python-scipy python-matplotlib python-qt4 mayavi2 python-lxml
     # Optional packages:
     sudo apt-get install python-xlwt python-h5py python-pandas

* Red Hat and derivatives (Fedora, CentOS):

  .. code-block:: bash

     sudo yum install numpy scipy python-matplotlib PyQt4 Mayavi python-lxml
     # Optional packages:
     sudo yum install python-xlwt h5py python-pandas

* SUSE Linux:

  .. code-block:: bash

     sudo zypper in python-numpy python-scipy python-matplotlib python-qt4 python-lxml
     # Optional packages:
     sudo zypper in python-xlwt h5py python-pandas

* Arch Linux:

  .. code-block:: bash

     # Python 2:
     sudo pacman -S python2-numpy python2-scipy python2-matplotlib python2-pyqt4 mayavi python-lxml
     # Optional packages:
     sudo pacman -S python2-xlwt python-h5py python-pandas

     # Python 3:
     sudo pacman -S python-numpy python-scipy python-matplotlib python-pyqt4 mayavi python-lxml
     # Optional packages:
     sudo pacman -S python-xlwt python-h5py python-pandas

Install one of scientific python distributions
//////////////////////////////////////////////

* `Anaconda <https://www.continuum.io/downloads>`_
* `Miniconda <https://conda.io/miniconda.html>`_

  Install dependencies using:

  .. code-block:: bash

     conda install numpy scipy matplotlib pyqt=4.11 lxml pandas h5py xlwt

* `Enthought Canopy <https://www.enthought.com/products/canopy>`_

Install DAE Tools
+++++++++++++++++
Unpack the downloaded archive, cd to the ``daetools-X.Y.Z-platform-architecture`` folder and install **DAE Tools** by typing
the following shell command:

.. code-block:: bash

   sudo python setup.py install

You can also install **DAE Tools** into a python virtual environment:

.. code-block:: bash

   source activate <environment_name>
   python setup.py install


MacOS
-----
Install Python and Python packages
++++++++++++++++++++++++++++++++++

Install one of scientific python distributions
//////////////////////////////////////////////
* `Anaconda <https://www.continuum.io/downloads>`_
* `Miniconda <https://conda.io/miniconda.html>`_

  Install dependencies using:
      
  .. code-block:: bash

    conda install numpy scipy matplotlib pyqt=4.11 lxml pandas h5py xlwt
  
* `Enthought Canopy <https://www.enthought.com/products/canopy>`_

Use the system's Python
///////////////////////
The system's python version usually does not work well (therefore, install a custom python).
First, install the mandatory packages: python 2.7, numpy, scipy, matplotlib and pyqt4.
As a starting point the following links can be used:

* `Python <http://www.python.org>`_
* `NumPy <http://sourceforge.net/projects/numpy/files/NumPy>`_
* `SciPy <http://sourceforge.net/projects/scipy/files/scipy>`_
* `Matplotlib <http://sourceforge.net/projects/matplotlib/files/matplotlib>`_
* `PyQt4 <http://www.riverbankcomputing.com/software/pyqt/download>`_

Install DAE Tools
+++++++++++++++++
Unpack the downloaded archive, cd to the ``daetools-X.Y.Z-platform-architecture`` folder and install **DAE Tools** by typing
the following shell command:

.. code-block:: bash

    sudo python setup.py install

You can also install **DAE Tools** into a python virtual environment:

.. code-block:: bash

   source activate <environment_name>
   python setup.py install


Windows
-------
Install Python and Python packages
++++++++++++++++++++++++++++++++++
The easiest way is to install one of available scientific python distributions:
    
* `Anaconda <https://www.continuum.io/downloads>`_
* `Miniconda <https://conda.io/miniconda.html>`_
  
  Install dependencies using:

  .. code-block:: bash

     conda install numpy scipy matplotlib pyqt=4.11 lxml pandas h5py xlwt
  
* `Enthought Canopy <https://www.enthought.com/products/canopy>`_
* `Python(x,y) <https://python-xy.github.io/>`_

.. note:: Only 32-bit version of **DAE Tools** is available for Windows. However, **DAE Tools** can be used on 64-bit
          versions of Windows by installing the 32-bit python.

To be able to create 3D plots you need to install Mayavi2 package. It can be installed using the following shell command:

.. code-block:: bash

    easy_install "Mayavi[app]"


Install DAE Tools
+++++++++++++++++
No installers are provided for Windows anymore. The installation process is the same for all platforms.
Unpack the downloaded archive, cd to the ``daetools-X.Y.Z-platform-architecture`` folder and install **DAE Tools** by typing
the following shell command:

.. code-block:: bash

    python setup.py install

You can also install **DAE Tools** into a python virtual environment:

.. code-block:: bash

   source activate <environment_name>
   python setup.py install

..
    Additional linear equation solvers (proprietary)
    ------------------------------------------------
    Optionally you can also install proprietary `AMD ACML <http://www.amd.com/acml>`_ and
    `Intel MKL <http://software.intel.com/en-us/intel-mkl/>`_ libraries.
    Please follow the installation procedures in the documentation. **pyAmdACML** and **pyIntelMKL/pyIntelPardiso**
    modules are compiled against ACML 4.4.0 and MKL 10.2.5.035 respectively. Also have a look on the licensing
    conditions (**these libraries are not** `**free software** <http://www.gnu.org/philosophy/free-sw.html>`_).

    In order to use AMD ACML and Intel MKL libraries you have to do some additional configuration.
    You can follow the instructions in the corresponding package documentation or do a quick setup as described below:

    #**GNU/Linux**: setup for a single user<br /> Copy `<acml_mkl_bashrc this file>`_ to your home folder,
    edit it so that it reflects your installation and add the line. $HOME/acml_mkl_bashrc  at the end of $HOME/.bashrc file
    #**GNU/Linux**: setup for all users<br /> Subject to your machine architecture and library versions
    (here **x86_64** GNU/Linux with **ACML v4.4.0** and **MKL v10.2.5.035**), put the following lines in
    /etc/ld.so.conf and execute ldconfig: /opt/intel/mkl/10.2.5.035/lib/em64t /opt/acml4.4.0/gfortran64_mp/lib
    #**Windows XP**:<br /> If not already added, add the following line to your **PATH** environment variable
    (Control Panel -> System): c:\AMD\acml4.4.0\ifort32_mp\lib;c:\Intel\MKL\10.2.5.035\ia32\bin\

    
Compiling from source
=====================

To compile the **DAE Tools** the following is needed:
    
* Installed ``python`` and ``numpy`` modules
* Compiled third party libraries and DAE/LA/NLP solvers: ``Sundials IDAS``, ``Trilinos``,
  ``SuperLU``, ``SuperLU_MT``, ``Bonmin``, ``NLopt``, ``deal.II``

All **DAE Tools** modules are developed using the QtCreator/QMake cross-platform integrated development environment.
The source code can be downloaded from the SourceForge website or checked out from the
`DAE Tools subversion repository <https://svn.code.sf.net/p/daetools/code>`_:

.. code-block:: bash

    svn checkout svn://svn.code.sf.net/p/daetools/code daetools


GNU/Linux and MacOS
-------------------

.. _from_the_command_line:

From the command line
+++++++++++++++++++++
First, install all the necessary dependencies by executing ``install_python_dependencies_linux.sh`` and
``install_dependencies_linux.sh`` shell script located in the ``trunk`` directory.
They will check the OS you are running (currently Debian, Ubuntu, Linux Mint, CentOS, Suse Linux,
Arch Linux and Fedora are supported but other can be easily added) and install all necessary packages needed for **DAE Tools**
development.

.. code-block:: bash

    # 'lsb_release' command might be missing on some GNU/Linux platforms
    # and has to be installed before proceeding.
    # On Debian based systems:
    # sudo apt-get install lsb-release
    # On red Hat based systems:
    # sudo yum install redhat-lsb

    cd daetools/trunk
    sh install_dependencies_linux.sh


Then, compile all the third party libraries by executing ``compile_libraries_linux.sh`` shell script located in the
``trunk`` directory. The script will download all necessary source archives from the **DAE Tools** SourceForge web-site,
unpack them, apply changes and compile them. If all dependencies are installed there should not be problems compiling
the libraries.

.. code-block:: bash

    sh compile_libraries_linux.sh all

It is also possible to compile individual libraries using one of the following options:

.. code-block:: none

    Individual libraries/solvers:
      boost            Boost libraries (system, filesystem, thread, python)
      ref_blas_lapack  reference BLAS and Lapack libraries
      umfpack          Umfpack solver
      idas             IDAS solver
      superlu          SuperLU solver
      superlu_mt       SuperLU_MT solver
      bonmin           Bonmin solver
      nlopt            NLopt solver
      trilinos         Trilinos Amesos and AztecOO solvers
      deal.ii          deal.II finite elements library

After compilation, the shared libraries will be located in ``trunk/daetools-package/daetools/solibs`` directory.

Finally, compile all **DAE Tools** libraries and python modules by executing ``compile_linux.sh`` shell script located
in the ``trunk`` directory.

.. code-block:: bash

    sh compile_linux.sh all

It is also possible to compile individual libraries using one of the following options:

.. code-block:: none

    all             Build all daetools c++ libraries, solvers and python extension modules.
                    Equivalent to: dae superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
    dae             Build all daetools c++ libraries and python extension modules (no 3rd party LA/(MI)NLP/FE solvers).
                    Equivalent to: units data_reporting idas core activity simulation_loader fmi
    solvers         Build all solvers and their python extension modules.
                    Equivalent to: superlu superlu_mt trilinos ipopt bonmin nlopt deal.ii
    pydae           Build daetools core python extension modules only.

    Individual projects:
        core                Build Core c++ library and its python extension module (pyCore).
        activity            Build Activity c++ library and its python extension module (pyActivity).
        data_reporting      Build DataReporting c++ library and its python extension module (pyDataReporting).
        idas                Build IDAS c++ library and its python extension module (pyIDAS).
        units               Build Units c++ library and its python extension module (pyUnits).
        simulation_loader   Build simulation_loader shared library.
        fmi                 Build FMI wrapper shared library.
        trilinos            Build Trilinos Amesos/AztecOO linear solver and its python extension module (pyTrilinos).
        superlu             Build SuperLU linear solver and its python extension module (pySuperLU).
        superlu_mt          Build SuperLU_MT linear solver and its python extension module (pySuperLU_MT).
        pardiso             Build PARDISO linear solver and its python extension module (pyPardiso).
        intel_pardiso       Build Intel PARDISO linear solver and its python extension module (pyIntelPardiso).
        bonmin              Build BONMIN minlp solver and its python extension module (pyBONMIN).
        ipopt               Build IPOPT nlp solver and its python extension module (pyIPOPT).
        nlopt               Build NLOPT nlp solver and its python extension module (pyNLOPT).
        deal.ii             Build deal.II FEM solvers and its python extension module (pyDealII).

All python extensions are located in the platform-dependent locations in ``trunk/daetools-package/daetools/pyDAE`` and
``trunk/daetools-package/daetools/solvers`` folders.

**DAE Tools** can be now installed using the information from the sections above.

.. _from_qtcreator_ide:

From QtCreator IDE
++++++++++++++++++
DAE Tools can also be compiled from within QtCreator IDE. First install dependencies and compile third party libraries
(as explained in the compilation :ref:`from the command line <from_the_command_line>`) and then do the following:
    
* Do not do the shadow build. Uncheck it (for all projects) and build everything in the release folder
* Choose the right specification file for your platform (usually it is done automatically by the IDE, but double-check it):
    
  * for GNU/Linux use ``-spec linux-g++``
  * for MacOS use ``-spec macx-g++``

* Compile the ``dae`` project (you can add the additional Make argument ``-jN`` to speed-up the compilation process,
  where N is the number of processors plus one; for instance on the quad-core machine you can use ``-j5``)
* Compile ``SuperLU/SuperLU_MT`` and ``Bonmin/Ipopt`` solvers.
  ``SuperLU/SuperLU_MT`` and ``Bonmin/Ipopt`` share the same code and the same project file so some
  hacking is needed. Here are the instructions how to compile them:
    
  * Compiling ``libcdaeBONMIN_MINLPSolver.a`` and ``pyBONMIN.so``:
 
    * Set ``CONFIG += BONMIN`` in ``BONMIN_MINLPSolver.pro``, run ``qmake`` and then compile
    * Set ``CONFIG += BONMIN`` in ``pyBONMIN.pro``, run ``qmake`` and then compile
  
  * Compiling ``libcdaeIPOPT_NLPSolver.a`` and ``pyIPOPT.so``:
 
    * Set ``CONFIG += IPOPT`` in ``BONMIN_MINLPSolver.pro``, run ``qmake`` and then compile
    * Set ``CONFIG += IPOPT`` in ``pyBONMIN.pro``, run ``qmake`` and then compile
  
  * Compiling ``libcdaeSuperLU_LASolver.a`` and ``pySuperLU.so``:
 
    * Set ``CONFIG += SuperLU`` in ``LA_SuperLU.pro``, run ``qmake`` and then compile
    * Set ``CONFIG += SuperLU`` in ``pySuperLU.pro``, run ``qmake`` and then compile
  
  * Compiling ``libcdaeSuperLU_MT_LASolver.a`` and ``pySuperLU_MT.so``:
 
    * Set ``CONFIG += SuperLU_MT`` in ``LA_SuperLU.pro``, run ``qmake`` and then compile
    * Set ``CONFIG += SuperLU_MT`` in ``pySuperLU.pro``, run ``qmake`` and then compile

* Compile the ``LA_Trilinos_Amesos`` and then ``pyTrilinos`` project
* Compile the ``NLOPT_NLPSolver`` and then ``pyNLOPT`` project
* Compile the ``pyDealII`` project (no compile needed for ``FE_DealII`` project since all files there are header files/templates)

Windows
-------
DAE Tools support cross-compilation since the version 1.3.0. For more information about the ``mingw-w64``
toolchain and options read the help sections in ``compile_libraries_linux.sh`` and ``compile_linux.sh`` scripts.

First, compile the third party libraries:

.. code-block:: none

   Prerequisities:
     1. Install the mingw-w64 package from the main Debian repository.

     2. Install Python on Windows using the binary from the python.org website
        and copy it to trunk/PythonXY-arch (i.e. Python34-win32).
        Modify PYTHON_MAJOR and PYTHON_MINOR in the crossCompile section in the dae.pri file (line ~90):
            PYTHON_MAJOR = 3
            PYTHON_MINOR = 4

     3. cmake cross-compilation requires the toolchain file: set it up using -DCMAKE_TOOLCHAIN_FILE=[path_to_toolchain_file].cmake
        Cross-compile .cmake files are provided by daetools and located in the trunk folder.
          cross-compile-i686-w64-mingw32.cmake   file targets a toolchain located in /usr/mingw32-i686 directory.
          cross-compile-x86_64-w64-mingw32.cmake file targets a toolchain located in /usr/mingw32-x86_64 directory.

     4. deal.II specific options:
        The native "expand_instantiations_exe" is required but cannot be run under the build architecture.
        and must be used from the native build.
        Therefore, set up a native deal.II build directory first and run the following command in it:
            make expand_instantiations_exe
        Typically, it is located in the deal.II/common/scripts directory.
        That directory will be added to the PATH environment variable by this script.
        If necessary, modify the line 'export PATH=...:${PATH}' to match the actual location.

     5. Boost specific options:
        boost-python linking will fail. Append the value of:
           ${DAE_CROSS_COMPILE_PYTHON_ROOT}/libs/libpython${PYTHON_MAJOR}${PYTHON_MINOR}.a
        at the end of the failed linking command, re-run it, and manually copy the stage/lib/*.dll(s) to the "daetools/solibs/${PLATFORM}_${HOST_ARCH}" directory.
        Win64 (x86_64-w64-mingw32):
         - Python 2.7 won't compile (probably issues with the MS Universal CRT voodoo mojo)
         - dl and util libraries are missing when compiling with x86_64-w64-mingw32.
           solution: just remove -ldl and -lutil from the linking line.

     6. Trilinos specific options
        i686-w64-mingw32 specific:
          1. In the file:
            - trilinos/packages/teuchos/src/Teuchos_BLAS.cpp
             "template BLAS<...>" (lines 96-104)
                #ifdef _WIN32
                #ifdef HAVE_TEUCHOS_COMPLEX
                    template class BLAS<long int, std::complex<float> >;
                    template class BLAS<long int, std::complex<double> >;
                #endif
                    template class BLAS<long int, float>;
                    template class BLAS<long int, double>;
                #endif
             should be replaced by "template class BLAS<...>"
          2. In the files:
               - trilinos/packages/ml/src/Utils/ml_epetra_utils.cpp,
               - trilinos/packages/ml/src/Utils/ml_utils.c
               - trilinos/packages/ml/src/MLAPI/MLAPI_Workspace.cpp:
              the functions "gethostname" and "sleep" do not exist
                a) Add include file:
                      #include <winsock2.h>
                   and if that does not work (getting unresolved _gethostname function in pyTrilinos),
                   then comment-out all "gethostname" occurences (they are not important - just for printing some info)
                b) Rename sleep() to Sleep() (if needed, wasn't needed for 10.12.2)

        x86_64-w64-mingw32 specific:
          All the same as above. Additionally:
          1. trilinos/packages/teuchos/src/Teuchos_SerializationTraits.hpp
             Comment lines: UndefinedSerializationTraits<T>::notDefined();
          2. trilinos/packages/epetra/src/Epetra_C_wrappers.cpp
             Add lines at the beggining of the file:
               #pragma GCC diagnostic push
               #pragma GCC diagnostic warning "-fpermissive"

   Cross compiling notes:
     1. Requirements for Boost:
          --with-python-version 3.4
          --cross-compile-python-root .../trunk/Python34-win32
          --host i686-w64-mingw32

     2. The other libraries:
          --host i686-w64-mingw32 (the only necessary)

   Example cross-compile call:
       sh compile_libraries_linux.sh --with-python-version 3.4 --cross-compile-python-root ~/daetools-win32-cross/trunk/Python34-win32 --host i686-w64-mingw32 boost
       sh compile_libraries_linux.sh --host i686-w64-mingw32 ref_blas_lapack umfpack idas superlu superlu_mt trilinos bonmin nlopt deal.ii

Finally, compile all **DAE Tools** libraries and python modules by executing ``compile_linux.sh`` shell script located
in the ``trunk`` directory.

.. code-block:: bash

    sh compile_linux.sh --host i686-w64-mingw32 all

**DAE Tools** can be now installed using the information from the sections above.
