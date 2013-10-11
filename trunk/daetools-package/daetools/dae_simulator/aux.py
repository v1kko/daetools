#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
***********************************************************************************
                             solvers_aux.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2013
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************
"""
import sys
from PyQt4 import QtCore, QtGui
from daetools.pyDAE import *
from daetools.pyDAE.logs import daePythonStdOutLog 
from daetools.pyDAE.data_reporters import daePlotDataReporter, daeMatlabMATFileDataReporter 

(laSundialsLU, laSuperLU, laSuperLU_MT) = range(0, 3)
(laAmesos_Klu, laAmesos_Superlu, laAmesos_Umfpack, laAmesos_Lapack, laAztecOO) = range(3, 8)
(laIntelPardiso, laIntelMKL, laAmdACML, laLapack, laMagmaLapack, laSuperLU_CUDA, laCUSP) = range(8, 15)

(nlpIPOPT, nlpNLOPT, nlpBONMIN) = range(0, 3)

(TCPIPLog, PythonStdOutLog, StdOutLog, BaseLog, FileLog, DelegateLog) = range(0, 6)

(TCPIPDataReporter, NoOpDataReporter, DelegateDataReporter, TEXTFileDataReporter, PlotDataReporter) = range(0, 5)
(MatlabMATFileDataReporter, BlackHoleDataReporter) = range(5, 7)
    
def getAvailableNLPSolvers():
    available_nlp_solvers = []
    
    try:
        from daetools.solvers.ipopt import pyIPOPT
        available_nlp_solvers.append(("IPOPT NLP", nlpIPOPT))
        if 'daetools.solvers.ipopt' in sys.modules:
            del sys.modules['daetools.solvers.ipopt']
    except Exception as e:
        pass
    
    try:
        from daetools.solvers.nlopt import pyNLOPT
        available_nlp_solvers.append(("NLOPT NLP", nlpNLOPT))
        if 'daetools.solvers.nlopt' in sys.modules:
            del sys.modules['daetools.solvers.nlopt']
    except Exception as e:
        pass
    
    try:
        from daetools.solvers.bonmin import pyBONMIN
        available_nlp_solvers.append(("BONMIN MINLP", nlpBONMIN))
        if 'daetools.solvers.bonmin' in sys.modules:
            del sys.modules['daetools.solvers.bonmin']
    except Exception as e:
        pass
    
    return available_nlp_solvers
    
def getAvailableLASolvers():
    available_la_solvers = []
    
    available_la_solvers.append(("Sundials LU (dense, sequential, direct)", laSundialsLU))
    try:
        from daetools.solvers.superlu import pySuperLU
        available_la_solvers.append(("SuperLU (sparse, sequential, direct)", laSuperLU))
        if 'daetools.solvers.superlu' in sys.modules:
            del sys.modules['daetools.solvers.superlu']
    except Exception as e:
        print str(e)

    try:
        from daetools.solvers.superlu_mt import pySuperLU_MT
        available_la_solvers.append(("SuperLU_MT (sparse, pthreads, direct)", laSuperLU_MT))
        if 'daetools.solvers.superlu_mt' in sys.modules:
            del sys.modules['daetools.solvers.superlu_mt']
    except Exception as e:
        pass

    try:
        from daetools.solvers.trilinos import pyTrilinos
        suppSolvers = pyTrilinos.daeTrilinosSupportedSolvers()
        if 'Amesos_Klu' in suppSolvers:
            available_la_solvers.append(("Trilinos Amesos - KLU (sparse, sequential, direct)", laAmesos_Klu))
        if 'Amesos_Superlu' in suppSolvers:
            available_la_solvers.append(("Trilinos Amesos - SuperLU (sparse, sequential, direct)", laAmesos_Superlu))
        if 'Amesos_Umfpack' in suppSolvers:
            available_la_solvers.append(("Trilinos Amesos - Umfpack (sparse, sequential, direct)", laAmesos_Umfpack))
        if 'Amesos_Lapack' in suppSolvers:
            available_la_solvers.append(("Trilinos Amesos - Lapack (dense, sequential, direct)", laAmesos_Lapack))
        if 'AztecOO' in suppSolvers:
            available_la_solvers.append(("Trilinos AztecOO - Krylov (sparse, sequential, iterative)", laAztecOO))
        if 'daetools.solvers.trilinos' in sys.modules:
            del sys.modules['daetools.solvers.trilinos']
    except Exception as e:
        pass

    try:
        from daetools.solvers.intel_pardiso import pyIntelPardiso
        available_la_solvers.append(("Intel Pardiso (sparse, OpenMP, direct)", laIntelPardiso))
        if 'daetools.solvers.intel_pardiso' in sys.modules:
            del sys.modules['daetools.solvers.intel_pardiso']
    except Exception as e:
        print e
    
    return available_la_solvers

def getAvailableDataReporters():
    available_datareporters = []
    
    available_datareporters.append(("TCPIPDataReporter", TCPIPDataReporter))
    available_datareporters.append(("NoOpDataReporter", NoOpDataReporter))
    available_datareporters.append(("DelegateDataReporter", DelegateDataReporter))
    available_datareporters.append(("TEXTFileDataReporter", TEXTFileDataReporter))
    available_datareporters.append(("PlotDataReporter", PlotDataReporter))
    available_datareporters.append(("MatlabMATFileDataReporter", MatlabMATFileDataReporter))
    available_datareporters.append(("BlackHoleDataReporter", BlackHoleDataReporter))
    
    return available_datareporters
    
def createDataReporter(datareporterIndex):
    datareporter = None
    
    if datareporterIndex == BlackHoleDataReporter:
        datareporter = daeBlackHoleDataReporter()

    elif datareporterIndex == NoOpDataReporter:
        datareporter = daeNoOpDataReporter()

    elif datareporterIndex == DelegateDataReporter:
        datareporter = daeDelegateDataReporter()

    elif datareporterIndex == TEXTFileDataReporter:
        datareporter = daeTEXTFileDataReporter()

    elif datareporterIndex == TCPIPDataReporter:
        datareporter = daeTCPIPDataReporter()

    elif datareporterIndex == PlotDataReporter:
        datareporter = daePlotDataReporter()

    elif datareporterIndex == MatlabMATFileDataReporter:
        datareporter = daeMatlabMATFileDataReporter()
     
    else:
        raise RuntimeError("Unsupported Log selected")
    
    return datareporter

def getAvailableLogs():
    available_logs = []
    
    available_logs.append(("TCPIPLog", TCPIPLog))
    available_logs.append(("PythonStdOutLog", PythonStdOutLog))
    available_logs.append(("StdOutLog", StdOutLog))
    available_logs.append(("BaseLog", BaseLog))
    available_logs.append(("FileLog", FileLog))
    available_logs.append(("DelegateLog", DelegateLog))
    
    return available_logs
    
def createLog(logIndex):
    log = None
    
    if logIndex == BaseLog:
        log = daeBaseLog()

    elif logIndex == FileLog:
        log = daeFileLog('') # Log will create a temp file somewhere

    elif logIndex == DelegateLog:
        log = daeDelegateLog()

    elif logIndex == StdOutLog:
        log = daeStdOutLog()

    elif logIndex == TCPIPLog:
        log = daeTCPIPLog()

    elif logIndex == PythonStdOutLog:
        log = daePythonStdOutLog()
     
    else:
        raise RuntimeError("Unsupported Log selected")
    
    return log
   
def createLASolver(lasolverIndex):
    lasolver = None
    
    if lasolverIndex == laSundialsLU:
        pass

    elif lasolverIndex == laSuperLU:
        from daetools.solvers.superlu import pySuperLU
        lasolver = pySuperLU.daeCreateSuperLUSolver()

    elif lasolverIndex == laSuperLU_MT:
        from daetools.solvers.superlu_mt import pySuperLU_MT
        lasolver = pySuperLU_MT.daeCreateSuperLUSolver()
    
    elif lasolverIndex == laAmesos_Klu:
        from daetools.solvers.trilinos import pyTrilinos
        lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")

    elif lasolverIndex == laAmesos_Superlu:
        from daetools.solvers.trilinos import pyTrilinos
        lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Superlu", "")

    elif lasolverIndex == laAmesos_Umfpack:
        from daetools.solvers.trilinos import pyTrilinos
        lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Umfpack", "")

    elif lasolverIndex == laAmesos_Lapack:
        from daetools.solvers.trilinos import pyTrilinos
        lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Lapack", "")

    elif lasolverIndex == laAztecOO:
        from daetools.solvers.trilinos import pyTrilinos
        lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO", "ILUT")

    elif lasolverIndex == laIntelPardiso:
        from daetools.solvers.intel_pardiso import pyIntelPardiso
        lasolver = pyIntelPardiso.daeCreateIntelPardisoSolver()
    
    elif lasolverIndex == laIntelMKL:
        from daetools.solvers import pyIntelMKL
        lasolver = pyIntelMKL.daeCreateLapackSolver()

    elif lasolverIndex == laAmdACML:
        from daetools.solvers import pyAmdACML
        lasolver = pyAmdACML.daeCreateLapackSolver()

    elif lasolverIndex == laLapack:
        from daetools.solvers import pyLapack
        lasolver = pyLapack.daeCreateLapackSolver()

    elif lasolverIndex == laMagmaLapack:
        from daetools.solvers import pyMagma
        lasolver = pyMagma.daeCreateLapackSolver()
    
    elif lasolverIndex == laSuperLU_CUDA:
        from daetools.solvers.superlu_cuda import pySuperLU_CUDA
        lasolver = pySuperLU_CUDA.daeCreateSuperLUSolver()
    
    elif lasolverIndex == laCUSP:
        from daetools.solvers import pyCUSP
        lasolver = pyCUSP.daeCreateCUSPSolver()
    
    else:
        raise RuntimeError("Unsupported LA Solver selected")
    
    return lasolver

def createNLPSolver(minlpsolverIndex):
    nlpsolver = None
    
    if minlpsolverIndex == nlpIPOPT:
        from daetools.solvers.ipopt import pyIPOPT
        nlpsolver = pyIPOPT.daeIPOPT()

    elif minlpsolverIndex == nlpNLOPT:
        from daetools.solvers.nlopt import pyNLOPT
        from PyQt4 import QtCore, QtGui
        algorithms = ['NLOPT_GN_DIRECT','NLOPT_GN_DIRECT_L','NLOPT_GN_DIRECT_L_RAND','NLOPT_GN_DIRECT_NOSCAL','NLOPT_GN_DIRECT_L_NOSCAL',
                      'NLOPT_GN_DIRECT_L_RAND_NOSCAL','NLOPT_GN_ORIG_DIRECT','NLOPT_GN_ORIG_DIRECT_L','NLOPT_GD_STOGO','NLOPT_GD_STOGO_RAND',
                      'NLOPT_LD_LBFGS_NOCEDAL','NLOPT_LD_LBFGS','NLOPT_LN_PRAXIS','NLOPT_LD_VAR1','NLOPT_LD_VAR2','NLOPT_LD_TNEWTON',
                      'NLOPT_LD_TNEWTON_RESTART','NLOPT_LD_TNEWTON_PRECOND','NLOPT_LD_TNEWTON_PRECOND_RESTART','NLOPT_GN_CRS2_LM',
                      'NLOPT_GN_MLSL','NLOPT_GD_MLSL','NLOPT_GN_MLSL_LDS','NLOPT_GD_MLSL_LDS','NLOPT_LD_MMA','NLOPT_LN_COBYLA',
                      'NLOPT_LN_NEWUOA','NLOPT_LN_NEWUOA_BOUND','NLOPT_LN_NELDERMEAD','NLOPT_LN_SBPLX','NLOPT_LN_AUGLAG','NLOPT_LD_AUGLAG',
                      'NLOPT_LN_AUGLAG_EQ','NLOPT_LD_AUGLAG_EQ','NLOPT_LN_BOBYQA','NLOPT_GN_ISRES',
                      'NLOPT_AUGLAG','NLOPT_AUGLAG_EQ','NLOPT_G_MLSL','NLOPT_G_MLSL_LDS','NLOPT_LD_SLSQP']
        # Show the input box to choose the algorithm (the default is len(algorithms)-1 that is: NLOPT_LD_SLSQP)
        algorithm, ok = QtGui.QInputDialog.getItem(None, "NLOPT Algorithm", "Choose the NLOPT algorithm:", algorithms, len(algorithms)-1, False)
        if not ok:
            algorithm = 'NLOPT_LD_SLSQP'
        nlpsolver = pyNLOPT.daeNLOPT(str(algorithm))

    elif minlpsolverIndex == nlpBONMIN:
        from daetools.solvers.bonmin import pyBONMIN
        nlpsolver = pyBONMIN.daeBONMIN()

    else:
        raise RuntimeError("Unsupported (MI)NLP Solver selected")
    
    return nlpsolver
