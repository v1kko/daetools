#import os, sys, platform

#PYTHON_MAJOR = str(sys.version_info[0])
#PYTHON_MINOR = str(sys.version_info[1])

## System := {'Linux', 'Windows', 'Darwin'}
#DAE_SYSTEM   = str(platform.system())

## Machine := {'i386', ..., 'i686', 'AMD64'}
#DAE_MACHINE  = str(platform.machine())

#solvers_sodir = os.path.join(os.path.dirname(__file__), '{0}_{1}_py{2}{3}'.format(DAE_SYSTEM, DAE_MACHINE, PYTHON_MAJOR, PYTHON_MINOR))
#sys.path.append(solvers_sodir)

#try:
    #import pySuperLU
#except ImportError as e:
    #pass

#try:
    #import pySuperLU_MT
#except ImportError as e:
    #pass

#try:
    #import pyTrilinos
#except ImportError as e:
    #pass

#try:
    #import pyIPOPT
#except ImportError as e:
    #pass

#try:
    #import pyBONMIN
#except ImportError as e:
    #pass

#try:
    #import pyNLOPT
#except ImportError as e:
    #pass
