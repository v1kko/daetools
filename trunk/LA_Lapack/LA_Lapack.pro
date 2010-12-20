include(../dae.pri)
QT -= core \
    gui
TARGET = Lapack
TEMPLATE = lib

####################################################################################
#                       Suppress some warnings
####################################################################################
#unix::QMAKE_CXXFLAGS += -ansi -pedantic
unix::QMAKE_CXXFLAGS += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
unix::QMAKE_CFLAGS   += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable

######################################################################################
#                                 SOLVER SPECIFICATION
# IntelMKL, AmdACML, or Lapack   
######################################################################################
CONFIG += IntelMKL

######################################################################################
#                                 INTEL MKL solver
######################################################################################
# LD_LIBRARY_PATH should be set
# MKL_NUM_THREADS=Ncpu should be set
# http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
#####################################################################################
IntelMKL{
QMAKE_CXXFLAGS += -DdaeHasIntelMKL
win32-msvc2008::LAPACK_PATH = c:\Intel\MKL\10.2.5.035
linux-g++::LAPACK_PATH = /opt/intel/mkl/10.2.5.035
linux-g++-64::LAPACK_PATH = /opt/intel/mkl/10.2.5.035

win32-msvc2008::INTEL_LIBS = $${LAPACK_PATH}\ia32\lib
linux-g++::INTEL_LIBS = $${LAPACK_PATH}/lib/32
linux-g++-64::INTEL_LIBS = $${LAPACK_PATH}/lib/em64t

LAPACK_INCLUDE = $${LAPACK_PATH}/include

win32-msvc2008::LAPACK_LIBS =  -L$${INTEL_LIBS} mkl_solver.lib mkl_intel_c.lib mkl_intel_thread.lib mkl_core.lib libiomp5mt.lib -Qopenmp

linux-g++::LAPACK_LIBS = -L$${INTEL_LIBS} \
						$${INTEL_LIBS}/libmkl_solver.a \
						-Wl,--start-group \
						  $${INTEL_LIBS}/libmkl_intel.a \
						  $${INTEL_LIBS}/libmkl_intel_thread.a \
						  $${INTEL_LIBS}/libmkl_core.a \
						-Wl,--end-group \
						-liomp5 -lpthread -lgfortran

linux-g++-64::LAPACK_LIBS = -L$${INTEL_LIBS} \
							$${INTEL_LIBS}/libmkl_solver_lp64.a \
							-Wl,--start-group \
							  $${INTEL_LIBS}/libmkl_intel_lp64.a \
							  $${INTEL_LIBS}/libmkl_intel_thread.a \
							  $${INTEL_LIBS}/libmkl_core.a \
							-Wl,--end-group \
							-liomp5 -lpthread -lgfortran

pyObject = pyIntelMKL
}

######################################################################################
#                                AMD ACML solver
######################################################################################
AmdACML{
QMAKE_CXXFLAGS += -DdaeHasAmdACML
win32-msvc2008::LAPACK_PATH = c:\AMD\acml4.4.0\ifort32_mp
linux-g++::LAPACK_PATH = /opt/acml4.4.0/gfortran32_mp
linux-g++-64::LAPACK_PATH = /opt/acml4.4.0/gfortran64_mp

AMD_LIBS = $${LAPACK_PATH}/lib
LAPACK_INCLUDE = $${LAPACK_PATH}/include

win32-msvc2008::LAPACK_LIBS =$${AMD_LIBS}/libacml_mp_dll.lib
linux-g++::LAPACK_LIBS = -L$${AMD_LIBS} -lacml_mp -lgfortran -lgomp -lrt -lm
linux-g++-64::LAPACK_LIBS = -L$${AMD_LIBS} -lacml_mp -lgfortran -lgomp -lrt -lm

pyObject = pyAmdACML
}

######################################################################################
#                              Generic Lapack solver
######################################################################################
Lapack{
QMAKE_CXXFLAGS += -DdaeHasLapack
win32-msvc2008::LAPACK_PATH = 
unix::LAPACK_PATH =

win32-msvc2008::LAPACK_INCLUDE =
unix::LAPACK_INCLUDE =

win32-msvc2008::LAPACK_LIBS = 
unix::LAPACK_LIBS = -lblas -llapack -lgfortran

pyObject = pyLapack
}

######################################################################################
#                                    Atlas solver
######################################################################################
Atlas{
QMAKE_CXXFLAGS += -DdaeHasAtlas
win32-msvc2008::LAPACK_PATH = 
unix::LAPACK_PATH =

win32-msvc2008::LAPACK_INCLUDE =
unix::LAPACK_INCLUDE =

win32-msvc2008::LAPACK_LIBS = 
unix::LAPACK_LIBS = -L/usr/lib/atlas-amd64sse3 \
					/usr/lib/atlas-amd64sse3/libptcblas.a \ 
					/usr/lib/atlas-amd64sse3/libatlas.a


pyObject = pyAtlas
}

INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR} \
    $${SUNDIALS_INCLUDE} \
    $${LAPACK_INCLUDE}
QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS += $${BOOST_PYTHON_LIB} \
    $${LAPACK_LIBS}
SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    lapack_la_solver.cpp
HEADERS += stdafx.h \
    lapack_la_solver.h

IntelMKL{
win32-msvc2008::QMAKE_POST_LINK =  move /y \
    $${DAE_DEST_DIR}/Lapack1.dll \
    $${DAE_DEST_DIR}/pyIntelMKL.pyd
}

AmdACML{
win32-msvc2008::QMAKE_POST_LINK =  move /y \
    $${DAE_DEST_DIR}/Lapack1.dll \
    $${DAE_DEST_DIR}/pyAmdACML.pyd
}

Lapack{
win32-msvc2008::QMAKE_POST_LINK =  move /y \
    $${DAE_DEST_DIR}/Lapack1.dll \
    $${DAE_DEST_DIR}/pyLapack.pyd
}

Atlas{
win32-msvc2008::QMAKE_POST_LINK =  move /y \
    $${DAE_DEST_DIR}/Lapack1.dll \
    $${DAE_DEST_DIR}/pyAtlas.pyd
}

unix::QMAKE_POST_LINK = cp \
            -f \
            $${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
            $${DAE_DEST_DIR}/$${pyObject}.so
