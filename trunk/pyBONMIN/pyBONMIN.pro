include(../dae.pri)
QT -= core \
	gui
TARGET = pyBONMIN
TEMPLATE = lib

###############################
# Could be: BONMIN, IPOPT
###############################
CONFIG += BONMIN

#####################################################################
# Small hack used when compiling from compile_linux.sh shell script
#####################################################################
shellCompile:message(shellCompile) {
shellIPOPT {
  CONFIG += IPOPT
}
shellBONMIN {
  CONFIG += BONMIN
}
}

######################################################################################
#                                BONMIN
######################################################################################
CONFIG(BONMIN, BONMIN|IPOPT):message(BONMIN) { 

QMAKE_CXXFLAGS += -DdaeBONMIN
pyObject = pyBONMIN

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NUMPY_INCLUDE_DIR} \
               $${BONMIN_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${BONMIN_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_BONMIN_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${BONMIN_LIBS} \
        $${MUMPS_LIBS} \
        $${BLAS_LAPACK_LIBS}
}

######################################################################################
#                                IPOPT
######################################################################################
CONFIG(IPOPT, BONMIN|IPOPT):message(IPOPT) { 

QMAKE_CXXFLAGS += -DdaeIPOPT
pyObject = pyIPOPT

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NUMPY_INCLUDE_DIR} \
               $${IPOPT_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${IPOPT_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_IPOPT_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${IPOPT_LIBS} \
        $${MUMPS_LIBS} \
        $${BLAS_LAPACK_LIBS}
}


SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    docstrings.h \
    python_wraps.h

#######################################################
#                Install files
#######################################################
win32{
BONMIN { 
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyBONMIN1.dll \
	$${SOLVERS_DIR}/pyBONMIN.pyd
}

IPOPT { 
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyBONMIN1.dll \
	$${SOLVERS_DIR}/pyIPOPT.pyd
}
}

unix{
QMAKE_POST_LINK = cp -f \
        $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
        $${SOLVERS_DIR}/$${pyObject}.so
}
