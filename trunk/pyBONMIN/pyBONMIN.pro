include(../dae.pri)
QT -= core \
	gui
TARGET = pyBIMINLP
TEMPLATE = lib

CONFIG += BONMIN

######################################################################################
#                                BONMIN
######################################################################################
BONMIN { 
QMAKE_CXXFLAGS += -DdaeBONMIN
pyObject = pyBONMIN

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${BONMIN_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${BONMIN_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_SIMULATION_LIB} \
        $${DAE_DATAREPORTERS_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_SOLVER_LIB} \
        $${DAE_BONMINSOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${BONMIN_LIBS} \
        $${MUMPS_LIBS}
}

######################################################################################
#                                IPOPT
######################################################################################
IPOPT { 
QMAKE_CXXFLAGS += -DdaeIPOPT
pyObject = pyIPOPT

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${IPOPT_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${IPOPT_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_SIMULATION_LIB} \
        $${DAE_DATAREPORTERS_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_SOLVER_LIB} \
        $${DAE_IPOPTSOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${IPOPT_LIBS} \
        $${MUMPS_LIBS}
}


SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

BONMIN { 
win32-msvc2008::QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyBIMINLP1.dll \
	$${DAE_DEST_DIR}/pyBONMIN.pyd
}

IPOPT { 
win32-msvc2008::QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyBIMINLP1.dll \
	$${DAE_DEST_DIR}/pyIPOPT.pyd
}

unix::QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/$${pyObject}.so
