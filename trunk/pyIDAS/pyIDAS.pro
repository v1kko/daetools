include(../dae.pri)
QT -= core \
	gui
TARGET = pyIDAS
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NUMPY_INCLUDE_DIR} \
               $${SUNDIALS_INCLUDE} \
	           $${MPI_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${SUNDIALS_LIBDIR}

LIBS +=	$${DAE_IDAS_SOLVER_LIB} \
        $${DAE_UNITS_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${MPI_LIBS}
        
SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyIDAS1.dll \
	$${DAE_DEST_DIR}/pyIDAS.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
        $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
        $${DAE_DEST_DIR}/$${TARGET}.so
}
