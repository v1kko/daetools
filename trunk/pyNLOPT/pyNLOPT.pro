include(../dae.pri)
QT -= core gui
TARGET = pyNLOPT
TEMPLATE = lib
CONFIG += shared

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               #$${NUMPY_INCLUDE_DIR} \
               $${NLOPT_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${NLOPT_LIBDIR}

LIBS += $${SOLIBS_RPATH}
                
LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_NLOPT_SOLVER_LIB} \
        $${DAE_CONFIG_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${NLOPT_LIBS} \
        $${BLAS_LAPACK_LIBS}        

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
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_APPEND} \
                  $${SOLVERS_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}
                  
# win32{
# QMAKE_POST_LINK = move /y \
# 	$${DAE_DEST_DIR}/pyNLOPT1.dll \
# 	$${SOLVERS_DIR}/pyNLOPT.pyd
# }
# 
# unix{
# QMAKE_POST_LINK = cp -f \
#         $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
#         $${SOLVERS_DIR}/$${TARGET}.so
# }
