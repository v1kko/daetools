include(../dae.pri)
QT -= core \
    gui
TARGET = pyActivity
TEMPLATE = lib
CONFIG += shared

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${MPI_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH}

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_NLPSOLVER_LIB} \
        $${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${MPI_LIBS}

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
                  $${PYDAE_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# win32{
# QMAKE_POST_LINK = move /y \
# 	$${DAE_DEST_DIR}/pyActivity1.dll \
# 	$${PYDAE_DIR}/pyActivity.pyd
# }
#
# unix{
# QMAKE_POST_LINK = cp -f \
#         $${DAE_DEST_DIR}/lib$${TARGET}.$${SHARED_LIB_APPEND} \
#         $${PYDAE_DIR}/$${TARGET}.so
# }
