include(../dae.pri)
QT -= core gui
TEMPLATE = lib
CONFIG += shared plugin

###############################
# Could be: BONMIN, IPOPT
###############################
CONFIG += IPOPT

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

LIBS += $${SOLIBS_RPATH}

######################################################################################
#                                BONMIN
######################################################################################
CONFIG(BONMIN, BONMIN|IPOPT):message(BONMIN) {

QMAKE_CXXFLAGS += -DdaeBONMIN
TARGET = pyBONMIN

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${BONMIN_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${BONMIN_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${SOLIBS_RPATH}
LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_BONMIN_SOLVER_LIB} \
        $${DAE_CONFIG_LIB} \
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
TARGET = pyIPOPT

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${IPOPT_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${IPOPT_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${SOLIBS_RPATH}
LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_IPOPT_SOLVER_LIB} \
        $${DAE_CONFIG_LIB} \
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
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLVERS_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# Rename libpyModule.so into pyModule.so
install_rename_module.commands = $${MOVE_FILE} \
                                 $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                                 $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}
QMAKE_EXTRA_TARGETS += install_rename_module

# Install into daetools-dev
install_python_module.depends += install_rename_module
install_python_module.path     = $${DAE_INSTALL_PY_MODULES_DIR}
install_python_module.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# Install into daetools-package
install_python_module2.depends += install_rename_module
install_python_module2.path     = $${SOLVERS_DIR}
install_python_module2.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

INSTALLS += install_python_module install_python_module2
