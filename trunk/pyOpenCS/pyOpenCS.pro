include(../dae.pri)
QT -= core \
    gui
TARGET = pyOpenCS
TEMPLATE = lib
CONFIG += shared plugin

unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11

OPEN_CS_DIR     = ../OpenCS/build
OPEN_CS_INCLUDE = $${OPEN_CS_DIR}/include
OPEN_CS_LIBS    = -L$${OPEN_CS_DIR}/lib -lOpenCS_Evaluators -lOpenCS_Models -lOpenCS_Simulators

unix::MPI_LIBS  = -lmpi_cxx -lmpi

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH}

LIBS +=	$${OPEN_CS_LIBS} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

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
message(install_python_module.path = $$install_python_module.path)
message(install_python_module.files = $$install_python_module.files)

# Install into daetools-package
install_python_module2.depends += install_rename_module
install_python_module2.path     = $${SOLVERS_DIR}
install_python_module2.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# For some reasons INSTALLS was ignored without target rule.
target.depends += install_rename_module
target.path     = $${DAE_DEST_DIR}
target.extra    = @echo Installing $${TARGET} # do nothing, overriding the default behaviour

INSTALLS += target install_python_module install_python_module2
