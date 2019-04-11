include(../dae.pri)
QT -= core \
    gui
TARGET = pyDataReporting
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH}

LIBS +=	$${DAE_DATAREPORTING_LIB} \
        $${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
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
#                  $${PYDAE_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

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
