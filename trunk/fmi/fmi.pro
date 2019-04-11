include(../dae.pri)

QT -= core
QT -= gui

TEMPLATE = lib
# Add "plugin" to avoid creation of symlinks
CONFIG += shared plugin

TARGET = cdaeFMU_CS-py$${PYTHON_MAJOR}$${PYTHON_MINOR}

unix::QMAKE_CXXFLAGS += -fvisibility=hidden

SOURCES += dllmain.cpp \
           daetools_fmi_cs.cpp

HEADERS += stdafx.h \
           daetools_fmi_cs.h

INCLUDEPATH += $${BOOSTDIR}

LIBS += $${SOLIBS_RPATH_SL}
LIBS += $${DAE_SIMULATION_LOADER_LIB}
LIBS += $${BOOST_LIBS}

# Achtung, Achtung!!
# It uses daetools/solibs for linking the config, simulation_loader libraries
# since the lib names contain major number at the end of the filename and can't be found
QMAKE_LIBDIR += $${SOLIBS_DIR}

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${FMI_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install headers and libs into daetools-dev
DAE_PROJECT_NAME = $$basename(PWD)

install_headers.path  = $${DAE_INSTALL_HEADERS_DIR}/$${DAE_PROJECT_NAME}
install_headers.files = *.h

install_libs.path  = $${DAE_INSTALL_LIBS_DIR}
install_libs.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install into daetools-package
install_py_solib.path  = $${FMI_DIR}
install_py_solib.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

INSTALLS += install_headers install_libs install_py_solib
