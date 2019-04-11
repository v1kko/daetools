include(../dae.pri)

QT -= core
QT -= gui

TEMPLATE = lib
# Add "plugin" to avoid creation of symlinks
CONFIG += shared plugin

win32-msvc2015::QMAKE_CXXFLAGS += -DDAE_DLL_EXPORTS

TARGET  = cdaeSimulationLoader-py$${PYTHON_MAJOR}$${PYTHON_MINOR}

SOURCES += dllmain.cpp \
           simulation_loader.cpp \
           simulation_loader_c.cpp

HEADERS += stdafx.h \
           simulation_loader.h \
           simulation_loader_c.h \
           simulation_loader_common.h

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH_SL}

LIBS += $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

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
