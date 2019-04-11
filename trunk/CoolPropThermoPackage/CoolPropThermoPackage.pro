include(../dae.pri)
QT -= core
QT -= gui

TARGET = cdaeCoolPropThermoPackage
TEMPLATE = lib
CONFIG += shared plugin

win32-msvc2015::QMAKE_CXXFLAGS += -DDAE_DLL_EXPORTS

INCLUDEPATH += $${BOOSTDIR} \
               $${COOLPROP_INCLUDE}

LIBS += $${SOLIBS_RPATH_SL}
LIBS +=	$${COOLPROP_LIBS} \
        $${BOOST_LIBS}

SOURCES += cool_prop.cpp \
    stdafx.cpp \
    dllmain.cpp

HEADERS += cool_prop.h \
    coolprop_thermo.h \
    stdafx.h

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
install_py_solib.path  = $${SOLIBS_DIR}
install_py_solib.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

INSTALLS += install_headers install_libs install_py_solib
