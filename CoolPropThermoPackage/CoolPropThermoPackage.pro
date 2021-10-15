include(../dae.pri)
QT -= core
QT -= gui

TARGET = cdaeCoolPropThermoPackage$${SHARED_LIB_POSTFIX}
TEMPLATE = lib
CONFIG += shared plugin

win32-msvc2015::QMAKE_CXXFLAGS += -DCOOL_PROP_EXPORTS

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

include(../dae_install_library.pri)
INSTALLS += $${COOLPROP_LIBS}
