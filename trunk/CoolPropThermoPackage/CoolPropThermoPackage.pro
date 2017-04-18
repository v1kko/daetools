include(../dae.pri)
QT -= core
QT -= gui

TARGET = cdaeCoolPropThermoPackage
TEMPLATE = lib
CONFIG += staticlib

win32-msvc2015::QMAKE_CXXFLAGS += -DDAE_DLL_EXPORTS

INCLUDEPATH += $${BOOSTDIR} \
               $${COOLPROP_INCLUDE}

LIBS +=	$${COOLPROP_LIBS}

SOURCES += cool_prop.cpp \
    stdafx.cpp \
    dllmain.cpp

HEADERS += cool_prop.h \
    coolprop_thermo.h \
    stdafx.h

#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}
