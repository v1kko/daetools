include(../dae.pri)
QT -= core gui

TARGET = cdaeUnits
TEMPLATE = lib
CONFIG += shared plugin

# There is a problem compiling Units under MacOS X
# All optimization flags should be removed!!
macx-g++::QMAKE_CFLAGS_RELEASE   -= -O1 -O2 -O3
macx-g++::QMAKE_CXXFLAGS_RELEASE -= -O1 -O2 -O3

win32::QMAKE_CXXFLAGS += -DUNITS_EXPORTS

INCLUDEPATH += $${BOOSTDIR} 

LIBS += $${SOLIBS_RPATH}
LIBS +=	$${BOOST_LIBS}

SOURCES += dllmain.cpp \
    units.cpp \
    stdafx.cpp \
    quantity.cpp

HEADERS += units.h \
    stdafx.h \
    units_pool.h

#######################################################
#                Install files
#######################################################
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

include(../dae_install_library.pri)
