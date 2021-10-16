include(../dae.pri)

QT -= core gui
TARGET = cdaeActivity$${SHARED_LIB_POSTFIX}
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR} \
               $${OPEN_CS_INCLUDE}

win32::QMAKE_CXXFLAGS += -DACTIVITY_EXPORTS

LIBS += $${SOLIBS_RPATH_SL}

LIBS += $${DAE_CORE_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
        $${BOOST_LIBS}
LIBS += $${STD_FILESYSTEM}

HEADERS += stdafx.h \
    simulation.h \
    activity_class_factory.h \
    base_activities.h \
    ../mmio.h

SOURCES += stdafx.cpp \
    simulation.cpp \
    dllmain.cpp \
    class_factory.cpp \
    base_activities.cpp \
    optimization.cpp \
    ../mmio.c

#######################################################
#                Install files
#######################################################
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}.$${SHARED_LIB_EXT} \
                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}.$${SHARED_LIB_EXT}

include(../dae_install_library.pri)
QMAKE_CXXFLAGS += -fpermissive
