include(../dae.pri)

QT -= core
QT -= gui

TEMPLATE = lib
CONFIG += shared

TARGET  = cdaeSimulationLoader

SOURCES += dllmain.cpp \
           simulation_loader.cpp

HEADERS += stdafx.h \
           simulation_loader.h \
           simulation_loader_c.h

message($$system($${PYTHON}-config --cflags))
message($$system($${PYTHON}-config --ldflags))

QMAKE_CXXFLAGS += $$system($${PYTHON}-config --cflags)
#QMAKE_LFLAGS  += $$system($${PYTHON}-config --ldflags)

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

unix::QMAKE_POST_LINK = $${COPY_FILE} \
                        $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}* \
                        $${SOLIBS_DIR}
win32::QMAKE_POST_LINK = $${COPY_FILE} \
                         $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}* \
                         $${SOLIBS_DIR}/$${TARGET}.$${SHARED_LIB_APPEND}
